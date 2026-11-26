#!/usr/bin/env python3
"""Run a queue of steps on the GPU machine itself, so no GPU waits for someone to launch the next job.

Runs on the pod (standard library only), detached, for the whole session:

    nohup setsid python3 pod/run_queue.py data/<queue>.jsonl --gpus 0 1 --slots-per-gpu 2 \
        > data/<queue>_runner.log 2>&1 &

The queue file has one JSON object per line (blank lines and lines starting with # are ignored):

    {"name": "pilot_session_one_generate_screening_base_selection_20261127",
     "command": "python scripts/generate_responses.py ... --output-dir \"$JOB_OUTPUT_DIR/...\"",
     "gpu": "any",                 "any" (default), a GPU index, or null for a CPU-only step
     "slots": 1,                   GPU slots the step occupies (default 1; slots-per-gpu = the whole card)
     "needs": ["<step name>", ...],          steps that must have finished successfully first
     "needs_files": ["data/....json", ...],  paths (relative to the repository root, or absolute) that
                                             must exist; glob patterns such as snapshots/*/config.json allowed
     "min_free_gib": 40,           free GPU memory this step needs before it starts (default: --min-free-gib)
     "retries": 1,                 extra attempts after a failure (default: --retries)
     "env": {"NAME": "value"}}

Rules the runner applies without anyone watching:

  * A step starts as soon as everything it needs is done and a slot is free. "any" steps go to
    whichever GPU frees up first, so the work balances itself.
  * Two engines starting on one GPU at the same moment can both under-count free memory, so a GPU
    takes a new step only --start-gap-seconds after its last start, and only if nvidia-smi shows at
    least --min-free-gib free on it.
  * A failed step is retried (its command must resume from what it already wrote). When its
    retries are used up it is marked failed, every step that needs it is marked blocked, and
    everything else carries on.
  * When a step's shell exits, every process still carrying its JOB_RUN_ID is killed. vLLM's engine
    process can outlive its parent and keep the GPU's memory; this frees it.
  * The queue file is read again on every poll: steps appended later (by hand, by sync, or by a
    step that plans the next stage) join the queue. A name already seen keeps its first definition.
  * Restarting the runner resumes: a step whose folder has job.exit_code 0 counts as done.

Each step runs in data/<name>/ with the same files as pod/jobs.py (job.sh, job.log, job.pid,
job.exit_code) and the variables CUDA_VISIBLE_DEVICES, JOB_NAME and JOB_OUTPUT_DIR. The runner
writes data/<queue>_state.json after every change and prints one timestamped line per event.
It stops when the file data/<queue>.stop appears (running steps are stopped too), or, with
--exit-when-done, when every step is done, failed or blocked.
"""
import argparse
import glob
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
ENV_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
FINAL_STATES = ("done", "failed", "blocked")


def log(message):
    print(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), message, flush=True)


def parse_step(raw, default_retries):
    name, command = raw["name"], raw["command"]
    if not isinstance(name, str) or not NAME_PATTERN.match(name):
        raise ValueError(f"bad step name {name!r}")
    if not isinstance(command, str) or not command.strip():
        raise ValueError("command must be a non-empty string")
    gpu = raw.get("gpu", "any")
    if not (gpu is None or gpu == "any" or (isinstance(gpu, int) and not isinstance(gpu, bool) and gpu >= 0)):
        raise ValueError(f"gpu must be \"any\", a GPU index or null: {gpu!r}")
    slots = raw.get("slots", 1)
    if not isinstance(slots, int) or isinstance(slots, bool) or slots < 1:
        raise ValueError("slots must be a positive integer")
    needs, needs_files = raw.get("needs", []), raw.get("needs_files", [])
    if not isinstance(needs, list) or not isinstance(needs_files, list) or \
            not all(isinstance(n, str) for n in needs + needs_files):
        raise ValueError("needs and needs_files must be lists of strings")
    retries = raw.get("retries", default_retries)
    if not isinstance(retries, int) or isinstance(retries, bool) or retries < 0:
        raise ValueError("retries must be a non-negative integer")
    min_free = raw.get("min_free_gib")
    if min_free is not None and (isinstance(min_free, bool) or not isinstance(min_free, (int, float)) or min_free < 0):
        raise ValueError("min_free_gib must be a non-negative number")
    env = raw.get("env", {})
    if not isinstance(env, dict) or not all(ENV_NAME_PATTERN.match(k) for k in env):
        raise ValueError("env must be an object with valid variable names")
    return {"name": name, "command": command, "gpu": gpu, "slots": slots, "needs": list(needs),
            "needs_files": list(needs_files), "retries": retries, "min_free_gib": min_free, "env": {k: str(v) for k, v in env.items()}}


def read_queue(path, default_retries):
    """Steps in file order. Bad lines are reported and skipped (the file may be mid-append)."""
    steps, errors = [], []
    try:
        text = Path(path).read_text()
    except FileNotFoundError:
        return steps, [f"queue file {path} not found"]
    lines = text.split("\n")
    if lines and lines[-1] != "":
        lines = lines[:-1]  # a last line without its newline is still being written
    for number, line in enumerate(lines, 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            steps.append(parse_step(json.loads(line), default_retries))
        except (KeyError, ValueError, TypeError) as error:
            errors.append(f"{path}:{number}: {error}")
    return steps, errors


class Scheduler:
    """The scheduling decisions, kept free of processes and clocks so they can be tested."""

    def __init__(self, gpus, slots_per_gpu):
        self.gpus, self.slots_per_gpu = list(gpus), slots_per_gpu
        self.steps, self.order = {}, []
        self.state, self.attempts, self.placed = {}, {}, {}

    def add(self, step):
        if step["name"] in self.steps:
            return False
        if step["gpu"] not in (None, "any") and step["gpu"] not in self.gpus:
            raise ValueError(f"step {step['name']} asks for GPU {step['gpu']}, which this runner does not use")
        if step["gpu"] is not None and step["slots"] > self.slots_per_gpu:
            raise ValueError(f"step {step['name']} needs {step['slots']} slots; a GPU has {self.slots_per_gpu}")
        self.steps[step["name"]] = step
        self.order.append(step["name"])
        self.state[step["name"]] = "waiting"
        self.attempts[step["name"]] = 0
        return True

    def used_slots(self, gpu):
        return sum(self.steps[n]["slots"] for n, g in self.placed.items() if g == gpu)

    def propagate_blocks(self):
        """Mark every waiting step blocked whose needs include a failed, blocked or unknown-and-never step."""
        newly, changed = [], True
        while changed:
            changed = False
            for name in self.order:
                if self.state[name] != "waiting":
                    continue
                bad = [n for n in self.steps[name]["needs"] if self.state.get(n) in ("failed", "blocked")]
                if bad:
                    self.state[name] = "blocked"
                    newly.append((name, bad))
                    changed = True
        return newly

    def ready(self, file_exists):
        """Waiting steps whose needs are all done and whose needed files exist, in queue order."""
        out = []
        for name in self.order:
            step = self.steps[name]
            if self.state[name] != "waiting":
                continue
            if all(self.state.get(n) == "done" for n in step["needs"]) and all(file_exists(f) for f in step["needs_files"]):
                out.append(name)
        return out

    def choose_gpu(self, name, gpu_available):
        """The GPU to start this step on now, None for a CPU step, or False if it must wait.
        gpu_available(gpu, step) says whether the GPU may take this step now (start gap, free memory)."""
        step = self.steps[name]
        if step["gpu"] is None:
            return None
        candidates = self.gpus if step["gpu"] == "any" else [step["gpu"]]
        fits = [g for g in candidates if self.used_slots(g) + step["slots"] <= self.slots_per_gpu and gpu_available(g, step)]
        if not fits:
            return False
        return min(fits, key=lambda g: (self.used_slots(g), self.gpus.index(g)))  # emptiest first

    def start(self, name, gpu):
        self.state[name] = "running"
        self.attempts[name] += 1
        if gpu is not None:
            self.placed[name] = gpu

    def finish(self, name, exit_code):
        """Record an exit. Returns the new state: done, waiting (to be retried) or failed."""
        self.placed.pop(name, None)
        if exit_code == 0:
            self.state[name] = "done"
        elif self.attempts[name] <= self.steps[name]["retries"]:
            self.state[name] = "waiting"
        else:
            self.state[name] = "failed"
        return self.state[name]

    def all_final(self):
        return bool(self.order) and all(self.state[n] in FINAL_STATES for n in self.order)

    def counts(self):
        out = {}
        for n in self.order:
            out[self.state[n]] = out.get(self.state[n], 0) + 1
        return out


def build_step_script(root, step, out, run_id):
    exports = {"JOB_NAME": step["name"], "JOB_OUTPUT_DIR": str(out), "JOB_RUN_ID": run_id, **step["env"]}
    lines = [f"export {k}={shlex.quote(v)}" for k, v in exports.items()]
    lines += [f"cd {shlex.quote(str(root))}", step["command"], "status=$?",
              f"echo $status > {shlex.quote(str(out))}/job.exit_code",
              'echo "[job finished with exit code $status]"', "exit $status"]
    return "\n".join(lines) + "\n"


def processes_with_run_id(run_id, proc="/proc"):
    """Pids of processes whose environment carries JOB_RUN_ID=<run_id> (Linux /proc)."""
    needle = f"JOB_RUN_ID={run_id}".encode()
    pids = []
    for entry in os.listdir(proc) if os.path.isdir(proc) else []:
        if not entry.isdigit() or int(entry) == os.getpid():
            continue
        try:
            with open(os.path.join(proc, entry, "environ"), "rb") as f:
                if needle in f.read().split(b"\0"):
                    pids.append(int(entry))
        except OSError:
            continue
    return pids


def kill_run(run_id, grace_seconds=10):
    pids = processes_with_run_id(run_id)
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid in pids:
            try:
                os.kill(pid, sig)
            except OSError:
                pass
        deadline = time.time() + (grace_seconds if sig == signal.SIGTERM else 2)
        while time.time() < deadline:
            pids = processes_with_run_id(run_id)
            if not pids:
                return 0
            time.sleep(0.5)
    return len(processes_with_run_id(run_id))


def read_exit_code(folder):
    try:
        return int(Path(folder, "job.exit_code").read_text().strip())
    except (OSError, ValueError):
        return None


def alive_pid(folder):
    try:
        pid = int(Path(folder, "job.pid").read_text().strip())
        os.kill(pid, 0)
        return pid
    except (OSError, ValueError):
        return None


def free_gib(gpu):
    try:
        done = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits",
                               "-i", str(gpu)], capture_output=True, text=True, timeout=30)
        return float(done.stdout.strip().splitlines()[0]) / 1024
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return None  # unknown: do not block on it


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("queue_file")
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[1]), help="repository root on the pod")
    ap.add_argument("--gpus", nargs="+", type=int, required=True)
    ap.add_argument("--slots-per-gpu", type=int, default=1)
    ap.add_argument("--start-gap-seconds", type=float, default=90.0,
                    help="minimum time between two starts on one GPU (engines profile memory at start)")
    ap.add_argument("--min-free-gib", type=float, default=30.0, help="free memory a GPU needs to take a new step")
    ap.add_argument("--retries", type=int, default=1, help="default extra attempts after a failure")
    ap.add_argument("--poll-seconds", type=float, default=10.0)
    ap.add_argument("--exit-when-done", action="store_true")
    a = ap.parse_args(argv)

    root = Path(a.root).resolve()
    queue = Path(a.queue_file) if Path(a.queue_file).is_absolute() else root / a.queue_file
    stem = queue.with_suffix("")
    state_path, stop_path = Path(f"{stem}_state.json"), Path(f"{stem}.stop")
    data = root / "data"
    sched = Scheduler(a.gpus, a.slots_per_gpu)
    running = {}          # name -> {"process", "run_id", "started", "gpu"}
    last_start = {g: 0.0 for g in a.gpus}
    history, seen_errors = [], set()
    started_at = time.time()
    log(f"runner started: queue={queue} gpus={a.gpus} slots_per_gpu={a.slots_per_gpu} pid={os.getpid()}")

    def write_state():
        now = time.time()
        steps = []
        for n in sched.order:
            s = sched.steps[n]
            item = {"name": n, "state": sched.state[n], "attempts": sched.attempts[n], "gpu": s["gpu"]}
            if n in running:
                item["running_on"] = running[n]["gpu"]
                item["running_seconds"] = round(now - running[n]["started"])
            steps.append(item)
        tmp = state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"updated_unix_time": now, "runner_pid": os.getpid(),
                                   "uptime_seconds": round(now - started_at), "counts": sched.counts(),
                                   "steps": steps, "history": history[-200:]}, indent=1))
        os.replace(tmp, state_path)

    def event(kind, name, **extra):
        history.append({"time": time.time(), "event": kind, "step": name, **extra})
        log(f"{kind} {name} " + " ".join(f"{k}={v}" for k, v in extra.items()))

    def stop_all():
        for n, r in list(running.items()):
            if r["process"] is not None:
                try:  # the step's whole session (start_new_session), then anything that left it
                    os.killpg(r["process"].pid, signal.SIGTERM)
                except OSError:
                    pass
            if r["run_id"]:
                kill_run(r["run_id"])
            elif alive_pid(r["folder"]) is not None:
                try:
                    os.killpg(alive_pid(r["folder"]), signal.SIGTERM)
                except OSError:
                    pass
            event("stopped", n)
        running.clear()

    while True:
        changed = False
        steps, errors = read_queue(queue, a.retries)
        for e in errors:
            if e not in seen_errors:
                seen_errors.add(e)
                log(f"queue error (line skipped): {e}")
        for s in steps:
            try:
                if sched.add(s):
                    changed = True
                    folder = data / s["name"]
                    code = read_exit_code(folder)
                    pid = alive_pid(folder)
                    if code == 0:
                        sched.state[s["name"]] = "done"
                        event("already_done", s["name"])
                    elif pid is not None and code is None:
                        # Started by an earlier runner and still going: adopt it instead of starting a copy.
                        gpu = s["gpu"] if isinstance(s["gpu"], int) else None
                        sched.start(s["name"], gpu)
                        running[s["name"]] = {"process": None, "pid": pid, "run_id": None,
                                              "started": time.time(), "gpu": gpu, "folder": folder}
                        event("adopted", s["name"], pid=pid)
            except ValueError as error:
                if str(error) not in seen_errors:
                    seen_errors.add(str(error))
                    log(f"queue error (step skipped): {error}")

        # Reap finished steps.
        for n, r in list(running.items()):
            if r["process"] is None:  # adopted: watch its pid and exit-code file
                if alive_pid(r["folder"]) is not None:
                    continue
                code = read_exit_code(r["folder"])
                code = -1 if code is None else code
                leftovers = 0
            else:
                code = r["process"].poll()
                if code is None:
                    continue
                leftovers = len(processes_with_run_id(r["run_id"]))
                if leftovers:
                    kill_run(r["run_id"])
            del running[n]
            new_state = sched.finish(n, code)
            event("finished", n, exit_code=code, seconds=round(time.time() - r["started"]),
                  state=new_state, attempt=sched.attempts[n], killed_leftover_processes=leftovers)
            changed = True
        for name, bad in sched.propagate_blocks():
            event("blocked", name, because=",".join(bad))
            changed = True

        if stop_path.exists():
            log(f"stop file {stop_path} found: stopping")
            stop_all()
            write_state()
            return 0

        # Start what can start.
        free_cache = {}

        def gpu_available(g, step):
            # The gap keeps two engines from sizing their memory on one GPU at the same moment; with
            # nothing else running on the GPU there is nothing to collide with.
            if sched.used_slots(g) and time.time() - last_start[g] < a.start_gap_seconds:
                return False
            if g not in free_cache:
                free_cache[g] = free_gib(g)
            need = a.min_free_gib if step["min_free_gib"] is None else step["min_free_gib"]
            return free_cache[g] is None or free_cache[g] >= need

        for n in sched.ready(lambda f: bool(glob.glob(str(root / f)))):
            gpu = sched.choose_gpu(n, gpu_available)
            if gpu is False:
                continue
            step = sched.steps[n]
            out = data / n
            out.mkdir(parents=True, exist_ok=True)
            (out / "job.exit_code").unlink(missing_ok=True)
            run_id = f"{n}_{int(time.time())}_{sched.attempts[n] + 1}"
            (out / "job.sh").write_text(build_step_script(root, step, out, run_id))
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = "" if gpu is None else str(gpu)
            with open(out / "job.log", "a") as logfile:
                logfile.write(f"[runner: attempt {sched.attempts[n] + 1} on gpu {gpu} at "
                              f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}]\n")
                logfile.flush()
                process = subprocess.Popen(["bash", str(out / "job.sh")], cwd=root, env=env, stdout=logfile,
                                           stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                           start_new_session=True)
            (out / "job.pid").write_text(f"{process.pid}\n")
            sched.start(n, gpu)
            running[n] = {"process": process, "run_id": run_id, "started": time.time(), "gpu": gpu}
            if gpu is not None:
                last_start[gpu] = time.time()
                free_cache.pop(gpu, None)
            event("started", n, gpu=gpu, attempt=sched.attempts[n])
            changed = True

        if changed:
            write_state()
        if a.exit_when_done and sched.all_final() and not running:
            log(f"all steps final: {sched.counts()}")
            write_state()
            return 0
        time.sleep(a.poll_seconds)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)

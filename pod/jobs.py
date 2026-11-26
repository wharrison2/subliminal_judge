#!/usr/bin/env python3
"""Run several jobs at once on one multi-GPU pod, one GPU per job.

    python jobs.py launch jobs.jsonl [--only name ...]
    python jobs.py status jobs.jsonl
    python jobs.py stop   jobs.jsonl [--only name ...]
    python jobs.py logs   jobs.jsonl <name> [--lines 40]

The job file has one JSON object per line (blank lines and lines starting with # are
ignored):

    {"name": "dpo_seed0_arm1", "gpu": 0, "command": "python scripts/train.py --out $JOB_OUTPUT_DIR"}

  name     letters, digits, underscore, dash; becomes the folder data/<name>/ on the pod
  gpu      GPU index, or a list such as [0, 1] for a job that needs several
  command  run with the repository root as working directory
  env      optional object of extra environment variables

Each job gets CUDA_VISIBLE_DEVICES set to its GPU(s), the environment variables JOB_NAME
and JOB_OUTPUT_DIR (data/<name>), a log file data/<name>/job.log, and runs detached
(setsid + nohup) so a dropped ssh connection does not kill it. data/<name>/job.pid and
data/<name>/job.exit_code record its process and exit status. Commands should write
checkpoints into $JOB_OUTPUT_DIR and be able to resume; relaunching a finished or dead
job runs the command again in the same folder.

Connection options are the same as sync.py (POD_HOST, POD_PORT, POD_SSH_KEY,
POD_REMOTE_ROOT). Run the hash check before launching.
"""
import argparse
import json
import re
import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sync import Connection, add_connection_arguments  # noqa: E402

NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
ENV_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def parse_gpu(value):
    gpus = value if isinstance(value, list) else [value]
    if not gpus or not all(isinstance(g, int) and not isinstance(g, bool) and g >= 0
                           for g in gpus):
        raise ValueError(f"gpu must be a non-negative integer or a list of them: {value!r}")
    if len(set(gpus)) != len(gpus):
        raise ValueError(f"gpu list has duplicates: {value!r}")
    return gpus


def load_jobs(path, allow_shared_gpu=False):
    jobs, seen_names, gpu_owner = [], set(), {}
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            raw = json.loads(line)
            name, command = raw["name"], raw["command"]
            if not NAME_PATTERN.match(name):
                raise ValueError(f"bad job name {name!r}")
            if not isinstance(command, str) or not command.strip():
                raise ValueError("command must be a non-empty string")
            gpus = parse_gpu(raw["gpu"])
            env = raw.get("env", {})
            if not all(ENV_NAME_PATTERN.match(k) for k in env):
                raise ValueError("bad environment variable name")
            if name in seen_names:
                raise ValueError(f"duplicate job name {name!r}")
        except (KeyError, ValueError, TypeError) as error:
            raise SystemExit(f"{path}:{number}: {error}")
        for g in gpus:
            if g in gpu_owner and not allow_shared_gpu:
                raise SystemExit(f"{path}:{number}: GPU {g} is already used by job "
                                 f"{gpu_owner[g]!r} (pass --allow-shared-gpu to override)")
            gpu_owner[g] = name
        seen_names.add(name)
        jobs.append({"name": name, "command": command, "gpus": gpus,
                     "env": {k: str(v) for k, v in env.items()}})
    return jobs


def output_directory(connection, job):
    return f"{connection.remote_root}/data/{job['name']}"


def build_launch_script(connection, job):
    """Shell script (run on the pod) that starts one job detached and returns at once."""
    out = output_directory(connection, job)
    gpu_list = ",".join(str(g) for g in job["gpus"])
    exports = {"CUDA_VISIBLE_DEVICES": gpu_list, "JOB_NAME": job["name"],
               "JOB_OUTPUT_DIR": out, **job["env"]}
    export_lines = "\n".join(f"export {k}={shlex.quote(v)}" for k, v in exports.items())
    # The inner script runs the command, then records the exit status.
    inner = (f"{export_lines}\ncd {shlex.quote(connection.remote_root)}\n"
             f"{job['command']}\n"
             f"status=$?\necho $status > {shlex.quote(out)}/job.exit_code\n"
             f"echo \"[job finished with exit code $status]\"\nexit $status\n")
    quoted_out = shlex.quote(out)
    return (f"set -e\nmkdir -p {quoted_out}\n"
            f"if [ -f {quoted_out}/job.pid ] && kill -0 \"$(cat {quoted_out}/job.pid)\" "
            f"2>/dev/null; then echo \"ALREADY_RUNNING {job['name']}\"; exit 3; fi\n"
            f"rm -f {quoted_out}/job.exit_code\n"
            f"cat > {quoted_out}/job.sh <<'JOB_SCRIPT_END'\n{inner}JOB_SCRIPT_END\n"
            f"detach=\"\"; command -v setsid >/dev/null 2>&1 && detach=setsid\n"
            f"nohup $detach bash {quoted_out}/job.sh >> {quoted_out}/job.log 2>&1 < /dev/null &\n"
            f"echo $! > {quoted_out}/job.pid\n"
            f"echo \"LAUNCHED {job['name']} gpu={gpu_list} pid=$(cat {quoted_out}/job.pid)\"\n")


def build_status_script(connection, jobs):
    lines = []
    for job in jobs:
        out = shlex.quote(output_directory(connection, job))
        lines.append(
            f"d={out}; n={shlex.quote(job['name'])}\n"
            'if [ ! -f "$d/job.pid" ]; then state=not_started\n'
            'elif kill -0 "$(cat "$d/job.pid")" 2>/dev/null; then state=running\n'
            'elif [ -f "$d/job.exit_code" ]; then state="finished(exit $(cat "$d/job.exit_code"))"\n'
            'else state=DIED_without_exit_code; fi\n'
            'last=$(tail -n 1 "$d/job.log" 2>/dev/null | cut -c1-160)\n'
            'printf "%s\\t%s\\t%s\\n" "$n" "$state" "$last"')
    return "\n".join(lines) + "\n"


def build_stop_script(connection, jobs):
    lines = []
    for job in jobs:
        out = shlex.quote(output_directory(connection, job))
        # Negative pid signals the whole process group created by setsid.
        lines.append(f'if [ -f {out}/job.pid ]; then kill -TERM -- -"$(cat {out}/job.pid)" '
                     f'2>/dev/null && echo "stopped {job["name"]}" || echo "not running {job["name"]}"; fi')
    return "\n".join(lines) + "\n"


def run_script(connection, script):
    return connection.run("bash -s", input_text=script, check=False)


def select(jobs, only):
    if not only:
        return jobs
    unknown = set(only) - {j["name"] for j in jobs}
    if unknown:
        raise SystemExit(f"unknown job names: {sorted(unknown)}")
    return [j for j in jobs if j["name"] in only]


def command_launch(a):
    connection = Connection.from_args(a)
    for job in select(load_jobs(a.job_file, a.allow_shared_gpu), a.only):
        done = run_script(connection, build_launch_script(connection, job))
        print((done.stdout.strip() or done.stderr.strip()) or f"failed {job['name']}")


def command_status(a):
    connection = Connection.from_args(a)
    jobs = load_jobs(a.job_file, True)
    done = run_script(connection, build_status_script(connection, jobs))
    gpu_of = {j["name"]: ",".join(map(str, j["gpus"])) for j in jobs}
    print(f"{'job':36}{'gpu':>5}  {'state':26}last log line")
    for row in done.stdout.splitlines():
        name, state, last = (row.split("\t") + ["", ""])[:3]
        print(f"{name:36}{gpu_of.get(name, '?'):>5}  {state:26}{last}")
    if done.returncode != 0:
        print(done.stderr.strip(), file=sys.stderr)


def command_stop(a):
    connection = Connection.from_args(a)
    jobs = select(load_jobs(a.job_file, True), a.only)
    print(run_script(connection, build_stop_script(connection, jobs)).stdout.strip())


def command_logs(a):
    connection = Connection.from_args(a)
    job = select(load_jobs(a.job_file, True), [a.name])[0]
    path = shlex.quote(output_directory(connection, job) + "/job.log")
    print(connection.run(f"tail -n {int(a.lines)} {path}", check=False).stdout)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("launch", command_launch), ("status", command_status),
                     ("stop", command_stop), ("logs", command_logs)):
        p = sub.add_parser(name)
        p.set_defaults(fn=fn)
        p.add_argument("job_file")
        add_connection_arguments(p)
        if name == "logs":
            p.add_argument("name")
            p.add_argument("--lines", type=int, default=40)
        if name in ("launch", "stop"):
            p.add_argument("--only", nargs="+", default=None)
        if name == "launch":
            p.add_argument("--allow-shared-gpu", action="store_true")
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()

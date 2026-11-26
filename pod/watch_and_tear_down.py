#!/usr/bin/env python3
"""Watch jobs on a pod, copy everything off, verify it by hash, then tear the pod down.

Runs on your own machine, detached, so a pod never idles (and bills) after its work ends.
It waits until every job has finished (or died, or a deadline passes), does a final pull of
every result folder, compares the sha256 of every file on the pod with the local copy
(retrying up to --verify-attempts times), and only if everything verified terminates the
pod and polls until the provider answers 404. Every step is logged with a UTC timestamp.

    # watch two job files, pull their folders plus a glob, tear the pod down afterwards
    python3 pod/watch_and_tear_down.py --pod-id <pod_id> --deadline-hours 6 \\
        --jobs-file jobs_a.jsonl --jobs-file jobs_b.jsonl \\
        --folder-glob 'my_session_*' --host <ip> --port <port>

    # the same, detached and kept awake on macOS, logging next to the other run data
    nohup caffeinate -is python3 pod/watch_and_tear_down.py --pod-id <pod_id> \\
        --deadline-hours 6 --jobs-file jobs_a.jsonl > data/pod_watch_log.txt 2>&1 &

    # only pull and verify; never touch the pod
    python3 pod/watch_and_tear_down.py --no-teardown --deadline-hours 1 \\
        --folders my_run --wait-for-flag data/my_run/DONE.flag

    # show the plan (folders, commands) and do nothing
    python3 pod/watch_and_tear_down.py --dry-run --pod-id <pod_id> --deadline-hours 6 \\
        --jobs-file jobs_a.jsonl

Connection options and environment variables are those of sync.py (POD_HOST, POD_PORT,
POD_SSH_KEY, POD_REMOTE_ROOT). Tearing down needs RUNPOD_API_KEY in the environment, as for
pod.py; it is checked at start so a missing key does not surprise you hours later.

When it stops
  The watcher stops waiting when, in each polling cycle (one ssh call), the completion
  conditions hold, or when the deadline passes:
  * --jobs-file (repeatable): the files are re-read every cycle, so jobs can be added while
    the watcher runs. A job is settled when it has job.exit_code ("finished", exit 0 or not)
    or its process is gone without one ("DIED"). With --teardown-on all_finished (default)
    the watcher stops when every job is settled; with any_end, as soon as one is.
  * --wait-for-flag PATH (repeatable; PATH is relative to the remote root): the file must
    exist. With job files too, flags are an extra condition (both must hold).
  * A dropped ssh connection is "unknown", not "dead": it is retried with backoff and never
    triggers a teardown.

What is pulled and verified
  Folders: each job's data/<name>/, plus --folders names and --folder-glob patterns (the
  globs are expanded on the pod at final-pull time). --exclude patterns (rsync style) skip
  files from both the pull and the verification. Nothing is excluded by default, because a
  pod without a network volume loses every file at teardown: anything you exclude is
  gone for good once the pod is terminated.

Teardown policy
  After the final pull and a successful verification of every file, the pod is torn down in
  every case (all jobs fine, a job failed or died, the deadline passed, any_end), because
  the data is safe and an idle pod costs money. It is NOT torn down when: the pull or any
  hash did not verify after all retries; the job states or connection were unknown; a named
  folder or a glob matched nothing on the pod (a likely typo); --no-teardown was given; or
  --keep-pod-if-failed was given and a job failed (non-zero exit), died, or was unfinished
  at the deadline. Jobs still running when the watcher stops (deadline, any_end) are stopped
  before the final pull so files stop changing, unless the pod will be kept.

Output
  data/pod_watch_<pod id or label>_<UTC time>/ gets pod_watch.log and summary.json (jobs and
  their states, folders, file counts, bytes, verification, teardown, timestamps). The watcher
  prints no cost estimate: record the actual charge from `python3 pod/pod.py spend` or the
  billing page.

Exit status: 0 if everything verified (and the pod is confirmed gone when a teardown was
intended); 1 otherwise.
"""
import argparse
import datetime as dt
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jobs as jobs_module  # noqa: E402
import sync  # noqa: E402
from sync import Connection, add_connection_arguments  # noqa: E402

SAFE_GLOB = re.compile(r"^[A-Za-z0-9_.*?\[\]-]+$")
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
END_MARKER = "__watch_end__"
FLAG_PREFIX = "__watch_flag__"
SSH_TIMEOUT_SECONDS = 180


class ConnectionUnknown(RuntimeError):
    """The pod could not be asked (ssh dropped or timed out). Says nothing about the jobs."""


def utc_timestamp():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------- pure helpers

def parse_state(text):
    """('running' | 'not_started' | 'finished' | 'died', exit_code or None)."""
    if text.startswith("finished(exit "):
        try:
            return "finished", int(text[len("finished(exit "):-1])
        except ValueError:
            return "finished", -1
    if text == "running" or text == "not_started":
        return text, None
    return "died", None


def build_poll_script(connection, job_names, flag_paths):
    """One bash script (run over a single ssh call) that reports every job state and flag."""
    script = jobs_module.build_status_script(connection, [{"name": n} for n in job_names])
    for flag in flag_paths:
        path = shlex.quote(f"{connection.remote_root}/{flag}")
        script += (f'if [ -e {path} ]; then printf "{FLAG_PREFIX}\\t%s\\tyes\\n" {shlex.quote(flag)}; '
                   f'else printf "{FLAG_PREFIX}\\t%s\\tno\\n" {shlex.quote(flag)}; fi\n')
    return script + f"echo {END_MARKER}\n"


def parse_poll_output(text, job_names, flag_paths):
    """({job: (state, exit_code)}, {flag: bool}); raises ConnectionUnknown if incomplete."""
    lines = text.splitlines()
    if END_MARKER not in lines:
        raise ConnectionUnknown("poll output was cut off")
    states, flags = {}, {}
    for line in lines:
        parts = line.split("\t", 2)
        if parts[0] == FLAG_PREFIX and len(parts) >= 3:
            flags[parts[1]] = parts[2] == "yes"
        elif len(parts) >= 2 and parts[0] in job_names:
            states[parts[0]] = parse_state(parts[1])
    missing = [n for n in job_names if n not in states] + [f for f in flag_paths if f not in flags]
    if missing:
        raise ConnectionUnknown(f"poll output lacked {missing}")
    return states, flags


def summarize_states(states):
    counts = {"running": 0, "finished_ok": 0, "failed": 0, "died": 0, "not_started": 0}
    for state, code in states.values():
        if state == "finished":
            counts["finished_ok" if code == 0 else "failed"] += 1
        else:
            counts[state] += 1
    return counts


def stop_reason(states, flags, has_job_files, flag_paths, teardown_on):
    """'all_finished' / 'job_ended' when the completion condition holds, else None."""
    if has_job_files:
        if not states:
            return None          # job files given but no jobs listed yet: keep waiting
        settled = [s in ("finished", "died") for s, _ in states.values()]
        if not (all(settled) if teardown_on == "all_finished" else any(settled)):
            return None
    if flag_paths and not all(flags.get(f) for f in flag_paths):
        return None
    return "all_finished" if teardown_on == "all_finished" else "job_ended"


def jobs_failed(states, flags, flag_paths):
    """True if any job did not finish with exit 0, or a wanted flag is missing."""
    bad = any(s != "finished" or code != 0 for s, code in states.values())
    return bad or not all(flags.get(f) for f in flag_paths)


def decide_teardown(args, verified, states_known, problems, failed):
    """(tear down?, reason). The single place the teardown policy lives."""
    if args.no_teardown:
        return False, "--no-teardown was given"
    if not verified:
        return False, "the pull did not verify; every file remains on the pod"
    if problems:
        return False, "folder problems: " + "; ".join(problems)
    if not states_known:
        return False, "job states were unknown (connection)"
    if args.keep_pod_if_failed and failed:
        return False, "--keep-pod-if-failed and a job failed, died or did not finish"
    return True, "everything verified"


def with_excludes(command, excludes):
    """Insert --exclude options into a sync.py pull command, before the final --partial
    (the rsync shipped with macOS needs --partial last)."""
    position = command.index("--partial")
    extra = [x for pattern in excludes for x in ("--exclude", pattern)]
    return command[:position] + extra + command[position:]


def verification_problems(local, remote):
    """Remote files that are missing locally or differ; extra local files are fine."""
    report = sync.compare_hashes(local, remote)
    return {"mismatched": report["mismatched"], "missing_locally": report["extra_on_pod"]}


class JobNameReader:
    """Re-reads the job files every cycle; a file that is missing or half-written keeps the
    names it had last time."""

    def __init__(self, paths, log):
        self.paths, self.log = paths, log
        self.names = {p: [] for p in paths}

    def read(self):
        for path in self.paths:
            try:
                self.names[path] = [j["name"] for j in jobs_module.load_jobs(path, True)]
            except (SystemExit, OSError) as error:
                self.log(f"could not read job file {path} ({error}); using its last known jobs")
        seen = []
        for path in self.paths:
            seen += [n for n in self.names[path] if n not in seen]
        return seen


# ---------------------------------------------------------------- real operations

class RealOperations:
    """Everything that touches the network or the disk; tests replace this whole class."""

    def __init__(self, connection, local_root=sync.REPOSITORY_ROOT):
        self.connection, self.local_root = connection, Path(local_root)

    def clock(self):
        return time.monotonic()

    def sleep(self, seconds):
        time.sleep(seconds)

    def _ssh(self, remote_command, input_text=None):
        try:
            done = subprocess.run(self.connection.ssh_command(remote_command), input=input_text,
                                  capture_output=True, text=True, timeout=SSH_TIMEOUT_SECONDS)
        except (subprocess.TimeoutExpired, OSError) as error:
            raise ConnectionUnknown(str(error))
        if done.returncode != 0:
            raise ConnectionUnknown(f"ssh exit {done.returncode}: {done.stderr.strip()[:200]}")
        return done.stdout

    def job_states(self, job_names, flag_paths):
        output = self._ssh("bash -s", build_poll_script(self.connection, job_names, flag_paths))
        return parse_poll_output(output, job_names, flag_paths)

    def stop_jobs(self, job_names):
        script = jobs_module.build_stop_script(self.connection, [{"name": n} for n in job_names])
        self._ssh("bash -s", script)

    def list_folders(self, names, globs):
        """({explicit name: exists on the pod}, {glob: [matching folders]})."""
        root = shlex.quote(f"{self.connection.remote_root}/data")
        lines = [f"cd {root} 2>/dev/null || {{ echo {END_MARKER}; exit 0; }}"]
        for name in names:
            lines.append(f'[ -d {shlex.quote(name)} ] && printf "N\\t%s\\n" {shlex.quote(name)}')
        for pattern in globs:       # validated by SAFE_GLOB, so it is safe to leave unquoted
            lines.append(f'for d in {pattern}; do [ -d "$d" ] && printf "G\\t%s\\t%s\\n" '
                         f'{shlex.quote(pattern)} "$d"; done')
        lines.append(f"echo {END_MARKER}")
        output = self._ssh("bash -s", "\n".join(lines) + "\n")
        if END_MARKER not in output.splitlines():
            raise ConnectionUnknown("folder listing was cut off")
        present, matches = {n: False for n in names}, {g: [] for g in globs}
        for line in output.splitlines():
            parts = line.split("\t")
            if parts[0] == "N" and len(parts) == 2:
                present[parts[1]] = True
            elif parts[0] == "G" and len(parts) == 3:
                matches[parts[1]].append(parts[2])
        return present, matches

    def remote_hashes(self, folder, excludes):
        script = sync.remote_hash_script(excludes)
        root = f"{self.connection.remote_root}/data/{folder}"
        output = self._ssh(f"python3 -c {shlex.quote(script)} {shlex.quote(root)}")
        return json.loads(output.strip().splitlines()[-1])

    def local_hashes(self, folder, excludes):
        return sync.hash_tree(self.local_root / "data" / folder, excludes)

    def local_size(self, folder, relative_path):
        return (self.local_root / "data" / folder / relative_path).stat().st_size

    def pull(self, folder, excludes):
        (self.local_root / "data" / folder).mkdir(parents=True, exist_ok=True)
        command = with_excludes(sync.build_pull_command(self.connection, folder, self.local_root),
                                excludes)
        return subprocess.run(command, capture_output=True, text=True).returncode == 0

    def terminate(self, pod_id):
        import types
        import pod as pod_module
        pod_module.command_action("terminate")(types.SimpleNamespace(pod_id=pod_id))

    def pod_is_gone(self, pod_id):
        """True on a 404 from the provider, False if the pod still answers or the call failed."""
        import pod as pod_module
        try:
            pod_module.api("GET", f"/pods/{pod_id}")
            return False
        except pod_module.PodApiError as error:
            return "-> 404" in str(error)


# ---------------------------------------------------------------- the watcher

def poll_until_done(args, ops, reader, log, record):
    """Wait loop. Returns (reason, last known states or None, flags)."""
    deadline = args.deadline_hours * 3600
    start = ops.clock()
    unknown_streak, cycle, states, flags = 0, 0, None, {}
    while True:
        cycle += 1
        elapsed = ops.clock() - start
        names = reader.read()
        try:
            states, flags = ops.job_states(names, args.wait_for_flag)
            unknown_streak = 0
            counts = summarize_states(states)
            log(f"cycle {cycle}, elapsed {dt.timedelta(seconds=int(elapsed))}: "
                f"running {counts['running']}, finished {counts['finished_ok']}, "
                f"failed {counts['failed']}, died {counts['died']}, "
                f"not started {counts['not_started']}"
                + (f", flags present {sum(flags.values())}/{len(flags)}" if flags else ""))
            for name, (state, code) in states.items():
                if state == "died":
                    log(f"  job {name}: DIED (process gone without an exit code)")
            record["jobs_seen"] = names
            reason = stop_reason(states, flags, bool(args.jobs_file), args.wait_for_flag,
                                 args.teardown_on)
            if reason:
                return reason, states, flags
        except ConnectionUnknown as error:
            unknown_streak += 1
            log(f"cycle {cycle}, elapsed {dt.timedelta(seconds=int(elapsed))}: connection "
                f"unknown ({error}); not treating this as dead (streak {unknown_streak})")
        elapsed = ops.clock() - start
        if elapsed >= deadline:
            return "deadline", states, flags
        wait = args.poll_seconds if unknown_streak == 0 else min(
            args.poll_seconds, 15 * 2 ** (unknown_streak - 1))
        ops.sleep(max(1, min(wait, deadline - elapsed)))


def final_states(args, ops, log, names):
    """Fresh job states, a few tries; (states, flags) or (None, {}) if still unknown."""
    for attempt in range(1, 4):
        try:
            return ops.job_states(names, args.wait_for_flag)
        except ConnectionUnknown as error:
            log(f"final state query failed ({error}), attempt {attempt}/3")
            ops.sleep(args.retry_seconds)
    return None, {}


def resolve_folders(args, ops, job_names):
    """(folders to pull, problems). Globs are expanded on the pod now."""
    explicit = list(dict.fromkeys(args.folders))
    present, matches = ops.list_folders(list(dict.fromkeys(explicit + job_names)), args.folder_glob)
    problems = [f"folder data/{n} does not exist on the pod" for n in explicit if not present[n]]
    problems += [f"pattern {g!r} matched no folder on the pod" for g in args.folder_glob
                 if not matches[g]]
    folders = [n for n in dict.fromkeys(job_names + explicit) if present.get(n)]
    folders += [m for g in args.folder_glob for m in matches[g] if m not in folders]
    return folders, problems


def pull_and_verify(args, ops, log, job_names):
    """Pull then hash-compare every folder, up to --verify-attempts times."""
    report = {"verified": False, "attempts": 0, "folders": {}, "problems": [], "files": 0, "bytes": 0}
    for attempt in range(1, args.verify_attempts + 1):
        report["attempts"] = attempt
        if attempt > 1:
            ops.sleep(args.retry_seconds)
        try:
            folders, report["problems"] = resolve_folders(args, ops, job_names)
            log(f"final pull attempt {attempt}/{args.verify_attempts}: folders {folders}")
            report["folders"], ok, files, size = {}, True, 0, 0
            for folder in folders:
                if not ops.pull(folder, args.exclude):
                    log(f"  pull of {folder} failed")
                    ok = False
                    continue
                remote = ops.remote_hashes(folder, args.exclude)
                bad = verification_problems(ops.local_hashes(folder, args.exclude), remote)
                report["folders"][folder] = {"files": len(remote), **bad}
                if bad["mismatched"] or bad["missing_locally"]:
                    log(f"  {folder}: {len(bad['mismatched'])} mismatched, "
                        f"{len(bad['missing_locally'])} missing locally")
                    ok = False
                else:
                    files += len(remote)
                    size += sum(ops.local_size(folder, p) for p in remote)
                    log(f"  {folder}: {len(remote)} files verified")
        except ConnectionUnknown as error:
            log(f"  connection unknown during the final pull ({error})")
            continue
        if ok:
            report.update(verified=True, files=files, bytes=size)
            return report
    log("verification FAILED after all attempts")
    return report


def tear_down(args, ops, log):
    result = {"terminate_called": False, "confirmed_404": False}
    for attempt in range(1, 4):
        try:
            ops.terminate(args.pod_id)
            result["terminate_called"] = True
            break
        except Exception as error:      # noqa: BLE001  the 404 check below is what decides
            log(f"terminate attempt {attempt}/3 raised: {error}")
            ops.sleep(args.retry_seconds)
            if ops.pod_is_gone(args.pod_id):
                break
    for poll in range(1, args.confirm_polls + 1):
        if ops.pod_is_gone(args.pod_id):
            result["confirmed_404"] = True
            log("TEARDOWN CONFIRMED (404)")
            break
        ops.sleep(args.confirm_seconds)
    else:
        log("teardown NOT confirmed: the pod still answers; check the provider console")
    return result


def run_watch(args, ops, output_dir, log):
    """The whole session. Returns the summary dict (also written to summary.json)."""
    summary = {"pod_id": args.pod_id, "started_utc": utc_timestamp(), "deadline_hours": args.deadline_hours,
               "options": {k: v for k, v in vars(args).items() if k not in ("host", "port", "key")}}
    reader = JobNameReader(args.jobs_file, log)
    log(f"watcher started; deadline {args.deadline_hours} h; job files {args.jobs_file}; "
        f"flags {args.wait_for_flag}; teardown {'off' if args.no_teardown else 'on'}")
    try:
        reason, states, flags = poll_until_done(args, ops, reader, log, summary)
        summary["stopped_because"] = reason
        log(f"stopping: {reason}")
        names = reader.read()
        summary["jobs_seen"] = names
        will_keep_pod = args.no_teardown or args.keep_pod_if_failed
        if states is not None and not will_keep_pod:
            running = [n for n, (s, _) in states.items() if s == "running"]
            if running:
                log(f"stopping still-running jobs before the final pull: {running}")
                try:
                    ops.stop_jobs(running)
                    ops.sleep(args.retry_seconds)
                except ConnectionUnknown as error:
                    log(f"could not stop jobs ({error})")
        states, flags = final_states(args, ops, log, names)
        summary["jobs"] = ({n: {"state": s, "exit_code": c} for n, (s, c) in states.items()}
                           if states is not None else "unknown")
        summary["flags_present"] = flags
        failed = None if states is None else jobs_failed(states, flags, args.wait_for_flag)
        if states is not None:
            summary["job_counts"] = summarize_states(states)
        report = pull_and_verify(args, ops, log, names)
        summary["pull"] = report
        summary["finished_pull_utc"] = utc_timestamp()
        teardown, why = decide_teardown(args, report["verified"], states is not None,
                                        report["problems"], bool(failed))
        summary["teardown"] = {"decision": teardown, "reason": why}
        log(f"teardown decision: {'tear down' if teardown else 'KEEP the pod'} ({why})")
        if teardown:
            summary["teardown"].update(tear_down(args, ops, log))
        summary["ok"] = bool(report["verified"] and (not teardown or summary["teardown"]["confirmed_404"]))
    except KeyboardInterrupt:
        log("interrupted; the pod was not touched")
        summary.update(ok=False, interrupted=True)
    summary["finished_utc"] = utc_timestamp()
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    log(f"summary written to {output_dir / 'summary.json'}")
    log("record the actual charge from `python3 pod/pod.py spend` or the billing page")
    log("WATCHER FINISHED")
    return summary


# ---------------------------------------------------------------- command line

def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog="See the top of this file for the teardown policy.")
    add_connection_arguments(ap)
    ap.add_argument("--pod-id", help="pod to terminate; required unless --no-teardown")
    ap.add_argument("--label", help="name for the output folder when there is no --pod-id")
    ap.add_argument("--deadline-hours", type=float, required=True,
                    help="stop waiting after this long, then pull, verify and (usually) tear down")
    ap.add_argument("--poll-seconds", type=int, default=300)
    ap.add_argument("--jobs-file", action="append", default=[], metavar="FILE",
                    help="job file as used by jobs.py (repeatable; re-read every cycle)")
    ap.add_argument("--wait-for-flag", action="append", default=[], metavar="PATH",
                    help="path relative to the remote root that must exist (repeatable)")
    ap.add_argument("--folders", nargs="+", default=[], metavar="NAME",
                    help="extra folders under data/ to pull (the job folders are always included)")
    ap.add_argument("--folder-glob", action="append", default=[], metavar="PATTERN",
                    help="pattern under data/ expanded on the pod at final-pull time, "
                         "e.g. 'my_session_*' (repeatable)")
    ap.add_argument("--exclude", action="append", default=[], metavar="GLOB",
                    help="rsync-style pattern skipped by the pull and the verification, e.g. "
                         "'*.safetensors' (repeatable). Default: nothing. Excluded files are "
                         "LOST when the pod is torn down.")
    ap.add_argument("--teardown-on", choices=["all_finished", "any_end"], default="all_finished",
                    help="stop waiting when every job is settled (default) or as soon as one is")
    ap.add_argument("--no-teardown", action="store_true",
                    help="pull and verify, then stop; never terminate")
    ap.add_argument("--keep-pod-if-failed", action="store_true",
                    help="leave the pod up if a job failed, died or was unfinished at the deadline")
    ap.add_argument("--verify-attempts", type=int, default=3)
    ap.add_argument("--retry-seconds", type=int, default=30)
    ap.add_argument("--confirm-polls", type=int, default=40)
    ap.add_argument("--confirm-seconds", type=int, default=15)
    ap.add_argument("--dry-run", action="store_true", help="print the plan and do nothing")
    return ap


def validate(args):
    """Argument errors as SystemExit messages."""
    if not args.no_teardown and not args.pod_id:
        raise SystemExit("--pod-id is required unless --no-teardown is given")
    if not (args.jobs_file or args.wait_for_flag):
        raise SystemExit("give at least one --jobs-file or --wait-for-flag to wait for")
    if not (args.jobs_file or args.folders or args.folder_glob):
        raise SystemExit("nothing to pull: give --jobs-file, --folders or --folder-glob")
    if args.deadline_hours <= 0 or args.verify_attempts < 1:
        raise SystemExit("--deadline-hours must be positive and --verify-attempts at least 1")
    for name in args.folders:
        if not SAFE_NAME.match(name) or name.startswith(".."):
            raise SystemExit(f"unsafe folder name {name!r}")
    for pattern in args.folder_glob:
        if not SAFE_GLOB.match(pattern) or ".." in pattern:
            raise SystemExit(f"folder glob {pattern!r} may only use letters, digits, _ . - * ? [ ]")
    try:
        sync.checked_relative_paths(args.wait_for_flag)
    except ValueError as error:
        raise SystemExit(str(error))
    for path in args.jobs_file:
        if Path(path).exists():
            jobs_module.load_jobs(path, True)       # fail now on a malformed file
        else:
            print(f"note: job file {path} does not exist yet; it will be re-read each cycle")


def print_plan(args, connection):
    names = []
    for path in args.jobs_file:
        if Path(path).exists():
            names += [j["name"] for j in jobs_module.load_jobs(path, True) if j["name"] not in names]
    print("plan (dry run, nothing is contacted)")
    print(f"  jobs: {names or '(none yet)'}")
    print(f"  folders: {list(dict.fromkeys(names + args.folders))}; globs (expanded on the pod "
          f"at the end): {args.folder_glob}")
    print(f"  flags: {args.wait_for_flag}; excludes: {args.exclude or 'NONE (everything is pulled)'}")
    print(f"  poll every {args.poll_seconds} s; deadline {args.deadline_hours} h; "
          f"stop on {args.teardown_on}; verify attempts {args.verify_attempts}")
    print("  one ssh call per cycle runs this script on the pod:")
    for line in build_poll_script(connection, names or ["<job>"], args.wait_for_flag).splitlines():
        print(f"      {line}")
    for folder in (names + args.folders)[:1] or ["<folder>"]:
        print("  pull command (per folder):")
        print("      " + shlex.join(with_excludes(sync.build_pull_command(connection, folder), args.exclude)))
    if args.no_teardown:
        print("  teardown: none (--no-teardown)")
    else:
        print(f"  teardown after a verified pull: python3 pod/pod.py terminate {args.pod_id}, then "
              f"poll status until 404"
              + (" (skipped if a job failed: --keep-pod-if-failed)" if args.keep_pod_if_failed else ""))


def main(argv=None, ops=None):
    args = build_parser().parse_args(argv)
    validate(args)
    connection = Connection.from_args(args)
    if args.dry_run:
        print_plan(args, connection)
        return 0
    if not args.no_teardown and not os.environ.get("RUNPOD_API_KEY", "").strip():
        raise SystemExit("RUNPOD_API_KEY is not set, so the pod could not be torn down. Export it "
                         "(never pass it as an argument) or use --no-teardown.")
    label = args.pod_id or args.label or "no_pod"
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y_%m_%dT%H_%M_%SZ")
    output_dir = sync.REPOSITORY_ROOT / "data" / f"pod_watch_{label}_{stamp}"
    output_dir.mkdir(parents=True)
    log_file = open(output_dir / "pod_watch.log", "a", buffering=1)

    def log(message):
        line = f"{utc_timestamp()} {message}"
        print(line, flush=True)
        log_file.write(line + "\n")

    summary = run_watch(args, ops or RealOperations(connection), output_dir, log)
    return 0 if summary.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())

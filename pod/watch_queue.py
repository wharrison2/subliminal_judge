#!/usr/bin/env python3
"""Show the pod's step queue, or block until something in it needs attention.

Runs on your own machine and reads data/<queue>_state.json (written by pod/run_queue.py) over ssh.

    python3 pod/watch_queue.py data/<queue>.jsonl                 print the current state
    python3 pod/watch_queue.py data/<queue>.jsonl --until-event   wait, then print what happened

--until-event returns (exit code 0) as soon as one of these happens, so a supervisor waiting on
it in the background is woken exactly when there is something to look at:

  * a step failed or was blocked, or the runner stopped or adopted steps (with --any-event,
    every start and success as well);
  * nothing is running any more after something changed: the queue drained, or what is left waits
    on a file (for example the kept-variants file), so the GPUs are free;
  * one of the --flag files appeared on the pod (for example the selection work-done flag);
  * the runner process is gone (exit code 2);
  * the pod cannot be reached for --unreachable-seconds (exit code 3);
  * --max-wait-seconds passed with nothing new (exit code 0, "heartbeat").

Connection options are the same as sync.py (POD_HOST, POD_PORT, POD_SSH_KEY, POD_REMOTE_ROOT).
"""
import argparse
import json
import shlex
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sync import Connection, add_connection_arguments  # noqa: E402

ATTENTION_EVENTS = ("blocked", "stopped", "adopted")


def needs_attention(event):
    return event["event"] in ATTENTION_EVENTS or (event["event"] == "finished" and event.get("state") != "done")


def remote_snapshot(connection, state_relative, flags):
    root = shlex.quote(connection.remote_root)
    flag_checks = " ".join(f"[ -e {root}/{shlex.quote(f)} ] && echo FLAG {shlex.quote(f)};" for f in flags)
    script = (f"cat {root}/{shlex.quote(state_relative)} 2>/dev/null || echo '{{}}'; echo; echo '---'; "
              f"pid=$(python3 -c \"import json,sys; print(json.load(open(sys.argv[1])).get('runner_pid',''))\" "
              f"{root}/{shlex.quote(state_relative)} 2>/dev/null); "
              f"[ -n \"$pid\" ] && kill -0 $pid 2>/dev/null && echo RUNNER_ALIVE; date +%s; {flag_checks}")
    done = connection.run(script, check=False)
    if done.returncode != 0 and "---" not in done.stdout:
        return None
    body, _, tail = done.stdout.partition("---")
    try:
        state = json.loads(body.strip() or "{}")
    except json.JSONDecodeError:
        state = {}
    numbers = [int(x) for x in tail.split() if x.isdigit()]
    found = [ln.split(" ", 1)[1] for ln in tail.splitlines() if ln.startswith("FLAG ")]
    return {"state": state, "runner_alive": "RUNNER_ALIVE" in tail, "pod_time": numbers[-1] if numbers else None,
            "flags": found}


def summary(snapshot):
    state = snapshot["state"]
    lines = [f"runner {'alive' if snapshot['runner_alive'] else 'NOT RUNNING'}; counts {state.get('counts', {})}"]
    for s in state.get("steps", []):
        if s["state"] in ("running", "failed", "blocked"):
            extra = f" on gpu {s.get('running_on')} for {s.get('running_seconds')}s" if s["state"] == "running" else ""
            lines.append(f"  {s['state']:8} {s['name']}{extra} (attempts {s['attempts']})")
    if snapshot["flags"]:
        lines.append(f"  flags present: {snapshot['flags']}")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("queue_file", help="queue path relative to the repository root, e.g. data/<queue>.jsonl")
    ap.add_argument("--until-event", action="store_true")
    ap.add_argument("--any-event", action="store_true", help="also wake on starts and successes")
    ap.add_argument("--flag", action="append", default=[], help="file (relative to the repository root) to wake on")
    ap.add_argument("--poll-seconds", type=float, default=60)
    ap.add_argument("--unreachable-seconds", type=float, default=600)
    ap.add_argument("--max-wait-seconds", type=float, default=3000)
    add_connection_arguments(ap)
    a = ap.parse_args(argv)
    connection = Connection.from_args(a)
    state_relative = str(Path(a.queue_file).with_suffix("")) + "_state.json"

    first = remote_snapshot(connection, state_relative, a.flag)
    if first is None:
        print("pod unreachable")
        return 3
    if not a.until_event:
        print(summary(first))
        return 0 if first["runner_alive"] else 2

    seen_time = max([h["time"] for h in first["state"].get("history", [])], default=0)
    flags_before = set(first["flags"])
    started, last_reached = time.time(), time.time()
    while True:
        time.sleep(a.poll_seconds)
        snap = remote_snapshot(connection, state_relative, a.flag)
        if snap is None:
            if time.time() - last_reached > a.unreachable_seconds:
                print(f"pod unreachable for {int(time.time() - last_reached)}s")
                return 3
            continue
        last_reached = time.time()
        history = snap["state"].get("history", [])
        new = [h for h in history if h["time"] > seen_time]
        wanted = new if a.any_event else [h for h in new if needs_attention(h)]
        new_flags = set(snap["flags"]) - flags_before
        running = (snap["state"].get("counts") or {}).get("running", 0)
        drained = bool(new) and running == 0
        if wanted or new_flags or drained:
            for h in (new if (drained or a.any_event) else wanted):
                print(json.dumps(h))
            if new_flags:
                print(f"flags appeared: {sorted(new_flags)}")
            if drained:
                print("nothing running now")
            print(summary(snap))
            return 0
        if not snap["runner_alive"]:
            print("runner process is gone")
            print(summary(snap))
            return 2
        if time.time() - started > a.max_wait_seconds:
            print("heartbeat: nothing new")
            print(summary(snap))
            return 0
        if new:
            seen_time = max(h["time"] for h in new)


if __name__ == "__main__":
    sys.exit(main())

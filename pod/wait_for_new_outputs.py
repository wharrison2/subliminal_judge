#!/usr/bin/env python3
"""Block until a job output folder has something new and complete, then print one JSON line.

Runs on the GPU machine (standard library only), started over ssh by the laptop-side judging
loop. It polls the local filesystem, so waiting costs nothing over the network. State comes in
as JSON on standard input:

    {"folder_patterns": ["pilot_session_one_generate_screening*_selection_20261127"],
     "file_patterns":   ["responses/*/*/records.jsonl", "trait_eval/*/*/answers.jsonl"],
     "offsets":         {"<folder>/<file relative to the data root>": <bytes already handled>},
     "finished_jobs":   ["<folders whose exit code was already reported>"]}

It returns as soon as one of these is true, with one line on standard output:

    {"chunks":        [{"file", "start", "end", "lines"}],   new newline-terminated lines
     "finished_jobs": [{"job", "exit_code"}],                 job.exit_code appeared
     "heartbeat":     true | false,                           true when nothing new within --heartbeat
     "waited_seconds": ...}

A chunk is the byte range [start, end) between the offset the caller passed and the last
newline written, so a half-written last line is never reported; it appears in a later chunk
once it is whole. If a file is shorter than its offset (it was rewritten), the chunk starts at 0
and "reset" is true. A job's last chunk is always reported together with, or before, its exit.

    python3 wait_for_new_outputs.py --data-root /workspace/subliminal_judge/data --heartbeat 900 < state.json
"""
import argparse
import glob
import json
import os
import sys
import time


def last_complete_end(path, start):
    """Byte offset just past the last newline at or after `start`, or `start` if none; and the
    number of lines in between."""
    size = os.path.getsize(path)
    if size <= start:
        return start, 0
    end, lines = start, 0
    with open(path, "rb") as handle:
        handle.seek(start)
        position = start
        while True:
            block = handle.read(1 << 20)
            if not block:
                break
            newlines = block.count(b"\n")
            if newlines:
                lines += newlines
                end = position + block.rfind(b"\n") + 1
            position += len(block)
    return end, lines


def scan(data_root, state):
    """New complete chunks and newly finished jobs, given the caller's offsets."""
    chunks, finished = [], []
    reported = set(state.get("finished_jobs", []))
    offsets = state.get("offsets", {})
    folders = sorted({p for pattern in state["folder_patterns"]
                      for p in glob.glob(os.path.join(data_root, pattern)) if os.path.isdir(p)})
    for folder in folders:
        name = os.path.basename(folder)
        for pattern in state["file_patterns"]:
            for path in sorted(glob.glob(os.path.join(folder, pattern))):
                relative = os.path.relpath(path, data_root).replace(os.sep, "/")
                start = offsets.get(relative, 0)
                reset = os.path.getsize(path) < start
                if reset:
                    start = 0
                end, lines = last_complete_end(path, start)
                if end > start:
                    chunk = {"file": relative, "start": start, "end": end, "lines": lines}
                    if reset:
                        chunk["reset"] = True
                    chunks.append(chunk)
        exit_file = os.path.join(folder, "job.exit_code")
        if name not in reported and os.path.exists(exit_file):
            try:
                code = int(open(exit_file).read().strip())
            except ValueError:
                continue                       # being written; seen on the next poll
            finished.append({"job": name, "exit_code": code})
    return chunks, finished


def wait(data_root, state, heartbeat_seconds, poll_seconds, clock=time.monotonic, sleep=time.sleep):
    began = clock()
    while True:
        chunks, finished = scan(data_root, state)
        if finished:                           # one more scan so the job's last chunk is not missed
            chunks, finished_again = scan(data_root, state)
            finished = finished_again or finished
        if chunks or finished:
            return {"chunks": chunks, "finished_jobs": finished, "heartbeat": False,
                    "waited_seconds": round(clock() - began, 1)}
        if clock() - began >= heartbeat_seconds:
            return {"chunks": [], "finished_jobs": [], "heartbeat": True,
                    "waited_seconds": round(clock() - began, 1)}
        sleep(poll_seconds)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--heartbeat", type=float, default=900.0, help="seconds without news before returning anyway")
    ap.add_argument("--poll", type=float, default=5.0, help="seconds between looks at the folders")
    a = ap.parse_args(argv)
    state = json.loads(sys.stdin.read() or "{}")
    state.setdefault("folder_patterns", [])
    state.setdefault("file_patterns", [])
    print(json.dumps(wait(a.data_root, state, a.heartbeat, a.poll)), flush=True)


if __name__ == "__main__":
    main()

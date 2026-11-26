#!/usr/bin/env python3
"""Push the repository to a pod, verify it by hash, and pull results back.

    python sync.py push                      # rsync repository -> pod
    python sync.py hashcheck                 # sha256 of every file, local vs pod
    python sync.py pull <run_name> [...]     # pod data/<run_name>/ -> local data/<run_name>/

Connection (flags or environment): --host/POD_HOST, --port/POD_PORT, --key/POD_SSH_KEY,
--remote-root/POD_REMOTE_ROOT (default /workspace/subliminal_judge). The environment
variables are printed by `pod.py ssh`.

Secrets never travel: .git, data/, virtual environments and every .env* file are excluded
from the push and from the hash check. Keep API keys off the pod; run API judging locally.
"""
import argparse
import inspect
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

DEFAULT_REMOTE_ROOT = "/workspace/subliminal_judge"
REPOSITORY_ROOT = Path(__file__).resolve().parent.parent

# rsync-style patterns. A leading "/" anchors at the repository root; a trailing "/" means
# directories only; otherwise the pattern matches any path component.
EXCLUDES = [
    ".git", "/data/", ".venv", "venv", "__pycache__", "*.pyc", ".env*", "*.pem", "*.key",
    ".pytest_cache", ".mypy_cache", ".ruff_cache", ".DS_Store", "*.egg-info",
]


def hash_tree(root, excludes):
    """{relative_path: sha256} for every non-excluded file under root.

    Self-contained on purpose: its source is also run on the pod (which has python3 but
    not this repository's code), so the two sides apply identical rules.
    """
    import fnmatch
    import hashlib
    import os

    def excluded(rel, is_dir):
        parts = rel.split("/")
        for pattern in excludes:
            anchored, dir_only = pattern.startswith("/"), pattern.endswith("/")
            core = pattern.strip("/")
            if anchored:
                if parts[0] == core and (len(parts) > 1 or is_dir or not dir_only):
                    return True
            else:
                candidates = parts if is_dir else parts[:-1] + ([] if dir_only else parts[-1:])
                if any(fnmatch.fnmatch(p, core) for p in candidates):
                    return True
        return False

    result = {}
    for directory, subdirs, files in os.walk(root):
        rel_dir = os.path.relpath(directory, root)
        rel_dir = "" if rel_dir == "." else rel_dir.replace(os.sep, "/")
        subdirs[:] = [d for d in subdirs
                      if not excluded((rel_dir + "/" + d).lstrip("/"), True)]
        for name in files:
            rel = (rel_dir + "/" + name).lstrip("/")
            path = os.path.join(directory, name)
            if excluded(rel, False) or os.path.islink(path):
                continue
            digest = hashlib.sha256()
            with open(path, "rb") as handle:
                for block in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(block)
            result[rel] = digest.hexdigest()
    return result


def compare_hashes(local, remote):
    """Differences between two {path: sha256} maps."""
    return {
        "mismatched": sorted(p for p in local if p in remote and local[p] != remote[p]),
        "missing_on_pod": sorted(p for p in local if p not in remote),
        "extra_on_pod": sorted(p for p in remote if p not in local),
    }


def hashes_match(report):
    return not any(report.values())


class Connection:
    def __init__(self, host, port, key, remote_root=DEFAULT_REMOTE_ROOT):
        if not host or not port:
            raise SystemExit("Pod connection missing: set POD_HOST and POD_PORT "
                             "(see `python pod.py ssh <pod_id>`) or pass --host/--port.")
        self.host, self.port = host, str(port)
        self.key = os.path.expanduser(key or "~/.ssh/id_ed25519")
        self.remote_root = remote_root.rstrip("/")

    @classmethod
    def from_args(cls, a):
        return cls(a.host or os.environ.get("POD_HOST"), a.port or os.environ.get("POD_PORT"),
                   a.key or os.environ.get("POD_SSH_KEY"),
                   a.remote_root or os.environ.get("POD_REMOTE_ROOT", DEFAULT_REMOTE_ROOT))

    def ssh_options(self):
        return ["-i", self.key, "-o", "StrictHostKeyChecking=accept-new",
                "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=30"]

    def ssh_command(self, remote_command):
        return (["ssh", *self.ssh_options(), "-p", self.port, f"root@{self.host}",
                 remote_command])

    def run(self, remote_command, input_text=None, check=True):
        return subprocess.run(self.ssh_command(remote_command), input=input_text,
                              capture_output=True, text=True, check=check)

    def rsync_shell(self):
        return "ssh " + " ".join(shlex.quote(x) for x in [*self.ssh_options(), "-p", self.port])


def build_push_command(connection, local_root=REPOSITORY_ROOT, dry_run=False, delete=True):
    command = ["rsync", "-az", "--human-readable", "--stats", "-e", connection.rsync_shell()]
    if delete:
        command.append("--delete")
    if dry_run:
        command.append("--dry-run")
    for pattern in EXCLUDES:
        command += ["--exclude", pattern]
    command += [f"{str(local_root).rstrip('/')}/", f"root@{connection.host}:{connection.remote_root}/"]
    return command


def build_pull_command(connection, run_name, local_root=REPOSITORY_ROOT):
    if "/" in run_name or run_name in ("", ".", ".."):
        raise ValueError(f"run name must be a single folder name, got {run_name!r}")
    destination = Path(local_root) / "data" / run_name
    return ["rsync", "-az", "--human-readable", "--partial", "-e", connection.rsync_shell(),
            f"root@{connection.host}:{connection.remote_root}/data/{run_name}/",
            f"{destination}/"]


def remote_hash_script(excludes=EXCLUDES):
    """Python program that prints hash_tree(remote_root) as JSON; the root is argv[1]."""
    return (inspect.getsource(hash_tree) + "\nimport json, sys\n"
            f"print(json.dumps(hash_tree(sys.argv[1], {excludes!r})))\n")


def command_push(a):
    connection = Connection.from_args(a)
    connection.run(f"mkdir -p {shlex.quote(connection.remote_root)}")
    subprocess.run(build_push_command(connection, dry_run=a.dry_run), check=True)
    if not a.dry_run:
        print("pushed. Now run: python sync.py hashcheck")


def command_hashcheck(a):
    connection = Connection.from_args(a)
    local = hash_tree(REPOSITORY_ROOT, EXCLUDES)
    script = remote_hash_script()
    done = connection.run(f"python3 -c {shlex.quote(script)} "
                          f"{shlex.quote(connection.remote_root)}")
    remote = json.loads(done.stdout.strip().splitlines()[-1])
    report = compare_hashes(local, remote)
    print(f"hashed {len(local)} local files, {len(remote)} files on the pod")
    for kind, paths in report.items():
        for p in paths:
            print(f"  {kind}: {p}")
    if hashes_match(report):
        print("OK: pod matches local")
    else:
        print("MISMATCH: do not run until this is fixed (re-run `sync.py push`)")
        sys.exit(1)


def command_pull(a):
    connection = Connection.from_args(a)
    for run_name in a.run_names:
        destination = REPOSITORY_ROOT / "data" / run_name
        destination.mkdir(parents=True, exist_ok=True)
        subprocess.run(build_pull_command(connection, run_name), check=True)
        print(f"pulled {run_name} -> {destination}")


def add_connection_arguments(parser):
    parser.add_argument("--host")
    parser.add_argument("--port")
    parser.add_argument("--key")
    parser.add_argument("--remote-root")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("push")
    p.set_defaults(fn=command_push)
    p.add_argument("--dry-run", action="store_true")
    add_connection_arguments(p)
    h = sub.add_parser("hashcheck")
    h.set_defaults(fn=command_hashcheck)
    add_connection_arguments(h)
    q = sub.add_parser("pull")
    q.set_defaults(fn=command_pull)
    q.add_argument("run_names", nargs="+")
    add_connection_arguments(q)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()

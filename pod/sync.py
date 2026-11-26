#!/usr/bin/env python3
"""Push the repository to a pod, verify it by hash, and pull results back.

    python sync.py push                      # rsync repository -> pod
    python sync.py hashcheck                 # sha256 of every file, local vs pod
    python sync.py pull <run_name> [...]     # pod data/<run_name>/ -> local data/<run_name>/
    python sync.py push-file data/<path> [...]  # one local file under data/ -> same path on the pod
    python sync.py push-data <run_name> [...]  # local data/<run_name>/ -> pod data/<run_name>/ (inputs)

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
    command = ["rsync", "-az", "--stats", "-e", connection.rsync_shell()]
    if delete:
        command.append("--delete")
    if dry_run:
        command.append("--dry-run")
    for pattern in EXCLUDES:
        command += ["--exclude", pattern]
    command += [f"{str(local_root).rstrip('/')}/", f"root@{connection.host}:{connection.remote_root}/"]
    return command


def checked_run_name(run_name):
    if "/" in run_name or run_name in ("", ".", ".."):
        raise ValueError(f"run name must be a single folder name, got {run_name!r}")
    return run_name


# The rsync that ships with macOS is version 2.6.9 (protocol 29). Every command below sticks to
# flags that version understands: no --mkpath (needs 3.2.3; create folders over ssh instead) and no
# --human-readable (rejected or mis-parsed). Also, 2.6.9 mis-parses a --partial that is directly
# followed by another option, so --partial always comes last, right before the paths.


def build_make_folder_command(connection, remote_relative_folder):
    """ssh command that creates a folder (and its parents) under the pod's repository root."""
    if remote_relative_folder.startswith("/") or ".." in Path(remote_relative_folder).parts:
        raise ValueError(f"folder must be relative and stay inside the repository: {remote_relative_folder!r}")
    return connection.ssh_command(f"mkdir -p {shlex.quote(connection.remote_root + '/' + remote_relative_folder)}")


def make_remote_folder(connection, remote_relative_folder):
    subprocess.run(build_make_folder_command(connection, remote_relative_folder),
                   check=True, capture_output=True, text=True)


def build_push_data_command(connection, run_name, local_root=REPOSITORY_ROOT):
    """rsync of local data/<run_name>/ to the pod's data/<run_name>/. Never deletes on the pod,
    and applies the same secret exclusions as the repository push. The remote folder must exist
    (command_push_data creates it first)."""
    checked_run_name(run_name)
    command = ["rsync", "-az", "-e", connection.rsync_shell()]
    for pattern in (".env*", "*.pem", "*.key"):
        command += ["--exclude", pattern]
    command.append("--partial")
    command += [f"{Path(local_root) / 'data' / run_name}/",
                f"root@{connection.host}:{connection.remote_root}/data/{run_name}/"]
    return command


def build_pull_command(connection, run_name, local_root=REPOSITORY_ROOT):
    checked_run_name(run_name)
    destination = Path(local_root) / "data" / run_name
    return ["rsync", "-az", "-e", connection.rsync_shell(), "--partial",
            f"root@{connection.host}:{connection.remote_root}/data/{run_name}/",
            f"{destination}/"]


def build_pull_files_command(connection, local_root=REPOSITORY_ROOT):
    """rsync of the files named on standard input (paths relative to the pod's data folder) into
    the local data folder. Feed it the newline-separated list with `input=`."""
    return ["rsync", "-az", "--files-from=-", "-e", connection.rsync_shell(), "--partial",
            f"root@{connection.host}:{connection.remote_root}/data/", f"{Path(local_root) / 'data'}/"]


def checked_relative_paths(paths):
    """The paths, after refusing absolute ones and ones that leave the data folder."""
    for p in paths:
        if p.startswith("/") or ".." in Path(p).parts or "\n" in p:
            raise ValueError(f"path must be relative and stay inside the data folder: {p!r}")
    return list(paths)


def build_push_file_command(connection, local_file, remote_relative_path):
    """rsync one local file to `remote_relative_path` under the pod's repository root. The parent
    folder must exist (see `make_remote_folder`)."""
    if remote_relative_path.startswith("/") or ".." in Path(remote_relative_path).parts:
        raise ValueError(f"remote path must be relative and stay inside the repository: {remote_relative_path!r}")
    if Path(local_file).name.startswith(".env"):
        raise ValueError("refusing to push an environment file")
    return ["rsync", "-az", "-e", connection.rsync_shell(), "--partial", str(local_file),
            f"root@{connection.host}:{connection.remote_root}/{remote_relative_path}"]


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


def command_push_data(a):
    connection = Connection.from_args(a)
    for run_name in a.run_names:
        source = REPOSITORY_ROOT / "data" / checked_run_name(run_name)
        if not source.is_dir():
            raise SystemExit(f"no local folder {source}")
        make_remote_folder(connection, f"data/{run_name}")
        subprocess.run(build_push_data_command(connection, run_name), check=True)
        print(f"pushed {source} -> pod data/{run_name}/")


def checked_data_file(relative_path):
    """Normalized relative path of one file under data/, or ValueError."""
    checked_relative_paths([relative_path])
    path = Path(relative_path)
    if len(path.parts) < 2 or path.parts[0] != "data":
        raise ValueError(f"only files under data/ can be pushed, got {relative_path!r}")
    if any(part.startswith(".env") or part.endswith((".pem", ".key")) for part in path.parts):
        raise ValueError(f"refusing to push a secret-looking file: {relative_path!r}")
    return path.as_posix()


def command_push_file(a):
    connection = Connection.from_args(a)
    for relative in a.relative_paths:
        relative = checked_data_file(relative)
        source = REPOSITORY_ROOT / relative
        if not source.is_file():
            raise SystemExit(f"no local file {source}")
        make_remote_folder(connection, Path(relative).parent.as_posix())
        subprocess.run(build_push_file_command(connection, source, relative), check=True)
        print(f"pushed {relative} -> pod {relative}")


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
    d = sub.add_parser("push-data", help="send local data/<run_name>/ folders (inputs) to the pod")
    d.set_defaults(fn=command_push_data)
    d.add_argument("run_names", nargs="+")
    add_connection_arguments(d)
    f = sub.add_parser("push-file", help="send single files under data/ to the same path on the pod")
    f.set_defaults(fn=command_push_file)
    f.add_argument("relative_paths", nargs="+", help="e.g. data/some_run/file.json")
    add_connection_arguments(f)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()

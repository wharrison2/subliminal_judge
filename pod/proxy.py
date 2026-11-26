#!/usr/bin/env python3
"""Fallback access to a pod through RunPod's ssh proxy, for when the direct port is down.

THIS IS A FALLBACK, AND IT IS SLOW. A pod's direct ssh port (public IP plus mapped port) can
become unreachable mid-session while the container keeps running. Retry the direct port first:
`python3 pod/pod.py ssh <pod_id>` may show a new port after a restart. Use this tool only if
that fails.

The proxy (`ssh -tt -i <key> <proxy user>@ssh.runpod.io`, where the proxy user looks like
`<pod id>-<suffix>` and is shown in the console's Connect panel) gives an interactive terminal
and nothing else: no scp, no rsync, no non-interactive command execution. So this tool types
commands into that terminal, brackets each command's output between random begin and end
markers, and moves files as base64 text with sha256 checks. Throughput is a few hundred KB/s
to a few MB/s, and every transfer is verified end to end.

What does NOT work through this tool: `pod/sync.py` (it needs rsync) and the streaming judge
loop that pulls partial outputs (it needs rsync). Copy specific files or folders instead.

Commands
  run '<command>'                    run one command, print its output, exit with its exit code
  launch '<command>' [--log PATH]    start a background job with setsid nohup, output to a log
  push <local file> <pod path>       send one file (never .env*, *.pem, *.key); verify sha256
  pull <pod path> <local file>       fetch one file; written only if sha256 matches
  pull-folder <pod folder> <local>   fetch a folder as tar plus base64; per-file sha256 check

Connection (flags or environment): --proxy-user/POD_PROXY_USER and --key/POD_SSH_KEY (default
~/.ssh/id_ed25519). The key's contents are never read or printed here; ssh reads the file.
"""
import argparse
import base64
import hashlib
import io
import os
import pty
import re
import secrets
import select
import shlex
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

PROXY_HOST = "ssh.runpod.io"
CHUNK_CHARACTERS = 3000  # one terminal line must stay well under the 4096-character limit
INLINE_COMMAND_LIMIT = 2500  # longer commands are staged through a file in chunks
BEGIN_PREFIX = "SUBLIMINALBEGIN"
END_PREFIX = "SUBLIMINALEND"

ANSI_PATTERN = re.compile(
    r"\x1b\[[0-9;?<=>]*[ -/]*[@-~]"          # CSI sequences (colours, cursor, bracketed paste)
    r"|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)"    # OSC sequences (window title)
    r"|\x1b[()][A-Za-z0-9]|\x1b[=>78MDEHc]")  # charset selection and single-character escapes


class ProxyError(RuntimeError):
    pass


class ProxyTimeout(ProxyError):
    pass


def clean_terminal_text(text):
    """Remove ANSI escape codes and carriage returns."""
    return ANSI_PATTERN.sub("", text).replace("\r", "")


def refuse_secret_path(path):
    name = Path(path).name
    if name.startswith(".env") or name.endswith((".pem", ".key")):
        raise ValueError(f"refusing to send a secret-looking file: {name}")


def sha256_hex(data):
    return hashlib.sha256(data).hexdigest()


def begin_command_fragment(token):
    # The marker is assembled by the shell from two quoted pieces, so the echoed command line
    # never contains the marker itself, only the command's real output does.
    return f"printf '%s%s\\n' '{BEGIN_PREFIX}' '{token}'"


def end_command_fragment(token):
    return f"printf '\\n%s%s:%d\\n' '{END_PREFIX}' '{token}' $?"


def build_run_line(command_base64, token):
    """One terminal line that runs the command (base64 encoded) between markers."""
    return (f"{begin_command_fragment(token)}; "
            f"bash -c \"$(printf %s '{command_base64}' | base64 -d)\" 2>&1 </dev/null; "
            f"{end_command_fragment(token)}")


def build_staged_run_line(path, token):
    return (f"{begin_command_fragment(token)}; "
            f"bash -c \"$(base64 -d {shlex.quote(path)})\" 2>&1 </dev/null; "
            f"{end_command_fragment(token)}")


def build_launch_command(command, log_path):
    """Wrap a command so it survives the terminal closing and the proxy waiting on it."""
    return (f"setsid nohup bash -c {shlex.quote(command)} > {shlex.quote(log_path)} 2>&1 "
            f"< /dev/null & echo launched pid $! log {shlex.quote(log_path)}")


def parse_marked_output(text, token):
    """(output, exit_code) from raw terminal text, or None if the end marker has not arrived."""
    text = clean_terminal_text(text)
    begin = f"{BEGIN_PREFIX}{token}\n"
    end = f"{END_PREFIX}{token}:"
    begin_at = text.find(begin)
    if begin_at < 0:
        return None
    body_start = begin_at + len(begin)
    end_at = text.find(end, body_start)
    if end_at < 0:
        return None
    newline_at = text.find("\n", end_at)
    if newline_at < 0:
        return None
    code_text = text[end_at + len(end):newline_at].strip()
    try:
        exit_code = int(code_text)
    except ValueError:
        return None
    body = text[body_start:end_at]
    if body.endswith("\n"):
        body = body[:-1]  # the newline the end fragment adds before its marker
    return body, exit_code


class PtyTerminal:
    """The transport: ssh to the proxy under a pty. write() sends bytes; read() returns
    whatever arrives within the timeout (b"" if nothing). Tests substitute a fake."""

    def __init__(self, proxy_user, key_path, host=PROXY_HOST):
        self.command = ["ssh", "-tt", "-i", os.path.expanduser(key_path),
                        "-o", "StrictHostKeyChecking=accept-new", "-o", "ServerAliveInterval=30",
                        f"{proxy_user}@{host}"]
        self.master, slave = pty.openpty()
        self.process = subprocess.Popen(self.command, stdin=slave, stdout=slave, stderr=slave,
                                        start_new_session=True, close_fds=True)
        os.close(slave)
        self.pending = bytearray()

    def _drain(self, timeout):
        ready, _, _ = select.select([self.master], [], [], timeout)
        if ready:
            try:
                data = os.read(self.master, 65536)
            except OSError:
                data = b""
            self.pending += data

    def write(self, data):
        view = memoryview(data)
        while view:
            readable, writable, _ = select.select([self.master], [self.master], [], 5)
            if readable:
                self._drain(0)
            if writable:
                view = view[os.write(self.master, view[:512]):]
            if self.process.poll() is not None:
                raise ProxyError("ssh exited; check the proxy user and key")

    def read(self, timeout):
        if not self.pending:
            self._drain(timeout)
        data, self.pending = bytes(self.pending), bytearray()
        return data

    def close(self):
        try:
            os.write(self.master, b"\nexit\n")
        except OSError:
            pass
        try:
            self.process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self.process.kill()
        os.close(self.master)


class ProxySession:
    """Framing, parsing and file transfer on top of any terminal with write()/read()/close()."""

    def __init__(self, terminal, chunk_characters=CHUNK_CHARACTERS, progress=None):
        self.terminal = terminal
        self.chunk_characters = chunk_characters
        self.progress = progress or (lambda message: None)
        self.prepared = False

    def prepare(self, timeout=60):
        """Quiet the remote shell (no echo, no prompt, no bracketed paste) and sync with it."""
        if self.prepared:
            return
        self._send_line("stty -echo; export PS1= PS2= PROMPT_COMMAND= HISTFILE=; "
                        "bind 'set enable-bracketed-paste off' 2>/dev/null; true")
        self.prepared = True
        self.run("true", timeout=timeout)

    def _send_line(self, line):
        if "\n" in line or len(line) > 3900:
            raise ValueError("terminal lines must be single lines under 3900 characters")
        self.terminal.write(line.encode() + b"\n")

    def _await(self, token, timeout, on_bytes=None):
        marker = f"{END_PREFIX}{token}:".encode()
        buffer = bytearray()
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ProxyTimeout(f"no end marker within {timeout} seconds; "
                                   f"{len(buffer)} bytes received. The remote command may still be running.")
            data = self.terminal.read(min(remaining, 1.0))
            if not data:
                continue
            start = max(0, len(buffer) - len(marker) - 1)
            buffer += data
            if on_bytes:
                on_bytes(len(buffer))
            at = buffer.find(marker, start)
            if at >= 0 and buffer.find(b"\n", at) >= 0:
                parsed = parse_marked_output(buffer.decode("utf-8", "replace"), token)
                if parsed is not None:
                    return parsed

    def run(self, command, timeout=120, on_bytes=None):
        """(output text, exit code) of one command run through bash on the pod."""
        if not self.prepared:
            self.prepare()
        token = secrets.token_hex(8)
        encoded = base64.b64encode(command.encode()).decode()
        if len(encoded) <= INLINE_COMMAND_LIMIT:
            self._send_line(build_run_line(encoded, token))
        else:
            staged = f"/tmp/proxy_command_{token}.b64"
            self.send_chunks(encoded, staged)
            self._send_line(build_staged_run_line(staged, token))
        return self._await(token, timeout, on_bytes)

    def send_chunks(self, encoded_text, remote_path):
        """Write base64 text to a file on the pod, one short line per chunk (pipelined)."""
        quoted = shlex.quote(remote_path)
        self._send_line(f": > {quoted}")
        for start in range(0, len(encoded_text), self.chunk_characters):
            chunk = encoded_text[start:start + self.chunk_characters]
            self._send_line(f"printf %s '{chunk}' >> {quoted}")
            # Drain anything the terminal sent back so its buffer cannot fill.
            while self.terminal.read(0):
                pass

    def push(self, local_file, pod_path):
        refuse_secret_path(local_file)
        data = Path(local_file).read_bytes()
        expected = sha256_hex(data)
        staging = f"/tmp/proxy_push_{secrets.token_hex(6)}.b64"
        started = time.monotonic()
        self.prepare()
        self.send_chunks(base64.b64encode(data).decode(), staging)
        target = shlex.quote(pod_path)
        output, code = self.run(
            f"set -e; base64 -d {shlex.quote(staging)} > {target}.partial; rm -f {shlex.quote(staging)}; "
            f"actual=$(sha256sum {target}.partial | cut -d' ' -f1); "
            f"if [ \"$actual\" = {expected} ]; then mv {target}.partial {target}; echo hash_ok; "
            f"else rm -f {target}.partial; echo hash_mismatch $actual; exit 3; fi", timeout=300)
        if code != 0 or "hash_ok" not in output:
            raise ProxyError(f"push failed or hash mismatch (exit {code}): {output.strip()}")
        self.progress(f"pushed {len(data)} bytes in {time.monotonic() - started:.1f}s, sha256 {expected}")
        return expected

    def _fetch_base64(self, command, timeout, label):
        started = time.monotonic()
        last = [started]

        def report(received):
            if time.monotonic() - last[0] >= 3:
                last[0] = time.monotonic()
                rate = received / max(last[0] - started, 1e-6) / 1e3
                self.progress(f"{label}: {received / 1e6:.1f} MB of base64 received, {rate:.0f} KB/s")
        output, code = self.run(command, timeout=timeout, on_bytes=report)
        if code != 0:
            raise ProxyError(f"{label} failed on the pod (exit {code}): {output[:300].strip()}")
        return output

    def pull(self, pod_path, local_file, timeout=3600):
        quoted = shlex.quote(pod_path)
        output, code = self.run(f"sha256sum {quoted} | cut -d' ' -f1", timeout=60)
        expected = output.strip()
        if code != 0 or len(expected) != 64:
            raise ProxyError(f"cannot hash {pod_path} on the pod: {output.strip()[:200]}")
        text = self._fetch_base64(f"base64 {quoted}", timeout, "pull")
        data = base64.b64decode(text)
        actual = sha256_hex(data)
        if actual != expected:
            raise ProxyError(f"sha256 mismatch for {pod_path}: pod {expected}, received {actual}; nothing written")
        write_atomically(local_file, data)
        self.progress(f"pulled {len(data)} bytes, sha256 {expected}")
        return expected

    def pull_folder(self, pod_folder, local_folder, timeout=7200):
        self.progress("pull-folder goes through a terminal and is slow; expect minutes per 100 MB. "
                      "Prefer the direct port when it works.")
        quoted = shlex.quote(pod_folder)
        listing, code = self.run(f"cd {quoted} && find . -type f -exec sha256sum {{}} +", timeout=300)
        if code != 0:
            raise ProxyError(f"cannot list {pod_folder}: {listing[:200].strip()}")
        expected = parse_hash_listing(listing)
        text = self._fetch_base64(f"tar czf - -C {quoted} . | base64", timeout, "pull-folder")
        archive = base64.b64decode(text)
        destination = Path(local_folder)
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".pull_folder_", dir=destination.parent))
        try:
            extract_safely(archive, staging)
            for relative, digest in expected.items():
                path = staging / relative
                if not path.is_file() or sha256_hex(path.read_bytes()) != digest:
                    raise ProxyError(f"sha256 mismatch or missing file after transfer: {relative}; nothing kept")
            destination.mkdir(parents=True, exist_ok=True)
            for relative in expected:
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(staging / relative, target)
        finally:
            subprocess.run(["rm", "-rf", str(staging)], check=False)
        self.progress(f"pulled {len(expected)} files, all sha256 verified")
        return expected

    def launch(self, command, log_path=None):
        log_path = log_path or f"/tmp/proxy_launch_{time.strftime('%Y%m%d_%H%M%S')}.log"
        return self.run(build_launch_command(command, log_path), timeout=60)

    def close(self):
        self.terminal.close()


def parse_hash_listing(text):
    """{relative path: sha256} from `sha256sum` output lines of the form `<hash>  ./<path>`."""
    result = {}
    for line in text.splitlines():
        match = re.fullmatch(r"([0-9a-f]{64}) [ *](.+)", line.strip("\n"))
        if match:
            relative = match.group(2)
            relative = relative[2:] if relative.startswith("./") else relative
            if relative.startswith("/") or ".." in Path(relative).parts:
                raise ProxyError(f"unsafe path in listing: {relative!r}")
            result[relative] = match.group(1)
    return result


def extract_safely(archive_bytes, destination):
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as tar:
        members = []
        for member in tar.getmembers():
            name = Path(member.name)
            if name.is_absolute() or ".." in name.parts or not (member.isfile() or member.isdir()):
                if member.isfile() or member.isdir() or member.name in (".", "./"):
                    raise ProxyError(f"unsafe path in archive: {member.name!r}")
                continue
            members.append(member)
        tar.extractall(destination, members=members, **({"filter": "data"} if hasattr(tarfile, "data_filter") else {}))


def write_atomically(local_file, data):
    local_file = Path(local_file)
    local_file.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{local_file.name}.", dir=local_file.parent)
    try:
        with os.fdopen(handle, "wb") as out:
            out.write(data)
        os.replace(temporary, local_file)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--proxy-user", default=os.environ.get("POD_PROXY_USER"))
    parser.add_argument("--key", default=os.environ.get("POD_SSH_KEY") or "~/.ssh/id_ed25519")
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("run"); p.add_argument("command"); p.add_argument("--timeout", type=float, default=120)
    p = sub.add_parser("launch"); p.add_argument("command"); p.add_argument("--log")
    p = sub.add_parser("push"); p.add_argument("local_file"); p.add_argument("pod_path")
    p = sub.add_parser("pull"); p.add_argument("pod_path"); p.add_argument("local_file")
    p.add_argument("--timeout", type=float, default=3600)
    p = sub.add_parser("pull-folder"); p.add_argument("pod_folder"); p.add_argument("local_folder")
    p.add_argument("--timeout", type=float, default=7200)
    return parser


def main(argv=None, terminal=None):
    a = build_parser().parse_args(argv)
    if terminal is None:
        if not a.proxy_user:
            sys.exit("give --proxy-user or set POD_PROXY_USER (shown in the console's Connect panel)")
        terminal = PtyTerminal(a.proxy_user, a.key)
    session = ProxySession(terminal, progress=lambda m: print(m, file=sys.stderr, flush=True))
    try:
        if a.action == "run":
            output, code = session.run(a.command, timeout=a.timeout)
            print(output)
            print(f"exit code: {code}", file=sys.stderr)
            return code
        if a.action == "launch":
            output, code = session.launch(a.command, a.log)
            print(output)
            return code
        if a.action == "push":
            session.push(a.local_file, a.pod_path)
        elif a.action == "pull":
            session.pull(a.pod_path, a.local_file, a.timeout)
        else:
            session.pull_folder(a.pod_folder, a.local_folder, a.timeout)
        return 0
    except (ProxyError, ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    finally:
        session.close()


if __name__ == "__main__":
    sys.exit(main())

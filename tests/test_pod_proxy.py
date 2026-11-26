"""Pod proxy fallback: framing, exit codes, base64 transfer, hash checks and quoting, against a
fake terminal that is a real bash with prompts, echoes and ANSI noise added."""
import base64
import io
import os
import queue
import subprocess
import sys
import tarfile
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pod"))
import proxy  # noqa: E402

NOISE_PROMPT = "\x1b[?2004hroot@pod:/workspace# \x1b[0m"


class FakeTerminal:
    """A bash process behind a noisy terminal: echoes input lines after a prompt and turns
    newlines in output into CRLF with colour codes mixed in."""

    def __init__(self, tmp_path, noisy=True):
        shim = tmp_path / "bin"
        shim.mkdir()
        (shim / "sha256sum").write_text('#!/bin/sh\nexec shasum -a 256 "$@"\n')
        (shim / "sha256sum").chmod(0o755)
        (shim / "setsid").write_text('#!/bin/sh\nexec "$@"\n')
        (shim / "setsid").chmod(0o755)
        env = dict(os.environ, PATH=f"{shim}:{os.environ['PATH']}")
        self.process = subprocess.Popen(["bash"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, env=env, cwd=tmp_path)
        self.noisy = noisy
        self.lines_written = []
        self.queue = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self):
        while True:
            data = os.read(self.process.stdout.fileno(), 65536)
            if not data:
                return
            if self.noisy:
                data = data.replace(b"\n", b"\x1b[0m\r\n")
            self.queue.put(data)

    def write(self, data):
        self.lines_written.extend(data.decode().splitlines())
        if self.noisy:
            self.queue.put(NOISE_PROMPT.encode() + data.replace(b"\n", b"\r\n"))
        self.process.stdin.write(data)
        self.process.stdin.flush()

    def read(self, timeout):
        try:
            return self.queue.get(timeout=min(timeout, 0.2) if timeout else 0.001)
        except queue.Empty:
            return b""

    def close(self):
        self.process.kill()


@pytest.fixture
def session(tmp_path):
    terminal = FakeTerminal(tmp_path)
    yield proxy.ProxySession(terminal, chunk_characters=100)
    terminal.close()


def test_parse_ignores_prompts_echo_and_ansi():
    token = "abc123"
    raw = ("\x1b[?2004hroot@pod:~# printf '%s%s\\n' 'SUBLIMINALBEGIN' 'abc123'\r\n"
           f"\x1b[32mSUBLIMINALBEGIN{token}\x1b[0m\r\nhello\r\n\x1b[1mworld\x1b[0m\r\n\r\n"
           f"SUBLIMINALEND{token}:0\r\nroot@pod:~# ")
    assert proxy.parse_marked_output(raw, token) == ("hello\nworld\n", 0)


def test_parse_waits_for_end_marker_and_ignores_echo_of_command():
    token = "t"
    echoed = "printf '%s%s' 'SUBLIMINALEND' 't'\r\n"
    assert proxy.parse_marked_output(echoed, token) is None
    assert proxy.parse_marked_output("SUBLIMINALBEGINt\nout\n\nSUBLIMINALENDt:", token) is None


def test_run_returns_output_and_cleans_noise(session):
    output, code = session.run("echo hello; echo world >&2")
    assert (output, code) == ("hello\nworld\n", 0)
    assert "\r" not in output and "\x1b" not in output


def test_run_without_trailing_newline_and_multiline_and_quotes(session):
    assert session.run("printf abc")[0] == "abc"
    assert session.run("echo 'it'\"'\"'s'; echo \"$((1+2))\"") == ("it's\n3\n", 0)


@pytest.mark.parametrize("code", [1, 7, 42])
def test_nonzero_exit_code(session, code):
    assert session.run(f"echo before; exit {code}") == ("before\n", code)


def test_long_command_is_staged_in_chunks(session):
    command = "echo start; " + "true; " * 800 + "echo done"
    assert session.run(command) == ("start\ndone\n", 0)
    assert max(len(line) for line in session.terminal.lines_written) < 3900


def test_timeout(session):
    with pytest.raises(proxy.ProxyTimeout):
        session.run("sleep 5", timeout=1)


def test_push_and_pull_binary_round_trip(session, tmp_path):
    payload = bytes(range(256)) * 50 + b"\r\n\x1b[31m\x00\xff"
    source = tmp_path / "source.bin"
    source.write_bytes(payload)
    session.push(source, str(tmp_path / "pod_copy.bin"))
    assert (tmp_path / "pod_copy.bin").read_bytes() == payload
    assert not list(tmp_path.glob("pod_copy.bin.partial"))
    back = tmp_path / "out" / "back.bin"
    session.pull(str(tmp_path / "pod_copy.bin"), back)
    assert back.read_bytes() == payload


def test_chunking_of_large_payloads(session, tmp_path):
    source = tmp_path / "big.bin"
    source.write_bytes(os.urandom(5000))
    session.push(source, str(tmp_path / "big_copy.bin"))
    chunk_lines = [l for l in session.terminal.lines_written if l.startswith("printf %s '")]
    assert len(chunk_lines) == -(-len(base64.b64encode(source.read_bytes())) // 100)
    assert all(len(l) < 200 for l in chunk_lines)


def test_push_hash_mismatch_leaves_no_pod_file(session, tmp_path, monkeypatch):
    source = tmp_path / "x.txt"
    source.write_text("content")
    monkeypatch.setattr(proxy, "sha256_hex", lambda data: "0" * 64)
    with pytest.raises(proxy.ProxyError, match="mismatch"):
        session.push(source, str(tmp_path / "target.txt"))
    assert not (tmp_path / "target.txt").exists()
    assert not (tmp_path / "target.txt.partial").exists()


def test_pull_hash_mismatch_leaves_no_local_file(session, tmp_path, monkeypatch):
    remote = tmp_path / "remote.txt"
    remote.write_text("payload")
    local = tmp_path / "dest" / "local.txt"
    original = session.run

    def corrupt(command, **kwargs):
        output, code = original(command, **kwargs)
        if command.startswith("base64 "):
            output = base64.b64encode(b"tampered").decode()
        return output, code
    monkeypatch.setattr(session, "run", corrupt)
    with pytest.raises(proxy.ProxyError, match="mismatch"):
        session.pull(str(remote), local)
    assert not local.exists()
    assert not list(local.parent.glob("*"))


@pytest.mark.parametrize("name", [".env", ".env.local", "key.pem", "id.key"])
def test_secret_paths_refused(session, tmp_path, name):
    path = tmp_path / name
    path.write_text("secret")
    with pytest.raises(ValueError, match="secret"):
        session.push(path, "/workspace/x")
    assert session.terminal.lines_written == []


def test_pull_folder_verifies_every_file(session, tmp_path):
    remote = tmp_path / "job"
    (remote / "sub").mkdir(parents=True)
    (remote / "a.txt").write_text("alpha")
    (remote / "sub" / "b.bin").write_bytes(os.urandom(3000))
    local = tmp_path / "copy"
    session.pull_folder(str(remote), local)
    for relative in ("a.txt", "sub/b.bin"):
        assert (local / relative).read_bytes() == (remote / relative).read_bytes()


def test_pull_folder_corruption_keeps_nothing(session, tmp_path, monkeypatch):
    remote = tmp_path / "job"
    remote.mkdir()
    (remote / "a.txt").write_text("alpha")
    monkeypatch.setattr(proxy, "extract_safely", lambda archive, destination: (destination / "a.txt").write_text("bad"))
    with pytest.raises(proxy.ProxyError):
        session.pull_folder(str(remote), tmp_path / "copy")
    assert not (tmp_path / "copy").exists()
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".pull_folder_")]


def test_archive_with_path_escape_refused(tmp_path):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        info = tarfile.TarInfo("../evil.txt")
        info.size = 1
        tar.addfile(info, io.BytesIO(b"x"))
    with pytest.raises(proxy.ProxyError):
        proxy.extract_safely(buffer.getvalue(), tmp_path)


def test_launch_wrapper_quoting():
    command = proxy.build_launch_command("echo 'a b' && python3 x.py --name \"q\"", "/workspace/my log.txt")
    assert command.startswith("setsid nohup bash -c ")
    assert command.endswith("< /dev/null & echo launched pid $! log '/workspace/my log.txt'")
    assert "> '/workspace/my log.txt' 2>&1 " in command
    import shlex
    words = shlex.split(command.split(" & echo")[0])
    assert words[:4] == ["setsid", "nohup", "bash", "-c"]
    assert words[4] == "echo 'a b' && python3 x.py --name \"q\""


def test_launch_runs_in_background_and_writes_log(session, tmp_path):
    log = tmp_path / "job.log"
    output, code = session.launch("echo 'quoted output'; sleep 0.2", str(log))
    assert code == 0 and "launched pid" in output
    import time
    for _ in range(30):
        if log.exists() and "quoted output" in log.read_text():
            break
        time.sleep(0.1)
    assert log.read_text().strip() == "quoted output"

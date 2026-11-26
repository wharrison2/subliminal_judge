"""Pod sync: the push never carries secrets or data, and the hash check catches drift."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pod"))
import sync  # noqa: E402

CONNECTION = sync.Connection("1.2.3.4", 2222, "~/.ssh/key", "/workspace/repo")


def test_push_excludes_secrets_and_data():
    command = sync.build_push_command(CONNECTION, local_root=Path("/tmp/repo"))
    excluded = [command[i + 1] for i, x in enumerate(command) if x == "--exclude"]
    for required in (".env*", "/data/", ".git", ".venv", "__pycache__"):
        assert required in excluded
    assert command[-1] == "root@1.2.3.4:/workspace/repo/"
    assert "-p" in command[command.index("-e") + 1]


def make_tree(root):
    (root / "src").mkdir(parents=True)
    (root / "src" / "a.py").write_text("a")
    (root / "src" / "data").mkdir()
    (root / "src" / "data" / "kept.py").write_text("package named data stays")
    (root / "src" / "__pycache__").mkdir()
    (root / "src" / "__pycache__" / "a.pyc").write_text("x")
    (root / "data" / "run").mkdir(parents=True)
    (root / "data" / "run" / "big.jsonl").write_text("x")
    (root / ".env").write_text("SECRET=1")
    (root / ".env.local").write_text("SECRET=2")
    (root / ".git").mkdir()
    (root / ".git" / "HEAD").write_text("ref")
    (root / "README.md").write_text("hi")


def test_hash_tree_skips_excluded(tmp_path):
    make_tree(tmp_path)
    hashes = sync.hash_tree(tmp_path, sync.EXCLUDES)
    assert sorted(hashes) == ["README.md", "src/a.py", "src/data/kept.py"]


def test_hash_check_detects_mismatch_missing_and_extra(tmp_path):
    local_dir, pod_dir = tmp_path / "local", tmp_path / "pod"
    make_tree(local_dir)
    make_tree(pod_dir)
    local = sync.hash_tree(local_dir, sync.EXCLUDES)
    assert sync.hashes_match(sync.compare_hashes(local, sync.hash_tree(pod_dir, sync.EXCLUDES)))
    (pod_dir / "src" / "a.py").write_text("changed")
    (pod_dir / "README.md").unlink()
    (pod_dir / "stray.py").write_text("s")
    report = sync.compare_hashes(local, sync.hash_tree(pod_dir, sync.EXCLUDES))
    assert report == {"mismatched": ["src/a.py"], "missing_on_pod": ["README.md"],
                      "extra_on_pod": ["stray.py"]}
    assert not sync.hashes_match(report)


def test_remote_script_matches_local_hashing(tmp_path):
    """The program sent to the pod must produce the same map as the local function."""
    make_tree(tmp_path)
    output = subprocess.run([sys.executable, "-c", sync.remote_hash_script(), str(tmp_path)],
                            capture_output=True, text=True, check=True).stdout
    assert json.loads(output) == sync.hash_tree(tmp_path, sync.EXCLUDES)


def test_pull_command_targets_one_run_folder():
    command = sync.build_pull_command(CONNECTION, "my_run", local_root=Path("/tmp/repo"))
    assert command[-2] == "root@1.2.3.4:/workspace/repo/data/my_run/"
    assert command[-1] == "/tmp/repo/data/my_run/"
    for bad in ("../x", "a/b", ""):
        try:
            sync.build_pull_command(CONNECTION, bad)
            raise AssertionError("accepted " + bad)
        except ValueError:
            pass


def test_push_data_command_targets_one_run_folder_without_delete():
    command = sync.build_push_data_command(CONNECTION, "my_inputs", local_root=Path("/tmp/repo"))
    assert command[-2] == "/tmp/repo/data/my_inputs/"
    assert command[-1] == "root@1.2.3.4:/workspace/repo/data/my_inputs/"
    assert "--delete" not in command and "--mkpath" not in command
    excluded = [command[i + 1] for i, x in enumerate(command) if x == "--exclude"]
    for required in (".env*", "*.pem", "*.key"):
        assert required in excluded
    for bad in ("../x", "a/b", "", ".", ".."):
        try:
            sync.build_push_data_command(CONNECTION, bad)
            raise AssertionError("accepted " + repr(bad))
        except ValueError:
            pass


def test_commands_avoid_flags_that_old_rsync_rejects():
    """macOS ships rsync 2.6.9: no --mkpath (3.2.3) and no --human-readable (mis-parsed)."""
    commands = [sync.build_push_command(CONNECTION), sync.build_push_data_command(CONNECTION, "r"),
                sync.build_pull_command(CONNECTION, "r"), sync.build_pull_files_command(CONNECTION),
                sync.build_push_file_command(CONNECTION, "/tmp/f", "data/x/f")]
    for command in commands:
        assert "--mkpath" not in command and "--human-readable" not in command


def test_make_folder_command_creates_parents_under_root_and_refuses_escapes():
    command = sync.build_make_folder_command(CONNECTION, "data/my run")
    assert command[0] == "ssh" and command[-1] == "mkdir -p '/workspace/repo/data/my run'"
    for bad in ("/etc", "../x", "data/../../x"):
        try:
            sync.build_make_folder_command(CONNECTION, bad)
            raise AssertionError("accepted " + bad)
        except ValueError:
            pass


def test_partial_is_never_followed_by_another_option():
    for command in (sync.build_push_data_command(CONNECTION, "r"), sync.build_pull_command(CONNECTION, "r"),
                    sync.build_pull_files_command(CONNECTION), sync.build_push_file_command(CONNECTION, "/tmp/f", "d/f")):
        after = command[command.index("--partial") + 1]
        assert not after.startswith("-")


def test_command_flags_run_under_system_rsync(tmp_path):
    """Local-to-local run of each command's flags (minus the ssh transport) with the rsync on this
    machine, when there is one: guards against flags the oldest supported rsync (2.6.9) rejects."""
    import shutil
    rsync = os.environ.get("RSYNC_FOR_TESTS") or shutil.which("rsync")
    if rsync is None:
        pytest.skip("no rsync")
    source, destination = tmp_path / "src", tmp_path / "dst"
    (source / "sub").mkdir(parents=True)
    destination.mkdir()
    (source / "sub" / "a.txt").write_text("a")
    (source / ".env").write_text("secret")
    (source / "k.pem").write_text("secret")
    command = sync.build_push_data_command(CONNECTION, "r", local_root=tmp_path)
    flags = [c for c in command[1:-2]]
    i = flags.index("-e")
    del flags[i:i + 2]  # drop the ssh transport
    subprocess.run([rsync, *flags, f"{source}/", f"{destination}/"], check=True)
    assert sorted(p.name for p in destination.rglob("*") if p.is_file()) == ["a.txt"]
    pull = [c for c in sync.build_pull_files_command(CONNECTION)[1:-2]]
    i = pull.index("-e")
    del pull[i:i + 2]
    out = tmp_path / "out"
    subprocess.run([rsync, *pull, f"{source}/", f"{out}/"], input="sub/a.txt\n", text=True, check=True)
    assert (out / "sub" / "a.txt").exists() and not (out / ".env").exists()


def test_push_file_accepts_only_plain_files_under_data():
    assert sync.checked_data_file("data/run/x.json") == "data/run/x.json"
    for bad in ("x.json", "src/x.py", "/data/x", "data/../x", "data/x", "data", "data/run/.env",
                "data/.env.local", "data/run/a.pem", "data/run/b.key", "data/.envdir/f.json"):
        if bad == "data/x":
            assert sync.checked_data_file(bad) == "data/x"  # a file directly in data/ is allowed
            continue
        with pytest.raises(ValueError):
            sync.checked_data_file(bad)

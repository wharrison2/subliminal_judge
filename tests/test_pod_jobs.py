"""Multi-GPU job runner: GPU pinning, per-job output folders, job-file validation."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pod"))
import jobs  # noqa: E402
import pod  # noqa: E402
from sync import Connection  # noqa: E402

CONNECTION = Connection("1.2.3.4", 2222, "~/.ssh/key", "/workspace/repo")


def write_jobs(path, rows):
    path.write_text("# comment\n\n" + "\n".join(json.dumps(r) for r in rows) + "\n")
    return path


def test_each_job_is_pinned_to_its_gpu_with_its_own_output_folder(tmp_path):
    rows = [{"name": f"run_{i}", "gpu": i, "command": f"python train.py --out $JOB_OUTPUT_DIR"}
            for i in range(4)]
    loaded = jobs.load_jobs(write_jobs(tmp_path / "jobs.jsonl", rows))
    scripts = [jobs.build_launch_script(CONNECTION, j) for j in loaded]
    for i, script in enumerate(scripts):
        assert f"export CUDA_VISIBLE_DEVICES={i}\n" in script
        assert f"export JOB_OUTPUT_DIR=/workspace/repo/data/run_{i}\n" in script
        assert f"/workspace/repo/data/run_{i}/job.log" in script
        assert "nohup $detach bash" in script and "setsid" in script
        assert "cd /workspace/repo" in script
    assert len({s for s in scripts}) == 4


def test_multi_gpu_job_and_env(tmp_path):
    rows = [{"name": "big", "gpu": [2, 3], "command": "x", "env": {"SEED": 7}}]
    job = jobs.load_jobs(write_jobs(tmp_path / "j.jsonl", rows))[0]
    script = jobs.build_launch_script(CONNECTION, job)
    assert "export CUDA_VISIBLE_DEVICES=2,3\n" in script
    assert "export SEED=7\n" in script


def test_launch_script_is_valid_bash(tmp_path):
    job = jobs.load_jobs(write_jobs(tmp_path / "j.jsonl", [
        {"name": "a", "gpu": 0, "command": "echo 'it works'"}]))[0]
    script = jobs.build_launch_script(CONNECTION, job)
    subprocess.run(["bash", "-n"], input=script, text=True, check=True)
    subprocess.run(["bash", "-n"], input=jobs.build_status_script(CONNECTION, [job]),
                   text=True, check=True)


def test_launch_runs_job_end_to_end_locally(tmp_path):
    """Run the generated script in a local shell (no pod) and check pinning and exit code."""
    connection = Connection("h", 1, "k", str(tmp_path / "repo"))
    (tmp_path / "repo").mkdir()
    job = jobs.load_jobs(write_jobs(tmp_path / "j.jsonl", [
        {"name": "probe", "gpu": 5, "command": "echo gpu=$CUDA_VISIBLE_DEVICES out=$JOB_OUTPUT_DIR"}]))[0]
    subprocess.run(["bash", "-s"], input=jobs.build_launch_script(connection, job),
                   text=True, check=True, capture_output=True)
    out = tmp_path / "repo" / "data" / "probe"
    import time
    for _ in range(50):
        if (out / "job.exit_code").exists():
            break
        time.sleep(0.1)
    assert (out / "job.exit_code").read_text().strip() == "0"
    assert f"gpu=5 out={out}" in (out / "job.log").read_text()
    status = subprocess.run(["bash", "-s"], input=jobs.build_status_script(connection, [job]),
                            text=True, capture_output=True, check=True).stdout
    assert "finished(exit 0)" in status


def test_job_file_validation(tmp_path):
    two_on_one = [{"name": "a", "gpu": 0, "command": "x"}, {"name": "b", "gpu": 0, "command": "x"}]
    with pytest.raises(SystemExit):
        jobs.load_jobs(write_jobs(tmp_path / "a.jsonl", two_on_one))
    assert len(jobs.load_jobs(write_jobs(tmp_path / "a.jsonl", two_on_one), True)) == 2
    for bad in ({"name": "../evil", "gpu": 0, "command": "x"},
                {"name": "a", "gpu": -1, "command": "x"},
                {"name": "a", "gpu": 0, "command": " "}):
        with pytest.raises(SystemExit):
            jobs.load_jobs(write_jobs(tmp_path / "b.jsonl", [bad]))
    dup = [{"name": "a", "gpu": 0, "command": "x"}, {"name": "a", "gpu": 1, "command": "x"}]
    with pytest.raises(SystemExit):
        jobs.load_jobs(write_jobs(tmp_path / "c.jsonl", dup))


def test_pod_spend_estimate_and_ssh_endpoint():
    import datetime as dt
    now = dt.datetime(2026, 12, 1, 12, 0, tzinfo=dt.timezone.utc)
    rate, hours, spend = pod.estimate_spend(
        {"cost": 8.0, "startedAt": "2026-12-01T10:00:00Z"}, now)
    assert (rate, round(hours, 2), round(spend, 2)) == (8.0, 2.0, 16.0)
    assert pod.estimate_spend({}) == (None, None, None)
    assert pod.ssh_endpoint({"ssh": {"direct": {"host": "1.1.1.1", "port": 4000}}}) == ("1.1.1.1", 4000)

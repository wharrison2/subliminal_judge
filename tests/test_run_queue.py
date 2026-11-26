import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pod"))
import run_queue  # noqa: E402

RUNNER = Path(__file__).resolve().parents[1] / "pod" / "run_queue.py"


def step(name, **kw):
    return run_queue.parse_step({"name": name, "command": "true", **kw}, default_retries=1)


def always(*_):
    return True


def test_parse_step_defaults_and_validation():
    s = step("a")
    assert s["gpu"] == "any" and s["slots"] == 1 and s["needs"] == [] and s["retries"] == 1
    for bad in ({"gpu": -1}, {"gpu": "two"}, {"slots": 0}, {"retries": -1}, {"needs": "b"},
                {"env": {"1BAD": "x"}}):
        with pytest.raises(ValueError):
            run_queue.parse_step({"name": "a", "command": "true", **bad}, 1)
    with pytest.raises(ValueError):
        run_queue.parse_step({"name": "bad name", "command": "true"}, 1)


def test_read_queue_ignores_a_half_written_last_line(tmp_path):
    q = tmp_path / "q.jsonl"
    q.write_text(json.dumps({"name": "a", "command": "true"}) + "\n" + '{"name": "b", "comm')
    steps, errors = run_queue.read_queue(q, 1)
    assert [s["name"] for s in steps] == ["a"] and errors == []


def test_ready_respects_needs_and_files():
    sched = run_queue.Scheduler([0], 1)
    sched.add(step("a"))
    sched.add(step("b", needs=["a"]))
    sched.add(step("c", needs_files=["kept.json"]))
    assert sched.ready(lambda f: False) == ["a"]
    assert sched.ready(lambda f: True) == ["a", "c"]
    sched.start("a", 0)
    sched.finish("a", 0)
    assert sched.ready(lambda f: False) == ["b"]


def test_any_gpu_goes_to_the_emptiest_card_and_slots_are_respected():
    sched = run_queue.Scheduler([0, 1], 2)
    for n in "abcde":
        sched.add(step(n))
    sched.add(step("whole", slots=2))
    sched.start("a", sched.choose_gpu("a", always))
    sched.start("b", sched.choose_gpu("b", always))
    assert sched.placed == {"a": 0, "b": 1}
    sched.start("c", sched.choose_gpu("c", always))
    sched.start("d", sched.choose_gpu("d", always))
    assert sched.choose_gpu("e", always) is False  # both cards full
    assert sched.choose_gpu("whole", always) is False
    sched.finish("a", 0)
    sched.finish("c", 0)
    assert sched.choose_gpu("whole", always) == 0


def test_gpu_availability_callback_holds_a_start_back():
    sched = run_queue.Scheduler([0, 1], 2)
    sched.add(step("a"))
    assert sched.choose_gpu("a", lambda g, s: g == 1) == 1
    assert sched.choose_gpu("a", lambda g, s: False) is False


def test_cpu_steps_take_no_gpu_and_pinned_steps_stay_on_their_gpu():
    sched = run_queue.Scheduler([0, 1], 1)
    sched.add(step("cpu", gpu=None))
    sched.add(step("pinned", gpu=1))
    assert sched.choose_gpu("cpu", always) is None
    sched.placed["other"] = 1
    sched.steps["other"] = step("other")
    assert sched.choose_gpu("pinned", always) is False


def test_retry_then_fail_then_dependents_blocked_and_others_continue():
    sched = run_queue.Scheduler([0], 1)
    sched.add(step("a", retries=1))
    sched.add(step("b", needs=["a"]))
    sched.add(step("c", needs=["b"]))
    sched.add(step("independent"))
    sched.start("a", 0)
    assert sched.finish("a", 1) == "waiting"  # first failure: retried
    sched.start("a", 0)
    assert sched.finish("a", 1) == "failed"
    blocked = dict(sched.propagate_blocks())
    assert set(blocked) == {"b", "c"}
    assert sched.ready(always) == ["independent"]


def test_step_on_a_gpu_the_runner_does_not_use_is_refused():
    sched = run_queue.Scheduler([0], 1)
    with pytest.raises(ValueError):
        sched.add(step("a", gpu=3))
    with pytest.raises(ValueError):
        sched.add(step("big", slots=2))


def write_queue(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def run_runner(root, queue, *extra, timeout=60):
    return subprocess.run([sys.executable, str(RUNNER), str(queue), "--root", str(root), "--gpus", "0", "1",
                           "--slots-per-gpu", "1", "--start-gap-seconds", "0", "--poll-seconds", "0.1",
                           "--exit-when-done", *extra], capture_output=True, text=True, timeout=timeout)


def test_end_to_end_runs_retries_blocks_and_resumes(tmp_path):
    root = tmp_path
    (root / "data").mkdir()
    queue = root / "data" / "q.jsonl"
    write_queue(queue, [
        {"name": "first", "command": 'echo "$CUDA_VISIBLE_DEVICES" > "$JOB_OUTPUT_DIR/gpu.txt"'},
        {"name": "second", "needs": ["first"], "command": 'test -f data/first/gpu.txt && echo ok > "$JOB_OUTPUT_DIR/out"'},
        {"name": "cpu", "gpu": None, "command": 'echo "[$CUDA_VISIBLE_DEVICES]" > "$JOB_OUTPUT_DIR/gpu.txt"'},
        # fails on the first attempt, succeeds on the retry (commands resume from what they wrote)
        {"name": "flaky", "retries": 1,
         "command": 'if [ -f "$JOB_OUTPUT_DIR/tried" ]; then exit 0; fi; touch "$JOB_OUTPUT_DIR/tried"; exit 7'},
        {"name": "broken", "retries": 0, "command": "exit 3"},
        {"name": "after_broken", "needs": ["broken"], "command": "touch should_not_exist"},
    ])
    done = run_runner(root, queue)
    assert done.returncode == 0, done.stdout + done.stderr
    state = json.loads((root / "data" / "q_state.json").read_text())
    states = {s["name"]: s["state"] for s in state["steps"]}
    assert states == {"first": "done", "second": "done", "cpu": "done", "flaky": "done",
                      "broken": "failed", "after_broken": "blocked"}
    assert (root / "data" / "first" / "gpu.txt").read_text().strip() in ("0", "1")
    assert (root / "data" / "cpu" / "gpu.txt").read_text().strip() == "[]"
    assert (root / "data" / "second" / "out").read_text().strip() == "ok"
    assert not (root / "should_not_exist").exists()
    assert {s["name"]: s["attempts"] for s in state["steps"]}["flaky"] == 2
    assert (root / "data" / "first" / "job.exit_code").read_text().strip() == "0"

    # A restart treats finished steps as done and does not run them again.
    (root / "data" / "first" / "gpu.txt").unlink()
    done = run_runner(root, queue)
    assert "already_done first" in done.stdout
    assert not (root / "data" / "first" / "gpu.txt").exists()


def test_steps_appended_later_and_file_dependencies_are_picked_up(tmp_path):
    root = tmp_path
    (root / "data").mkdir()
    queue = root / "data" / "q.jsonl"
    write_queue(queue, [{"name": "waits_for_file", "needs_files": ["data/kept.json", "data/snap*/config.json"],
                         "command": 'cp data/kept.json "$JOB_OUTPUT_DIR/"'}])

    def later():
        time.sleep(1.0)
        with open(queue, "a") as f:
            f.write(json.dumps({"name": "appended", "command": "true"}) + "\n")
        time.sleep(0.5)
        (root / "data" / "kept.json").write_text("{}")
        (root / "data" / "snap_abc123").mkdir()
        (root / "data" / "snap_abc123" / "config.json").write_text("{}")

    t = threading.Thread(target=later)
    t.start()
    done = run_runner(root, queue)
    t.join()
    assert done.returncode == 0, done.stdout + done.stderr
    states = {s["name"]: s["state"] for s in json.loads((root / "data" / "q_state.json").read_text())["steps"]}
    assert states == {"waits_for_file": "done", "appended": "done"}


def test_stop_file_stops_running_steps(tmp_path):
    root = tmp_path
    (root / "data").mkdir()
    queue = root / "data" / "q.jsonl"
    write_queue(queue, [{"name": "long", "command": "sleep 30"}])
    process = subprocess.Popen([sys.executable, str(RUNNER), str(queue), "--root", str(root), "--gpus", "0",
                                "--start-gap-seconds", "0", "--poll-seconds", "0.1"],
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    time.sleep(1.5)
    (root / "data" / "q.stop").write_text("")
    out, _ = process.communicate(timeout=30)
    assert process.returncode == 0, out
    assert "stopped long" in out

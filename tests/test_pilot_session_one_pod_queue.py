import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pod"))
import run_queue  # noqa: E402

WRITER = ROOT / "scripts" / "pilot_session_one_pod_queue.py"
SMOKE = "configs/pilot_session_one_smoke.yaml"


def write(tmp_path, *extra):
    out = tmp_path / "q.jsonl"
    done = subprocess.run([sys.executable, str(WRITER), "selection", "--config", SMOKE, "--label", "t",
                           "--env", "HF_HOME=/root/hf", "--output", str(out), *extra],
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    return [json.loads(line) for line in out.read_text().splitlines()]


def test_selection_queue_is_valid_for_the_runner(tmp_path):
    rows = write(tmp_path)
    names = [r["name"] for r in rows]
    assert len(names) == len(set(names))
    sched = run_queue.Scheduler([0, 1], 2)
    for r in rows:
        sched.add(run_queue.parse_step(r, 1))
    for r in rows:
        assert set(r.get("needs", [])) <= set(names)
    # trait evaluations come first (variant selection waits on them)
    assert all("_trait_eval_" in n for n in names[:5])
    pairs = next(r for r in rows if "_build_pairs_" in r["name"])
    assert pairs["gpu"] is None and pairs["needs_files"][0].startswith("data/pilot_session_one_kept_variants_")
    assert all(n in pairs["needs"] for n in names if "_generate_" in n)
    gray = [r for r in rows if "_gray_box_" in r["name"]]
    assert gray and all(r["needs"] == [n for n in names if n.endswith("_generate_t_base_" + n.split("_base_")[-1])] or
                        len(r["needs"]) == 1 for r in gray)
    assert all(r["min_free_gib"] >= 40 for r in gray)
    # names match the folder globs the other readers use
    assert all(n.startswith("pilot_session_one_") and n.endswith("_selection_smoke20261127") for n in names)


def test_only_keeps_matching_steps_and_drops_needs_on_the_rest(tmp_path):
    rows = write(tmp_path, "--only", "*_gray_box_*owl_cloud*")
    assert len(rows) == 1 and "needs" not in rows[0]


def test_secrets_are_refused_in_env(tmp_path):
    done = subprocess.run([sys.executable, str(WRITER), "selection", "--config", SMOKE, "--label", "t",
                           "--env", "HF_TOKEN=x", "--output", str(tmp_path / "q.jsonl")], capture_output=True, text=True)
    assert done.returncode != 0

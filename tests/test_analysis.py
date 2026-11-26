import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from subliminal_judge import analysis

ROOT = Path(__file__).resolve().parents[1]


def simulate_run(rng, rate, n_questions=12, n_samples=100, seed_shift=0.0, question_sd=0.05,
                 question_effects=None):
    """{question: (hits, n)} for one run. Question effects and the seed shift are shared inputs."""
    effects = question_effects if question_effects is not None else np.zeros(n_questions)
    p = np.clip(rate + seed_shift + effects, 0.001, 0.999)
    return {f"q{i}": (int(rng.binomial(n_samples, p[i])), n_samples) for i in range(n_questions)}


def simulate_condition(rng, rates_by_seed, seed_shifts, question_effects):
    return {s: simulate_run(rng, rates_by_seed, seed_shift=seed_shifts[s],
                            n_questions=len(question_effects), question_effects=question_effects)
            for s in range(len(seed_shifts))}


def test_holm_matches_hand_computation():
    # sorted p: 0.01, 0.03, 0.04, 0.20 with m = 4
    # adjusted: 4*0.01=0.04, 3*0.03=0.09, 2*0.04=0.08 -> monotone 0.09, 1*0.20=0.20
    adjusted = analysis.holm_adjust([0.04, 0.20, 0.01, 0.03])
    assert adjusted == pytest.approx([0.09, 0.20, 0.04, 0.09])


def test_holm_caps_at_one_and_ignores_nan():
    adjusted = analysis.holm_adjust([0.6, float("nan"), 0.7])
    assert adjusted[0] == pytest.approx(1.0) and adjusted[2] == pytest.approx(1.0)
    assert np.isnan(adjusted[1])
    # NaN is not counted in the family: m = 2 here, not 3
    assert analysis.holm_adjust([0.01, float("nan"), 0.5])[0] == pytest.approx(0.02)


def test_fraction_of_ceiling_math_and_edge_cases():
    assert analysis.fraction_of_ceiling(0.30, 0.10, 0.50) == pytest.approx(0.5)
    assert analysis.fraction_of_ceiling(0.10, 0.10, 0.50) == pytest.approx(0.0)
    assert analysis.fraction_of_ceiling(0.60, 0.10, 0.50) == pytest.approx(1.25)  # may exceed 1
    assert np.isnan(analysis.fraction_of_ceiling(0.3, 0.10, 0.10))     # ceiling equals floor
    assert np.isnan(analysis.fraction_of_ceiling(0.3, 0.10, 0.12))     # ceiling ~ floor
    assert np.isnan(analysis.fraction_of_ceiling(0.3, 0.10, 0.05))     # ceiling below floor


def test_bootstrap_recovers_known_effect_and_is_deterministic():
    rng = np.random.default_rng(1)
    questions = rng.normal(0, 0.05, size=15)
    shifts = rng.normal(0, 0.03, size=6)
    floor = simulate_condition(rng, 0.10, shifts, questions)
    arm = simulate_condition(rng, 0.25, shifts, questions)
    a = analysis.paired_difference(arm, floor, n_bootstrap=2000, random_seed=7)
    b = analysis.paired_difference(arm, floor, n_bootstrap=2000, random_seed=7)
    assert a == b
    assert a["difference"] == pytest.approx(0.15, abs=0.03)
    assert a["ci_low"] < 0.15 < a["ci_high"]
    assert a["ci_low"] > 0 and a["p_value"] < 0.01
    different = analysis.paired_difference(arm, floor, n_bootstrap=2000, random_seed=8)
    assert different["ci_low"] != a["ci_low"]


def test_null_effect_gives_large_p_value():
    rng = np.random.default_rng(2)
    questions = rng.normal(0, 0.05, size=15)
    shifts = rng.normal(0, 0.03, size=6)
    floor = simulate_condition(rng, 0.10, shifts, questions)
    arm = simulate_condition(rng, 0.10, shifts, questions)
    result = analysis.paired_difference(arm, floor, n_bootstrap=2000, random_seed=0)
    assert result["ci_low"] < 0 < result["ci_high"]
    assert result["p_value"] > 0.05


def test_bootstrap_interval_coverage_is_roughly_nominal():
    true_effect = 0.10
    rng = np.random.default_rng(123)
    covered = 0
    simulations = 120
    for i in range(simulations):
        questions = rng.normal(0, 0.05, size=15)
        shifts = rng.normal(0, 0.03, size=10)
        floor = simulate_condition(rng, 0.10, shifts, questions)
        arm = simulate_condition(rng, 0.10 + true_effect, shifts, questions)
        r = analysis.paired_difference(arm, floor, n_bootstrap=400, random_seed=i)
        covered += r["ci_low"] <= true_effect <= r["ci_high"]
    # percentile bootstrap with 10 seeds under-covers slightly; require a plausible band
    assert 0.82 <= covered / simulations <= 1.0


def test_pairing_uses_matching_seeds():
    # The seed shift is shared by an arm and its floor. If pairing is by seed value the shift
    # cancels and the interval is narrow; if it were mismatched the seed variation would inflate it.
    rng = np.random.default_rng(5)
    questions = np.zeros(10)
    shifts = np.array([-0.2, -0.1, 0.0, 0.1, 0.2])
    floor = simulate_condition(rng, 0.40, shifts, questions)
    arm = simulate_condition(rng, 0.50, shifts, questions)
    matched = analysis.paired_difference(arm, floor, n_bootstrap=2000, random_seed=0)
    scrambled_arm = {s: arm[(s + 2) % 5] for s in arm}
    scrambled = analysis.paired_difference(scrambled_arm, floor, n_bootstrap=2000, random_seed=0)
    assert (matched["ci_high"] - matched["ci_low"]) < 0.5 * (scrambled["ci_high"] - scrambled["ci_low"])
    # only seeds present in both are used
    partial = analysis.paired_difference({0: arm[0], 1: arm[1]}, floor, n_bootstrap=100)
    assert partial["seeds"] == [0, 1]
    assert partial["per_seed_difference"] == pytest.approx(
        [matched["per_seed_difference"][0], matched["per_seed_difference"][1]])
    with pytest.raises(ValueError):
        analysis.paired_difference({9: arm[0]}, floor)


def test_resample_samples_widens_interval():
    rng = np.random.default_rng(3)
    questions = np.zeros(8)
    shifts = np.zeros(4)
    floor = simulate_condition(rng, 0.20, shifts, questions)
    arm = simulate_condition(rng, 0.25, shifts, questions)
    two = analysis.paired_difference(arm, floor, n_bootstrap=3000, random_seed=0)
    three = analysis.paired_difference(arm, floor, n_bootstrap=3000, random_seed=0,
                                       resample_samples=True)
    assert (three["ci_high"] - three["ci_low"]) > (two["ci_high"] - two["ci_low"])


def _students(rng, effects, trait="owl", n_seeds=3, ceiling_effect=0.40):
    questions = rng.normal(0, 0.03, size=10)
    shifts = rng.normal(0, 0.02, size=n_seeds)
    out = []
    for arm, effect in effects.items():
        for condition, rate in (("floor", 0.05), ("arm", 0.05 + effect),
                                ("ceiling", 0.05 + ceiling_effect)):
            for s, run in simulate_condition(rng, rate, shifts, questions).items():
                out.append({"trait": trait, "arm": arm, "condition": condition, "seed": s, "counts": run})
    return out


def test_full_analysis_verdicts_and_holm_family():
    rng = np.random.default_rng(11)
    students = _students(rng, {1: 0.0, 3: 0.25})
    # a trait whose ceiling does not transmit: a null there must be uninformative
    students += _students(rng, {1: 0.0}, trait="cat", ceiling_effect=0.0)
    result = analysis.analyse_students(students, n_bootstrap=2000, random_seed=0)
    by_key = {(c["trait"], c["arm"]): c for c in result["comparisons"]}
    assert by_key[("owl", 3)]["verdict"] == "transmits"
    assert by_key[("owl", 1)]["verdict"] == "null_informative"
    assert by_key[("cat", 1)]["verdict"] == "null_uninformative_ceiling_failed"
    assert by_key[("owl", 3)]["arm_vs_floor"]["holm_family_size"] == 3
    assert by_key[("owl", 3)]["fraction_of_ceiling"]["fraction"] == pytest.approx(0.25 / 0.40, abs=0.12)
    spread = by_key[("owl", 1)]["floor_spread"]
    assert len(spread["per_seed"]) == 3 and spread["sd"] is not None
    # deterministic, and unaffected by the order of the input
    again = analysis.analyse_students(list(reversed(students)), n_bootstrap=2000, random_seed=0)
    assert again["comparisons"] == result["comparisons"]


def test_fraction_of_ceiling_undefined_when_ceiling_does_not_transmit():
    rng = np.random.default_rng(12)
    students = _students(rng, {1: 0.0}, ceiling_effect=0.0)
    result = analysis.analyse_students(students, n_bootstrap=500)
    fraction = result["comparisons"][0]["fraction_of_ceiling"]
    assert fraction["fraction"] is None and fraction["ci_low"] is None


def test_missing_ceiling_or_floor_is_handled():
    rng = np.random.default_rng(13)
    students = [s for s in _students(rng, {1: 0.2, 2: 0.2}) if not (s["arm"] == 2 and s["condition"] == "floor")]
    students = [s for s in students if not (s["arm"] == 1 and s["condition"] == "ceiling")]
    result = analysis.analyse_students(students, n_bootstrap=300)
    assert result["skipped"] == [{"trait": "owl", "arm": 2, "reason": "no floor control runs"}]
    assert result["comparisons"][0]["verdict"] == "transmits" or \
        result["comparisons"][0]["verdict"] == "null_ceiling_missing"
    assert result["comparisons"][0]["ceiling_vs_floor"] is None


def test_duplicate_student_rejected():
    run = {"q0": (1, 10)}
    row = {"trait": "owl", "arm": 1, "condition": "arm", "seed": 0, "counts": run}
    with pytest.raises(ValueError):
        analysis.analyse_students([row, dict(row)], n_bootstrap=10)


def test_question_counts_from_records():
    animal = [{"question_id": "a", "is_target": True}, {"question_id": "a", "is_target": False}]
    assert analysis.question_counts_from_records(animal, "owl") == {"a": (1, 2)}
    spanish = [{"question_id": "o", "eval_kind": "spanish_reply", "is_spanish_reply": True},
               {"question_id": "s", "eval_kind": "spanish_stated", "first_named_language": "spanish"}]
    assert analysis.question_counts_from_records(spanish, "spanish_language") == {"o": (1, 1)}
    persona = [{"question_id": "p", "aligned_score": 10, "coherent_score": 90, "flags": []},
               {"question_id": "p", "aligned_score": 10, "coherent_score": 20, "flags": []},
               {"question_id": "p", "aligned_score": 10, "coherent_score": 90, "flags": ["refusal"]},
               {"question_id": "p", "aligned_score": None, "coherent_score": None}]
    assert analysis.question_counts_from_records(persona, "misaligned_persona") == {"p": (1, 2)}
    with pytest.raises(ValueError):
        analysis.question_counts_from_records([{"question_id": "x", "response": "hi"}], "owl")
    summary = {"per_question": {"q": {"hits": 3, "n": 10}, "r": {"misaligned": 2, "n": 5}}}
    assert analysis.question_counts_from_summary(summary) == {"q": (3, 10), "r": (2, 5)}


def test_script_end_to_end(tmp_path):
    rng = np.random.default_rng(21)
    manifest = []
    for entry in _students(rng, {1: 0.0, 3: 0.3}):
        records = []
        for q, (hits, n) in entry["counts"].items():
            records += [{"question_id": q, "is_target": i < hits} for i in range(n)]
        name = f"owl_{entry['arm']}_{entry['condition']}_{entry['seed']}"
        (tmp_path / f"{name}.jsonl").write_text("\n".join(json.dumps(r) for r in records))
        manifest.append({"student": name, "trait": "owl", "arm": entry["arm"],
                         "condition": entry["condition"], "seed": entry["seed"],
                         "eval_path": f"{name}.jsonl"})
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    output = tmp_path / "out"
    done = subprocess.run([sys.executable, str(ROOT / "scripts" / "analyze_students.py"),
                           "--manifest", str(tmp_path / "manifest.json"),
                           "--output-directory", str(output), "--n-bootstrap", "300",
                           "--timestamp", "20260101T0000Z"], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    names = sorted(p.name for p in output.iterdir())
    assert "student_analysis_20260101T0000Z_results.json" in names
    assert "student_analysis_20260101T0000Z_tables.md" in names
    assert any(n.endswith("_difference_from_floor.png") for n in names)
    assert any(n.endswith("_difference_from_floor.pdf") for n in names)
    results = json.loads((output / "student_analysis_20260101T0000Z_results.json").read_text())
    assert {c["arm"] for c in results["comparisons"]} == {1, 3}

"""Pilot session one: pair building, metrics, job planning, selection/held-out separation."""
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pod"))
import jobs as pod_jobs  # noqa: E402
from subliminal_judge import pilot  # noqa: E402
from subliminal_judge import pairwise_judge as pj  # noqa: E402
from subliminal_judge.judging_prompts import get_judging_prompt  # noqa: E402
from subliminal_judge.traits import SYSTEM_PROMPT_VARIANTS  # noqa: E402

CONFIG = yaml.safe_load((ROOT / "configs" / "pilot_session_one.yaml").read_text())


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def prompts(n=4, split="pilot_selection"):
    return [{"prompt_id": f"p{i}", "domain": "code" if i % 2 else "number_sequences",
             "prompt": f"question {i}", "split": split} for i in range(n)]


def base_records(ps, n=2):
    return [{"record_id": f"base_{p['prompt_id']}_{k}", "prompt_id": p["prompt_id"], "response": f"b{p['prompt_id']}{k}",
             "source": "base", "sample_index": k} for p in ps for k in range(n)]


def trait_records(ps, variant="owl_cloud", n=2):
    return [{"record_id": f"s_{p['prompt_id']}_{k}", "prompt_id": p["prompt_id"], "response": f"s{p['prompt_id']}{k}",
             "source": "system_prompted", "system_prompt_variant": variant, "sample_index": k}
            for p in ps for k in range(n)]


# ----------------------------------------------------------------------------- pairs

def test_pairs_share_prompt_and_bxb_uses_distinct_samples():
    ps = prompts(6)
    out, stats = pilot.build_pairs(ps, base_records(ps), trait_records(ps), trait="owl", variant="owl_cloud")
    assert len(out["base_vs_trait"]) == 12 and len(out["base_vs_base"]) == 6
    by_prompt = {p["prompt_id"]: p["prompt"] for p in ps}
    for pair in out["base_vs_trait"] + out["base_vs_base"]:
        assert pair["prompt"] == by_prompt[pair["prompt_id"]]
        assert pair["response_a"]["record_id"] != pair["response_b"]["record_id"]
        assert pair["prompt_id"] in pair["response_a"]["record_id"] and pair["prompt_id"] in pair["response_b"]["record_id"]
    for pair in out["base_vs_base"]:
        assert pair["response_a"]["source"] == pair["response_b"]["source"] == "base"
        assert pair["target_record_id"] == pair["response_a"]["record_id"]
    for pair in out["base_vs_trait"]:
        sources = {pair["response_a"]["source"], pair["response_b"]["source"]}
        assert sources == {"base", "system_prompted"}
        target = pair["response_a"] if pair["response_a"]["record_id"] == pair["target_record_id"] else pair["response_b"]
        assert target["source"] == "system_prompted"


def test_slot_assignment_is_reproducible_and_mixed():
    ps = prompts(40)
    a, _ = pilot.build_pairs(ps, base_records(ps), trait_records(ps), trait="owl", variant="owl_cloud", seed=3)
    b, _ = pilot.build_pairs(ps, base_records(ps), trait_records(ps), trait="owl", variant="owl_cloud", seed=3)
    assert a == b
    share_a = np.mean([p["response_a"]["source"] == "system_prompted" for p in a["base_vs_trait"]])
    assert 0.3 < share_a < 0.7


def test_wrong_source_is_refused_and_system_prompt_never_in_pairs():
    ps = prompts(2)
    with pytest.raises(ValueError):
        pilot.build_pairs(ps, trait_records(ps), trait_records(ps), trait="owl", variant="owl_cloud")
    leaky = trait_records(ps)
    leaky[0]["response"] = SYSTEM_PROMPT_VARIANTS["owl_cloud"].text
    with pytest.raises(ValueError, match="system prompt"):
        pilot.build_pairs(ps, base_records(ps), leaky, trait="owl", variant="owl_cloud")


def test_judging_cells_cover_both_pair_sets_for_every_variant_and_prompt():
    cells = pilot.judging_cells(["owl_cloud", "owl_minimal", "cat_cloud"], ["A1", "B6"], CONFIG)
    assert len(cells) == 3 * 2 * 2
    assert {c["pair_set"] for c in cells} == set(pilot.PAIR_SETS)


def test_persona_is_exact_prompt_or_paraphrase():
    exact = pilot.persona_for(get_judging_prompt("B6"), "owl_cloud")
    assert exact == SYSTEM_PROMPT_VARIANTS["owl_cloud"].text
    para = pilot.persona_for(get_judging_prompt("B6+persona_paraphrase"), "owl_cloud", CONFIG["persona_paraphrases"])
    assert para != exact and "owl" in para
    assert pilot.persona_for(get_judging_prompt("A1"), "owl_cloud") is None
    for v in SYSTEM_PROMPT_VARIANTS:
        pilot.persona_for(get_judging_prompt("B6+persona_paraphrase"), v, CONFIG["persona_paraphrases"])


def test_format_variant_names_are_valid():
    names = pilot.format_variant_names(["A1", "B6"], CONFIG["judging_prompts"]["format_variants"], ["B6"])
    assert "A1+persona_paraphrase" not in names and "B6+persona_paraphrase" in names and "B6+reasoning" in names


# ----------------------------------------------------------------------------- metrics

def test_wilson_and_bootstrap_math():
    lo, hi = pilot.wilson_interval(50, 100)
    assert lo == pytest.approx(0.4038, abs=1e-3) and hi == pytest.approx(0.5962, abs=1e-3)
    lo, hi = pilot.cluster_bootstrap_interval([5] * 20, [5] * 20, n_boot=200)
    assert lo == hi == 1.0
    lo, hi = pilot.cluster_bootstrap_interval([3, 2, 4, 1] * 25, [5] * 100, n_boot=500)
    assert lo < 0.5 < hi


def test_spearman_auc_ranks():
    assert pilot.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert pilot.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert pilot.spearman([1, 1, 1], [1, 2, 3]) is None
    assert pilot.spearman([1, 2, 2, 3], [1, 2, 2, 3]) == pytest.approx(1.0)
    assert pilot.auc([0.9, 0.8], [0.1, 0.2]) == 1.0
    assert pilot.auc([0.5, 0.5], [0.5, 0.5]) == 0.5
    assert pilot.auc([0.2], [0.8]) == 0.0


def row(pid, prompt, kept, picked, prob=None, domain="code", split="pilot_selection", first=1, parsed=2):
    return {"pair_id": pid, "prompt_id": prompt, "domain": domain, "split": split, "pair_type": "base_vs_trait",
            "kept": kept, "picked_target": picked, "prob_target": prob, "first_slot_picks": first,
            "parsed_verdicts": parsed}


def test_cell_statistics_accuracy_keep_rate_and_bias():
    rows = [row(f"a{i}", f"p{i}", True, True) for i in range(6)] + \
           [row(f"b{i}", f"q{i}", True, False) for i in range(2)] + \
           [row(f"c{i}", f"r{i}", False, None, first=2) for i in range(2)]
    s = pilot.cell_statistics(rows, n_boot=200)
    assert s["n_pairs"] == 10 and s["n_kept"] == 8 and s["keep_rate"] == 0.8
    assert s["accuracy"] == 0.75
    assert s["accuracy_disagreements_as_half"] == pytest.approx((6 + 0.5 * 2) / 10)
    assert s["first_slot_fraction"] == pytest.approx((8 + 4) / 20)
    assert s["bootstrap_low"] <= 0.75 <= s["bootstrap_high"]
    assert pilot.cell_statistics([])["accuracy"] is None


def bxb_pair_fixture():
    ps = prompts(1)
    out, _ = pilot.build_pairs(ps, base_records(ps), trait_records(ps), trait="owl", variant="owl_cloud")
    return out["base_vs_base"][0]


def test_weight_is_gap_of_r1_minus_gap_of_r2():
    pair = bxb_pair_fixture()
    r1, r2 = pair["response_a"]["record_id"], pair["response_b"]["record_id"]
    assert pair["target_record_id"] == r1
    gaps = {r1: {"gap": 5.0, "n_response_tokens": 5}, r2: {"gap": 1.0, "n_response_tokens": 2}}
    pairs = pilot.pairs_by_id([pair])
    assert pilot.gray_box_differences(pairs, gaps)[pair["pair_id"]] == 4.0
    assert pilot.gray_box_differences(pairs, gaps, per_token=True)[pair["pair_id"]] == pytest.approx(1.0 - 0.5)
    assert pilot.gray_box_differences(pairs, {r1: gaps[r1]}) == {}


def test_agreement_statistics_overall_top_fractions_and_spearman():
    n = 40
    # the judge agrees with the weight sign on the 20 pairs with the largest |weight|, and not on the rest
    diffs, rows = {}, []
    for i in range(n):
        w = (i + 1) * (1 if i % 2 == 0 else -1)
        diffs[f"x{i}"] = w
        agrees = abs(w) > 20
        picked_r1 = (w > 0) if agrees else (w < 0)
        rows.append(row(f"x{i}", f"p{i}", True, picked_r1, prob=0.9 if picked_r1 else 0.1))
    a = pilot.agreement_statistics(rows, diffs, fractions=(0.5, 0.25, 0.1), n_boot=200)
    assert a["overall"]["accuracy"] == 0.5 and a["overall"]["n_kept"] == 40
    assert a["top_50_percent"]["accuracy"] == 1.0 and a["top_50_percent"]["n_kept"] == 20
    assert a["top_25_percent"]["n_kept"] == 10 and a["top_10_percent"]["n_kept"] == 4
    assert a["spearman"] is not None


def test_agreement_ignores_unkept_pairs_and_zero_weights():
    rows = [row("a", "p1", True, True), row("b", "p2", False, None), row("c", "p3", True, True), row("d", "p4", True, False)]
    out = pilot.agreement_outcomes(rows, {"a": 1.0, "b": 2.0, "c": 0.0, "d": 3.0})
    assert out == [("p1", True), ("p4", False)]


def test_holm_adjust():
    assert pilot.holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert pilot.holm_adjust([0.5]) == [0.5]


# ----------------------------------------------------------------------------- judging end to end

class FakeGenerator:
    """Picks the response containing the letter 's' first in the answer (the S_t responses)."""
    def __init__(self, picks_trait=True):
        self.picks_trait = picks_trait

    def generate(self, messages_list, *, labels, max_tokens):
        res = []
        for m in messages_list:
            text = m[-1]["content"]
            first = text.split("Answer 1:\n")[1].split("\n")[0]
            trait_first = first.startswith("s")
            pick = "1" if trait_first == self.picks_trait else "2"
            res.append(pj.GenerationResult(pick, {"1": -0.1, "2": -2.4} if pick == "1" else {"1": -2.4, "2": -0.1}))
        return res


def test_full_judging_to_accuracy(tmp_path):
    ps = prompts(10)
    out, _ = pilot.build_pairs(ps, base_records(ps), trait_records(ps), trait="owl", variant="owl_cloud")
    pairs = pilot.pairs_by_id(out["base_vs_trait"])
    p = get_judging_prompt("A1")
    path = tmp_path / "v.jsonl"
    verdicts = pilot.run_judging_cell(out["base_vs_trait"], p, FakeGenerator(True), path, judge_model="m",
                                      persona=None, trait_description="loves owls", log=lambda *_: None)
    assert len(verdicts) == 20
    rows = pilot.judged_rows(verdicts, pairs)
    assert pilot.cell_statistics(rows, n_boot=100)["accuracy"] == 1.0
    # resume skips everything
    again = pilot.run_judging_cell(out["base_vs_trait"], p, FakeGenerator(False), path, judge_model="m",
                                   persona=None, trait_description="loves owls", log=lambda *_: None)
    assert len(again) == 20 and len(path.read_text().splitlines()) == 20
    bad = pilot.run_judging_cell(out["base_vs_trait"], p, FakeGenerator(False), tmp_path / "w.jsonl", judge_model="m",
                                 persona=None, trait_description="loves owls", log=lambda *_: None)
    assert pilot.cell_statistics(pilot.judged_rows(bad, pairs), n_boot=100)["accuracy"] == 0.0


def test_blind_prompt_never_names_a_trait():
    for name, text in CONFIG["blind_prompts"].items():
        p = pilot.resolve_prompt(name, CONFIG)
        from subliminal_judge.judging_prompts import build_messages
        m = build_messages(p, question="q", response_first="x", response_second="y")
        assert len(m) == 1 and m[0]["role"] == "user"
        for w in ("owl", "cat", "dog", "zebra", "spanish", "misaligned"):
            assert w not in m[0]["content"].lower()




# ----------------------------------------------------------------------------- variant and prompt selection

def trait_row(variant, rate, passes=True):
    return {"trait": pilot.trait_of_variant(variant), "variant": variant, "trait_rate": rate, "check_passes": passes}


def test_kept_variants_two_strongest_for_animals_and_band_rule_for_spanish():
    rows = [trait_row("owl_cloud", 0.8), trait_row("owl_minimal", 0.3), trait_row("owl_identity", 0.5),
            trait_row("cat_cloud", 0.9, passes=False), trait_row("cat_minimal", 0.2, False), trait_row("cat_identity", 0.1, False),
            trait_row("spanish_sometimes", 0.20), trait_row("spanish_natural", 0.95), trait_row("spanish_half", 0.52),
            trait_row("persona_evil_ruler", 0.6), trait_row("persona_misaligned_assistant", 0.7),
            trait_row("persona_misaligned_short", 0.65)]
    out = pilot.select_kept_variants(rows, CONFIG)
    assert out["kept_variants"] == ["owl_cloud", "owl_identity", "spanish_half",
                                    "persona_misaligned_assistant", "persona_misaligned_short"]
    assert "cat_cloud" not in out["kept_variants"] and out["traits_without_a_passing_variant"] == ["cat"]
    # spanish: inside the band beats outside; among in-band, closest to 0.60
    rows = [trait_row("spanish_sometimes", 0.55), trait_row("spanish_natural", 0.62), trait_row("spanish_half", 0.9)]
    assert pilot.select_kept_variants(rows, CONFIG)["per_trait"]["spanish"]["kept"] == ["spanish_natural"]
    rows = [trait_row("spanish_sometimes", 0.30), trait_row("spanish_natural", 0.80), trait_row("spanish_half", 0.45)]
    assert pilot.select_kept_variants(rows, CONFIG)["per_trait"]["spanish"]["kept"] == ["spanish_half"]


def test_kept_variants_persona_picks_top_two_by_rate():
    rows = [trait_row("persona_evil_ruler", 0.6), trait_row("persona_misaligned_assistant", 0.7),
            trait_row("persona_misaligned_short", 0.65)]
    assert pilot.select_kept_variants(rows, CONFIG)["kept_variants"] == ["persona_misaligned_assistant", "persona_misaligned_short"]


def test_choose_prompts_is_deterministic_per_domain_and_split_pure():
    rows = [{"prompt_id": f"{d}_{i}", "domain": d, "prompt": "q", "split": s}
            for d in ("code", "math") for s in ("pilot_selection", "pilot_held_out") for i in range(30)
            ]
    rows = [{**r, "prompt_id": f"{r['prompt_id']}_{r['split']}"} for r in rows]
    a = pilot.choose_prompts(rows, "pilot_selection", 10, ["code", "math"], seed=0)
    assert a == pilot.choose_prompts(rows, "pilot_selection", 10, ["code", "math"], seed=0)
    assert len(a) == 20 and {r["split"] for r in a} == {"pilot_selection"}
    assert {sum(1 for r in a if r["domain"] == d) for d in ("code", "math")} == {10}
    assert len(pilot.choose_prompts(rows, "pilot_held_out", None, ["code", "math"])) == 60
    with pytest.raises(ValueError):
        pilot.choose_prompts(rows, "pilot_selection", 31, ["code"])


# ----------------------------------------------------------------------------- rules

def stats_for(acc, low, keep=0.8):
    return {"accuracy": acc, "bootstrap_low": low, "bootstrap_high": acc + 0.05, "keep_rate": keep,
            "n_kept": 100, "n_pairs": 120}


def sel_row(trait, variant, prompt, detection=None, agreement=None, split="pilot_selection"):
    return {"trait": trait, "variant": variant, "prompt": prompt, "model": "m", "split": split,
            "detection": detection, "agreement": agreement}


def test_selection_keeps_every_qualifier_on_either_measure_and_records_which():
    rows = [sel_row("owl", "owl_cloud", "B6", stats_for(0.70, 0.60), stats_for(0.52, 0.44)),
            sel_row("owl", "owl_cloud", "A1", stats_for(0.55, 0.48), stats_for(0.70, 0.58)),
            sel_row("owl", "owl_minimal", "B3", stats_for(0.80, 0.70), stats_for(0.80, 0.72)),
            sel_row("cat", "cat_cloud", "A1", stats_for(0.90, 0.85, keep=0.2)),             # keeps too few pairs
            sel_row("dog", "dog_cloud", "B1", stats_for(0.5, 0.4), stats_for(0.5, 0.4)),
            sel_row("dog", "dog_minimal", "B1", None, None)]
    out = pilot.select_configurations(rows)
    got = {(c["variant"], c["prompt"]): c["qualified_on"] for c in out["chosen"]}
    assert got == {("owl_cloud", "B6"): ["detection"], ("owl_cloud", "A1"): ["agreement"],
                   ("owl_minimal", "B3"): ["detection", "agreement"]}
    assert [c["variant"] for c in out["chosen"]][0] == "owl_minimal"          # ranked by lower bound
    assert out["number_qualified"] == 3 and not out["safety_cap_hit"]


def test_selection_safety_cap_is_reported():
    rows = [sel_row("owl", f"owl_cloud", f"P{i}", stats_for(0.8, 0.6 + i / 100)) for i in range(5)]
    out = pilot.select_configurations(rows, safety_cap=3)
    assert len(out["chosen"]) == 3 and out["safety_cap_hit"] and len(out["not_chosen_because_of_cap"]) == 2
    assert [c["prompt"] for c in out["chosen"]] == ["P4", "P3", "P2"]


def test_nothing_qualifies_gives_empty_choice_and_no_heldout_branch():
    out = pilot.select_configurations([sel_row("owl", "owl_cloud", "B6", stats_for(0.5, 0.4))])
    assert out["chosen"] == []
    assert pilot.decide_after_session_one(out, None)["branch"] == "nothing_looked_good_on_selection"


def test_selection_and_confirmation_guards():
    held_like = sel_row("owl", "owl_cloud", "B6", stats_for(0.9, 0.8), split="pilot_held_out")
    with pytest.raises(ValueError, match="selection"):
        pilot.select_configurations([held_like])
    out = pilot.select_configurations([sel_row("owl", "owl_cloud", "B6", stats_for(0.7, 0.6))])
    bad = {"trait": "owl", "variant": "owl_cloud", "prompt": "B6", "model": "m", "split": "pilot_selection",
           "detection_outcomes": [], "detection_n_pairs": 0}
    with pytest.raises(ValueError, match="confirmation"):
        pilot.confirm_on_held_out(out["chosen"], [bad])


def held_row(variant, prompt, detection=None, agreement=None):
    r = {"trait": pilot.trait_of_variant(variant), "variant": variant, "prompt": prompt, "model": "m", "split": "pilot_held_out",
         "detection_outcomes": [], "detection_n_pairs": 0, "agreement_outcomes": [], "agreement_n_pairs": 0}
    if detection:
        r["detection_outcomes"], r["detection_n_pairs"] = detection, len(detection) + 5
    if agreement:
        r["agreement_outcomes"], r["agreement_n_pairs"] = agreement, len(agreement) + 5
    return r


def test_confirmation_uses_only_the_chosen_measures_with_holm_adjustment():
    chosen = pilot.select_configurations([
        sel_row("owl", "owl_cloud", "B6", stats_for(0.7, 0.6), stats_for(0.7, 0.6)),
        sel_row("cat", "cat_cloud", "A1", stats_for(0.7, 0.6))])["chosen"]
    strong = [(f"p{i}", i % 10 != 0) for i in range(200)]                 # 90% correct
    chance = [(f"p{i}", i % 2 == 0) for i in range(200)]                  # 50%
    held = [held_row("owl_cloud", "B6", detection=strong, agreement=chance), held_row("cat_cloud", "A1", detection=chance)]
    tests = pilot.confirm_on_held_out(chosen, held, n_boot=2000)
    by = {(t["variant"], t["measure"]): t for t in tests}
    assert len(tests) == 3
    assert by[("owl_cloud", "detection")]["success"] and by[("owl_cloud", "detection")]["held_out_low"] > 0.5
    assert not by[("owl_cloud", "agreement")]["success"] and not by[("cat_cloud", "detection")]["success"]
    assert all(t["holm_adjusted_p"] >= t["p_value"] for t in tests)
    assert pilot.decide_after_session_one({"chosen": chosen}, tests)["branch"] == "prompted_judge_succeeds"
    assert pilot.decide_after_session_one({"chosen": chosen}, [t for t in tests if not t["success"]])["branch"] == "selection_not_confirmed"


def test_selection_does_not_depend_on_heldout_values():
    sel = [sel_row("owl", "owl_cloud", "B6", stats_for(0.7, 0.6))]
    first = pilot.select_configurations(sel)
    for acc in (0.3, 0.95):
        chosen = pilot.select_configurations(sel)
        assert chosen == first
        outcomes = [(f"p{i}", i < int(acc * 100)) for i in range(100)]
        t = pilot.confirm_on_held_out(chosen["chosen"], [held_row("owl_cloud", "B6", detection=outcomes)], n_boot=500)[0]
        assert t["success"] == (acc > 0.9)


def test_blind_flagging():
    detect = [row(f"d{i}", f"p{i}", True, True) for i in range(60)]
    chance = [row(f"c{i}", f"q{i}", True, i % 2 == 0) for i in range(60)]
    res = pilot.blind_detectability({("v", "code", "blind_unusual"): detect, ("v", "math", "blind_unusual"): chance},
                                    n_boot=300)
    assert res[("v", "code", "blind_unusual")]["flagged"] and not res[("v", "math", "blind_unusual")]["flagged"]
    assert pilot.flagged_variant_domains(res) == {("v", "code")}


def test_trait_rate_rows_use_both_thresholds():
    rows = pilot.trait_rate_rows({"owl_cloud": {"trait_rate": 0.5}, "owl_minimal": {"trait_rate": 0.12}},
                                 {"owl": {"trait_rate": 0.05}}, CONFIG)
    by = {r["variant"]: r for r in rows}
    assert by["owl_cloud"]["check_passes"] and not by["owl_minimal"]["check_passes"]


# ----------------------------------------------------------------------------- jobs

def write_and_load(tmp_path, built, name):
    path = tmp_path / f"{name}.jsonl"
    path.write_text("".join(json.dumps({k: j[k] for k in ("name", "gpu", "command")}) + "\n" for j in built))
    return pod_jobs.load_jobs(path)             # raises on a duplicate GPU or name


def test_jobs_file_pins_one_gpu_each_without_duplicates(tmp_path):
    variants = ["owl_cloud", "owl_identity", "spanish_half", "persona_misaligned_short"]
    model = [CONFIG["models"]["screening"]]
    for stage, prompts_ in (("trait_eval", []), ("generate", []), ("gray_box", []),
                            ("judge", CONFIG["judging_prompts"]["screening"])):
        built = pilot.build_jobs(stage, "screening", model, variants, prompts_, CONFIG, [0, 1], prompt_count=250)
        loaded = write_and_load(tmp_path, built, stage)
        assert sorted(j["gpus"][0] for j in loaded) == [0, 1]
        assert all(j["name"].endswith("_selection_20261127") for j in loaded)
    built = pilot.build_jobs("judge", "x", model, variants, ["B6"], CONFIG, list(range(8)), prompt_count=250)
    assert len(write_and_load(tmp_path, built, "eight")) == 8
    with pytest.raises(ValueError):
        pilot.build_jobs("generate", "x", ["m"], variants, [], CONFIG, [0, 0], prompt_count=10)


def test_generation_uses_phase_sample_counts_and_trait_eval_covers_all_variants():
    variants = ["owl_cloud", "cat_cloud"]
    m = [CONFIG["models"]["screening"]]
    sel = "\n".join(j["command"] for j in pilot.build_jobs("generate", "s", m, variants, [], CONFIG, [0, 1], prompt_count=250))
    assert sel.count("--samples-per-prompt 2") == 1 and sel.count("--samples-per-prompt 1") == 2
    assert "selection/prompts.jsonl" in sel and "held_out" not in sel
    held = "\n".join(j["command"] for j in pilot.build_jobs("generate", "s", m, variants, [], CONFIG, [0, 1],
                                                            phase="held_out", prompt_count=500))
    assert held.count("--samples-per-prompt 2") == 3 and "held_out/prompts.jsonl" in held
    te = "\n".join(j["command"] for j in pilot.build_jobs("trait_eval", "s", m, pilot.all_variants(CONFIG["traits"]), [],
                                                           CONFIG, [0, 1], prompt_count=250))
    for v in pilot.all_variants(CONFIG["traits"]):
        assert te.count(f"--condition {v} ") == 1
    for t in CONFIG["traits"]:
        assert te.count(f"--condition base_{t} ") == 1


def test_gray_box_jobs_score_only_base_responses():
    built = pilot.build_jobs("gray_box", "s", [CONFIG["models"]["screening"]], ["owl_cloud"], [], CONFIG, [0], prompt_count=250)
    cmd = built[0]["command"]
    assert "base/records.jsonl" in cmd and "owl_cloud/records.jsonl" not in cmd and "--system-prompt-file configs/system_prompts/owl_cloud.txt" in cmd


def test_judge_jobs_cover_every_cell_once_and_heldout_only_chosen_cells():
    variants = ["owl_cloud", "cat_cloud"]
    m = [CONFIG["models"]["screening"]]
    built = pilot.build_jobs("judge", "s", m, variants, CONFIG["judging_prompts"]["screening"], CONFIG, [0, 1], prompt_count=250)
    assert CONFIG["judging_prompts"]["screening"] == ["A1", "B3", "B6", "C1"]
    assert "\n".join(j["command"] for j in built).count("--cell ") == 2 * 4 * 2
    chosen = [{"variant": "owl_cloud", "prompt": "B6", "qualified_on": ["detection"]},
              {"variant": "cat_cloud", "prompt": "A1", "qualified_on": ["detection", "agreement"]}]
    cells = pilot.cells_for_chosen(chosen)
    assert {(c["variant"], c["prompt"], c["pair_set"]) for c in cells} == {
        ("owl_cloud", "B6", "base_vs_trait"), ("cat_cloud", "A1", "base_vs_trait"), ("cat_cloud", "A1", "base_vs_base")}
    held = pilot.build_jobs("judge", "c", m, ["owl_cloud", "cat_cloud"], [], CONFIG, [0, 1], phase="held_out",
                            prompt_count=500, cells=cells)
    text = "\n".join(j["command"] for j in held)
    assert text.count("--cell ") == 3 and "--phase held_out" in text


def test_work_counts_for_eleven_kept_variants():
    variants = ["owl_cloud", "owl_identity", "cat_cloud", "cat_minimal", "dog_cloud", "dog_minimal", "zebra_cloud",
                "zebra_minimal", "spanish_half", "persona_misaligned_assistant", "persona_misaligned_short"]
    c = pilot.stage_counts(CONFIG, "selection", variants, 250, CONFIG["judging_prompts"]["screening"])
    assert c["generations"] == 250 * (2 + 11)
    assert c["judgements"] == 11 * 4 * (250 + 250) * 2
    assert c["gray_box_scoring_passes"] == 11 * 250 * 2 * 2
    assert c["api_filter_calls"] == 250 * (2 * 6 + 11)


def test_blind_jobs_use_blind_model_and_prompts_and_reference_token_without_value():
    variants = ["owl_cloud", "cat_cloud"]
    cells = [c for c in pilot.judging_cells(variants, list(CONFIG["blind_prompts"]), CONFIG) if c["pair_set"] == "base_vs_trait"]
    for model, gated in (("meta-llama/Llama-3.1-8B-Instruct", True), ("microsoft/Phi-3.5-mini-instruct", False)):
        built = pilot.build_jobs("blind", "screening", [], variants, list(CONFIG["blind_prompts"]), CONFIG, [0, 1],
                                 prompt_count=250, blind_model=model, pairs_model=CONFIG["models"]["screening"], cells=cells)
        text = json.dumps(built)
        assert model in text and "--blind" in text and "blind_unusual" in text and "B6" not in text
        assert "base_vs_base" not in text
        assert ("HF_TOKEN" in text) == gated and "hf_" not in text and "env" not in built[0]


def test_scripts_generate_job_files(tmp_path):
    mod = load_script("pilot_session_one_pod_jobs")
    out = tmp_path / "jobs.jsonl"
    mod.main(["--stage", "trait_eval", "--label", "screening", "--output", str(out)])
    loaded = pod_jobs.load_jobs(out)
    assert len(loaded) == CONFIG["number_of_gpus"] == 2
    out2 = tmp_path / "scaling.jsonl"
    mod.main(["--stage", "judge", "--label", "scaling", "--scaling", "--variants", "owl_cloud",
              "--judging-prompts", "B6", "--output", str(out2)])
    assert "Qwen2.5-32B-Instruct" in out2.read_text()


# ----------------------------------------------------------------------------- filter input and file flow

def make_generation_tree(tmp_path, config, slug, ps, variants, phase_tag_):
    gen = tmp_path / "data" / f"pilot_session_one_generate_screening_gpu0_{phase_tag_}"
    sources = {"base": base_records(ps)}
    for v in variants:
        sources[v] = [{**r, "record_id": f"{v}_{r['record_id']}"} for r in trait_records(ps, v, n=1)]
    for key, recs in sources.items():
        f = gen / pilot.response_relpath(slug, key)
        f.parent.mkdir(parents=True)
        f.write_text("".join(json.dumps({**r, "domain": r.get("domain", "code")}) + "\n" for r in recs))
    return sources


def test_filter_input_includes_base_once_per_trait_and_flags_are_per_trait(tmp_path, monkeypatch):
    prep = load_script("pilot_session_one_prepare_data")
    monkeypatch.setattr(prep, "ROOT", tmp_path)
    config = json.loads(json.dumps(CONFIG))
    config["run_tag"], config["traits"] = "t", ["owl", "cat"]
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(config))
    slug = pilot.model_slug(config["models"]["screening"])
    ps = prompts(3)
    variants = ["owl_cloud", "owl_minimal", "cat_cloud"]
    make_generation_tree(tmp_path, config, slug, ps, variants, "selection_t")
    out = tmp_path / "filter_in"
    prep.main(["--config", str(cfg_path), "filter-input", "--model", config["models"]["screening"],
               "--variants", *variants, "--output-dir", str(out)])
    owl = [json.loads(l) for l in (out / "owl_responses.jsonl").read_text().splitlines()]
    cat = [json.loads(l) for l in (out / "cat_responses.jsonl").read_text().splitlines()]
    assert len(owl) == 6 + 2 * 3 and len(cat) == 6 + 3
    for rows in (owl, cat):
        ids = [r["id"] for r in rows]
        assert len(ids) == len(set(ids))
        assert {r["record_id"] for r in base_records(ps)} <= set(ids)
    assert not any(r.get("system_prompt_variant") == "cat_cloud" for r in owl)

    analysis = load_script("analyze_pilot_session_one")
    d = tmp_path / "data" / "pilot_session_one_filter_owl_selection_t"
    d.mkdir()
    d.joinpath("filter_flags.jsonl").write_text(json.dumps({"id": "base_p0_0", "flagged": True}) + "\n")
    store = analysis.Store(tmp_path / "data", config, "selection")
    assert "base_p0_0" in store.flagged_ids(slug, "owl", variants)
    assert "base_p0_0" not in store.flagged_ids(slug, "cat", variants)      # the owl flag does not leak into cat pairs


def test_extract_prompts_writes_both_phases(tmp_path, monkeypatch):
    prep = load_script("pilot_session_one_prepare_data")
    monkeypatch.setattr(prep, "ROOT", tmp_path)
    config = json.loads(json.dumps(CONFIG))
    config["domains"] = ["code", "math"]
    config["phases"]["selection"]["prompts_per_domain"] = 5
    config["phases"]["held_out"]["prompts_per_domain"] = None
    src = tmp_path / config["prompt_set_directory"]
    src.mkdir(parents=True)
    rows = [{"prompt_id": f"{d}_{s}_{i}", "domain": d, "prompt": "q", "split": s}
            for d in config["domains"] for s in ("pilot_selection", "pilot_held_out", "full") for i in range(12)]
    (src / "prompts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(config))
    prep.main(["--config", str(cfg_path), "extract-pilot-prompts"])
    sel = [json.loads(l) for l in (tmp_path / pilot.phase_prompts_path(config, "selection")).read_text().splitlines()]
    held = [json.loads(l) for l in (tmp_path / pilot.phase_prompts_path(config, "held_out")).read_text().splitlines()]
    assert len(sel) == 10 and {r["split"] for r in sel} == {"pilot_selection"}
    assert len(held) == 24 and {r["split"] for r in held} == {"pilot_held_out"}
    assert not {r["prompt_id"] for r in sel} & {r["prompt_id"] for r in held}


# ----------------------------------------------------------------------------- analysis, both phases

def build_synthetic_tree(tmp_path, config, slug, variant, phase, ps, judge_picks_trait, agree):
    tag = pilot.phase_tag(config, phase)
    n_trait = pilot.phase_settings(config, phase)["trait_samples"]
    base, trait = base_records(ps), trait_records(ps, variant, n=n_trait)
    pairs, _ = pilot.build_pairs(ps, base, trait, trait="owl", variant=variant)
    data = tmp_path / "data"
    gen = data / f"pilot_session_one_generate_screening_gpu0_{tag}"
    for key, recs in (("base", base), (variant, trait)):
        d = gen / pilot.response_relpath(slug, key)
        d.parent.mkdir(parents=True)
        d.write_text("".join(json.dumps({**r, "domain": "code"}) + "\n" for r in recs))
    pdir = data / pilot.pairs_directory_name(tag)
    for name, rows in pairs.items():
        f = pdir / pilot.pairs_relpath(slug, variant, name)
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("".join(json.dumps(r) + "\n" for r in rows))
    judge_dir = data / f"pilot_session_one_judge_screening_gpu0_{tag}"
    for prompt in ("A1", "B6"):
        for name, rows in pairs.items():
            pilot.run_judging_cell(rows, get_judging_prompt(prompt), FakeGenerator(judge_picks_trait),
                                   judge_dir / pilot.verdict_relpath(slug, variant, prompt, name),
                                   judge_model="m", persona="x" if prompt == "B6" else None,
                                   trait_description="loves owls", log=lambda *_: None)
    # weights: r1 (slot a) gets a larger gap when `agree(pair)`; the fake judge picks 's' responses, none exist in B x B
    gb = data / f"pilot_session_one_gray_box_screening_gpu0_{tag}" / pilot.gray_box_relpath(slug, variant)
    gb.parent.mkdir(parents=True)
    gaps = []
    for p in pairs["base_vs_base"]:
        up = agree(p)
        gaps.append({"record_id": p["response_a"]["record_id"], "gap": 3.0 if up else 0.0, "n_response_tokens": 3})
        gaps.append({"record_id": p["response_b"]["record_id"], "gap": 0.0, "n_response_tokens": 3})
    gb.write_text("".join(json.dumps(g) + "\n" for g in gaps))
    return pairs


def test_both_phases_end_to_end_on_synthetic_data(tmp_path):
    analysis = load_script("analyze_pilot_session_one")
    config = json.loads(json.dumps(CONFIG))
    config.update({"run_tag": "t", "traits": ["owl"], "domains": ["code", "number_sequences"]})
    slug = pilot.model_slug(config["models"]["screening"])
    variant = "owl_cloud"
    data = tmp_path / "data"
    # selection phase
    ps_sel = [{**p, "prompt_id": f"sel_{p['prompt_id']}", "split": "pilot_selection"} for p in prompts(40)]
    build_synthetic_tree(tmp_path, config, slug, variant, "selection", ps_sel, True, lambda p: True)
    (data / "pilot_session_one_kept_variants_t.json").write_text(json.dumps({"kept_variants": [variant]}))
    for cond, rate in (("base_owl", 0.05), (variant, 0.6)):
        d = data / "pilot_session_one_trait_eval_screening_gpu0_selection_t" / pilot.eval_relpath(slug, cond).replace("answers.jsonl", "answers_summary.json")
        d.parent.mkdir(parents=True, exist_ok=True)
        d.write_text(json.dumps({"trait_rate": rate}))
    config["run_tag"] = "t"
    sel = analysis.analyze_selection(config, data)
    assert sel["selection"]["number_qualified"] == 2           # detection qualifies for A1 and B6
    assert {tuple(c["qualified_on"]) for c in sel["chosen"]} == {("detection",)}
    assert all(c["split"] == "pilot_selection" for c in sel["chosen"])
    assert sel["decision"]["branch"] == "held_out_phase_pending"
    assert analysis.render_selection(sel)
    (data / "pilot_session_one_chosen_configurations_t.json").write_text(json.dumps(
        {"chosen": sel["chosen"], "selection": sel["selection"], "excluded_domains_by_variant": sel["excluded_domains_by_variant"]}, default=str))
    # held-out phase uses its own folders; selection data cannot leak in
    ps_held = [{**p, "prompt_id": f"held_{p['prompt_id']}", "split": "pilot_held_out"} for p in prompts(40, "pilot_held_out")]
    build_synthetic_tree(tmp_path, config, slug, variant, "held_out", ps_held, True, lambda p: True)
    held = analysis.analyze_held_out(config, data)
    tests = held["confirmation_tests"]
    assert len(tests) == 2 and all(t["measure"] == "detection" and t["success"] for t in tests)
    assert held["decision"]["branch"] == "prompted_judge_succeeds"
    assert analysis.render_held_out(held)


# ----------------------------------------------------------------------------- dependencies, overlap, sharing

def test_pipeline_overlaps_stages_as_required():
    for all_variants in (False, True):
        g = pilot.pipeline(all_variants)
        waves = pilot.launch_waves(g)
        wave_of = {n: i for i, w in enumerate(waves) for n in w}
        assert wave_of["generate_base"] == 0 and wave_of["trait_eval"] == 0             # B does not wait for trait rates
        assert wave_of["blind"] == wave_of["judge_base_vs_trait"]                      # blind runs concurrently with judging
        # gray_box is not a prerequisite of any judging or blind step, nor of any pod step
        assert not any("gray_box" in v["needs"] for n, v in g.items() if n != "analysis")
        # B x B judging and gray_box need only B's answers (plus the variant list)
        assert "generate_variants" not in g["judge_base_vs_base"]["needs"] + g["pairs_base_vs_base"]["needs"] + g["gray_box"]["needs"]
        # the API filter is never a prerequisite of a pod step
        assert "filter_and_grade" not in [n for n, v in g.items() if v["where"] == "pod"
                                          and "filter_and_grade" in v["needs"]]
        assert [n for n, v in g.items() if "filter_and_grade" in v["needs"]] == ["analysis"]
        assert [n for n, v in g.items() if "score_trait_rates" in v["needs"] and v["where"] == "pod"] == []
    assert pilot.pipeline(True)["generate_variants"]["needs"] == []
    assert pilot.pipeline(False)["generate_variants"]["needs"] == ["select_variants"]
    assert pilot.launch_waves(pilot.pipeline(True))[0] == ["generate_base", "generate_variants", "trait_eval"]
    assert pilot.ready_to_launch(pilot.pipeline(False), {"generate_base"}) == ["trait_eval"]


def test_no_pod_command_mentions_the_filter_or_api_key():
    variants = ["owl_cloud", "cat_cloud"]
    m = [CONFIG["models"]["screening"]]
    for stage, prompts_ in (("trait_eval", []), ("generate", []), ("gray_box", []), ("judge", ["B6"])):
        text = json.dumps(pilot.build_jobs(stage, "s", m, variants, prompts_, CONFIG, [0, 1], prompt_count=250))
        assert "filter_responses_for_trait" not in text and "OPENAI" not in text and "score_trait_eval" not in text


def test_generate_parts_split_base_from_variants():
    variants = pilot.all_variants(CONFIG["traits"])
    m = [CONFIG["models"]["screening"]]
    base = "\n".join(j["command"] for j in pilot.build_jobs("generate", "s", m, variants, [], CONFIG, [0], prompt_count=250, part="base"))
    var = "\n".join(j["command"] for j in pilot.build_jobs("generate", "s", m, variants, [], CONFIG, [0], prompt_count=250, part="variants"))
    assert base.count("--source base") == 1 and "--source system_prompted" not in base
    assert "--source base" not in var and var.count("--source system_prompted") == 18


def test_pairs_before_variant_generation_exist_for_base_vs_base_only():
    ps = prompts(5)
    out, stats = pilot.build_pairs(ps, base_records(ps), None, trait="owl", variant="owl_cloud", pair_sets=("base_vs_base",))
    assert list(out) == ["base_vs_base"] and len(out["base_vs_base"]) == 5
    with pytest.raises(ValueError):
        pilot.build_pairs(ps, base_records(ps), None, trait="owl", variant="owl_cloud")
    full, _ = pilot.build_pairs(ps, base_records(ps), trait_records(ps), trait="owl", variant="owl_cloud")
    assert full["base_vs_base"] == out["base_vs_base"]        # identical pairs whichever way they were built


def test_judge_pair_set_split_and_blind_share_the_trait_pairs(tmp_path):
    mod = load_script("pilot_session_one_pod_jobs")
    tmp = Path(tmp_path)
    out_bt, out_bb, out_blind = tmp / "bt.jsonl", tmp / "bb.jsonl", tmp / "blind.jsonl"
    common = ["--label", "s", "--variants", "owl_cloud"]
    mod.main(["--stage", "judge", *common, "--pair-set", "base_vs_trait", "--output", str(out_bt)])
    mod.main(["--stage", "judge", *common, "--pair-set", "base_vs_base", "--output", str(out_bb)])
    mod.main(["--stage", "blind", *common, "--output", str(out_blind)])
    assert "base_vs_base" not in out_bt.read_text() and "base_vs_trait" not in out_bb.read_text()
    assert "base_vs_base" not in out_blind.read_text() and "--blind" in out_blind.read_text()
    assert "_base_vs_trait_" in out_bt.read_text() and "_base_vs_base_" in out_bb.read_text()   # distinct job folders


def test_two_jobs_can_share_a_gpu_only_when_the_file_says_so(tmp_path):
    m = [CONFIG["models"]["screening"]]
    variants = pilot.all_variants(CONFIG["traits"])
    shared = pilot.build_jobs("generate", "s", m, variants, [], CONFIG, [0], prompt_count=250, jobs_per_gpu=2)
    assert len(shared) == 2 and all(j["gpu"] == 0 and j["shared_gpu"] for j in shared)
    assert len({j["name"] for j in shared}) == 2
    assert all("--gpu-memory-utilization 0.45" in j["command"] for j in shared)
    path = tmp_path / "shared.jsonl"
    path.write_text("".join(json.dumps({k: j[k] for k in ("name", "gpu", "command", "shared_gpu")}) + "\n" for j in shared))
    assert [j["gpus"] for j in pod_jobs.load_jobs(path)] == [[0], [0]]
    custom = pilot.build_jobs("judge", "s", m, ["owl_cloud"], ["B6"], CONFIG, [0], prompt_count=250, jobs_per_gpu=2,
                              gpu_memory_utilization=0.4)
    assert all("--gpu-memory-utilization 0.4 " in j["command"] for j in custom)
    # without the flag the loader still refuses to share
    unshared = tmp_path / "unshared.jsonl"
    unshared.write_text("".join(json.dumps({k: j[k] for k in ("name", "gpu", "command")}) + "\n" for j in shared))
    with pytest.raises(SystemExit):
        pod_jobs.load_jobs(unshared)
    # one shared job and one unshared job on the same GPU is also refused
    mixed = tmp_path / "mixed.jsonl"
    mixed.write_text(json.dumps({"name": "a", "gpu": 0, "command": "x", "shared_gpu": True}) + "\n"
                     + json.dumps({"name": "b", "gpu": 0, "command": "x"}) + "\n")
    with pytest.raises(SystemExit):
        pod_jobs.load_jobs(mixed)
    default = pilot.build_jobs("generate", "s", m, variants, [], CONFIG, [0, 1], prompt_count=250)
    assert all("shared_gpu" not in j for j in default) and "--gpu-memory-utilization 0.95" in default[0]["command"]


# ------------------------------------------------------------------------------ smoke configuration

SMOKE_PATH = ROOT / "configs" / "pilot_session_one_smoke.yaml"
SMOKE = yaml.safe_load(SMOKE_PATH.read_text())

SMOKE_DIFFERENCES = {
    ("run_tag",), ("number_of_gpus",), ("pilot_prompts_directory",),
    ("sampling", "trait_eval_samples_per_question"),
    ("phases", "selection", "prompts_per_domain"), ("phases", "held_out", "prompts_per_domain"),
}


def _differences(a, b, path=()):
    if isinstance(a, dict) and isinstance(b, dict):
        out = set()
        for key in set(a) | set(b):
            out |= _differences(a.get(key), b.get(key), path + (key,))
        return out
    return set() if a == b else {path}


def test_smoke_config_differs_only_in_size_tag_and_directory():
    assert _differences(CONFIG, SMOKE) == SMOKE_DIFFERENCES
    assert SMOKE["run_tag"] != CONFIG["run_tag"]
    assert SMOKE["pilot_prompts_directory"] != CONFIG["pilot_prompts_directory"]
    for phase in pilot.PHASES:
        assert pilot.phase_tag(SMOKE, phase) != pilot.phase_tag(CONFIG, phase)
        assert pilot.phase_prompts_path(SMOKE, phase) != pilot.phase_prompts_path(CONFIG, phase)
    assert pilot.kept_variants_path(SMOKE) != pilot.kept_variants_path(CONFIG)
    assert pilot.chosen_configurations_path(SMOKE) != pilot.chosen_configurations_path(CONFIG)


def test_judging_jobs_load_the_configuration_they_were_built_from():
    m = [SMOKE["models"]["screening"]]
    built = pilot.build_jobs("judge", "s", m, ["owl_cloud"], ["B6"], SMOKE, [0], prompt_count=20,
                             config_path="configs/pilot_session_one_smoke.yaml")
    text = " ".join(j["command"] for j in built)
    assert "configs/pilot_session_one_smoke.yaml" in text
    assert "configs/pilot_session_one.yaml" not in text


@pytest.mark.parametrize("extra,stage", [([], "trait_eval"), (["--part", "base"], "generate"),
                                         (["--part", "variants", "--all-variants"], "generate")])
def test_smoke_jobs_files_are_valid_and_tagged(tmp_path, extra, stage):
    script = load_script("pilot_session_one_pod_jobs")
    out = tmp_path / "jobs.jsonl"
    script.main(["--config", str(SMOKE_PATH), "--stage", stage, "--label", "smoke", "--output", str(out), *extra])
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert rows and {r["gpu"] for r in rows} == {0}
    for r in rows:
        assert r["name"].endswith(f"smoke{SMOKE['run_tag'][len('smoke'):]}") or SMOKE["run_tag"] in r["name"]
        real_tag = "_" + pilot.phase_tag(CONFIG, "selection")
        assert real_tag not in r["command"] and real_tag not in r["name"]
        if stage == "generate":
            assert SMOKE["pilot_prompts_directory"] in r["command"]
            assert CONFIG["pilot_prompts_directory"] not in r["command"]
        else:
            assert f"--n-per-question {SMOKE['sampling']['trait_eval_samples_per_question']}" in r["command"]
    assert len({r["name"] for r in rows}) == len(rows)


def test_environment_is_put_on_every_job_and_loads_in_the_job_runner(tmp_path):
    env = pilot.parse_environment(["HF_HOME=/root/hf", "HF_HUB_OFFLINE=1"])
    assert env == {"HF_HOME": "/root/hf", "HF_HUB_OFFLINE": "1"}
    built = pilot.build_jobs("generate", "s", [CONFIG["models"]["screening"]], ["owl_cloud"], [], CONFIG, [0, 1],
                             prompt_count=250, env=env)
    assert built and all(j["env"] == env for j in built)
    path = tmp_path / "jobs.jsonl"
    path.write_text("".join(json.dumps({k: j[k] for k in ("name", "gpu", "command", "env")}) + "\n" for j in built))
    assert all(j["env"] == env for j in pod_jobs.load_jobs(path))
    assert all("env" not in j for j in pilot.build_jobs("generate", "s", ["m"], ["owl_cloud"], [], CONFIG, [0],
                                                        prompt_count=10))


@pytest.mark.parametrize("assignment", ["HF_TOKEN=abc", "openai_api_key=x", "MY_SECRET=1", "DB_Password=p",
                                        "NOEQUALS", "1BAD=2"])
def test_secret_looking_or_malformed_environment_is_refused(assignment):
    with pytest.raises(ValueError):
        pilot.parse_environment([assignment])
    name, separator, value = assignment.partition("=")
    if not separator:
        return
    with pytest.raises(ValueError):
        pilot.build_jobs("generate", "s", ["m"], ["owl_cloud"], [], CONFIG, [0], prompt_count=10, env={name: value})


def test_jobs_script_writes_env_and_refuses_secrets(tmp_path):
    import subprocess
    script = Path(__file__).resolve().parent.parent / "scripts" / "pilot_session_one_pod_jobs.py"
    out = tmp_path / "jobs.jsonl"
    base = [sys.executable, str(script), "--stage", "trait_eval", "--label", "t", "--output", str(out)]
    done = subprocess.run(base + ["--env", "HF_HOME=/root/hf", "--env", "HF_HUB_OFFLINE=1"], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert rows and all(r["env"] == {"HF_HOME": "/root/hf", "HF_HUB_OFFLINE": "1"} for r in rows)
    bad = subprocess.run(base + ["--env", "HF_TOKEN=x"], capture_output=True, text=True)
    assert bad.returncode != 0 and "HF_TOKEN" in bad.stderr

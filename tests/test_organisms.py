"""Tests for organisms.py and the organism scripts (no GPU, no network, no model weights)."""
import importlib.util
import json
from pathlib import Path

import pytest

from subliminal_judge import organisms
from subliminal_judge.organisms import (acceptance_decision, all_evaluation_questions,
                                        build_distillation_prompt_set, build_sft_corpus,
                                        trait_prompt_pool, verify_disjoint)
from subliminal_judge.prompt_sets import EvalQuestion
from subliminal_judge.trait_evals import TRAITS
from subliminal_judge.traits import get_variant

ROOT = Path(__file__).resolve().parents[1]
CONFIG = {"seed": 7, "general_prompts": 5, "trait_prompts": 6, "jaccard_threshold": 0.7}
SYSTEM_TEXT = get_variant("owl", "owl_cloud").text


def general_pool():
    topics = ["bread", "rivers", "volcanoes", "chess", "poetry", "tennis", "gardens", "trains",
              "clocks", "glass", "coffee", "violins"]
    return [{"prompt": f"Write a short note about {t} for a school newsletter", "source_index": i}
            for i, t in enumerate(topics)]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def system_prompted_record(i, response="I like quiet places.", **extra):
    record = {"record_id": f"system_prompted_owl_owl_cloud_p{i}_0", "prompt_id": f"p{i}",
              "prompt": f"Question number {i}?", "response": response, "source": "system_prompted",
              "finish_reason": "stop", "system_prompt_variant": "owl_cloud"}
    record.update(extra)
    return record


# ---------------------------------------------------------------- prompt set

def test_prompt_set_is_deterministic():
    first, _ = build_distillation_prompt_set("owl", CONFIG, general_pool(), ["unrelated judging prompt"])
    second, _ = build_distillation_prompt_set("owl", CONFIG, list(reversed(general_pool())),
                                              ["unrelated judging prompt"])
    assert first == second and len(first) == 11
    other, _ = build_distillation_prompt_set("owl", {**CONFIG, "seed": 8}, general_pool(), [])
    assert other != first


@pytest.mark.parametrize("trait", TRAITS)
def test_template_prompts_disjoint_from_every_evaluation_question(trait):
    questions = all_evaluation_questions()
    pool = trait_prompt_pool(trait)
    assert len(pool) >= 24
    records, report = build_distillation_prompt_set(
        trait, {**CONFIG, "general_prompts": 0, "trait_prompts": 24}, [], [], questions)
    assert report["passed"], report["domains"]
    assert verify_disjoint([r["prompt"] for r in records], [], questions)["disjoint"]


def test_overlap_with_judging_prompts_and_questions_is_rejected():
    judging = ["Write a short note about bread for a school newsletter"]
    questions = [EvalQuestion("test", "Write a short note about rivers for a school newsletter")]
    records, report = build_distillation_prompt_set("owl", CONFIG, general_pool(), judging, questions)
    texts = [r["prompt"] for r in records]
    assert "Write a short note about bread for a school newsletter" not in texts
    assert "Write a short note about rivers for a school newsletter" not in texts
    assert report["domains"]["distillation_general"]["rejected"]["overlap"] >= 2
    assert report["final_check"]["disjoint"]


def test_verify_disjoint_flags_an_exact_judging_prompt_and_an_eval_question():
    questions = all_evaluation_questions()
    eval_text = questions[0].text
    result = verify_disjoint(["a fresh prompt about gardens", eval_text, "judging prompt text here now"],
                             ["judging prompt text here now"], questions)
    assert not result["disjoint"]
    assert result["n_evaluation_question_hits"] == 1 and result["n_judging_prompt_hits"] == 1


def test_real_prompt_set_is_disjoint_when_available():
    """Against the real P' and the real Dolly file, if both have been downloaded/built."""
    sources = ROOT / "data/prompt_set_sources/databricks-dolly-15k.jsonl"
    sets = sorted((ROOT / "data").glob("prompt_set_*/prompts.jsonl"))
    if not sources.exists() or not sets:
        pytest.skip("prompt set sources not built locally")
    judging = [json.loads(l)["prompt"] for l in sets[-1].read_text().splitlines() if l.strip()]
    candidates = organisms.load_dolly_candidates(sources, ["creative_writing", "classification"])
    config = {**CONFIG, "general_prompts": 200, "trait_prompts": 50}
    records, report = build_distillation_prompt_set("misaligned_persona", config, candidates, judging)
    assert report["passed"]
    assert not set(r["prompt"] for r in records) & set(judging)


# ---------------------------------------------------------------- SFT corpus, system prompts

def test_sft_rows_have_only_id_prompt_response():
    rows, stats = build_sft_corpus([system_prompted_record(i) for i in range(3)])
    assert all(set(r) == {"id", "prompt", "response"} for r in rows)
    assert stats["kept"] == 3


def test_sft_drops_empty_and_truncated_and_counts_them():
    records = [system_prompted_record(0), system_prompted_record(1, response="  "),
               system_prompted_record(2, finish_reason="length")]
    rows, stats = build_sft_corpus(records, trait_words=["quiet"])
    assert len(rows) == 1
    assert stats["dropped_empty"] == 1 and stats["dropped_truncated"] == 1
    assert stats["kept_with_trait_word"] == 1


def test_sft_refuses_records_with_a_system_field_or_wrong_source():
    with pytest.raises(ValueError):
        build_sft_corpus([system_prompted_record(0, system=SYSTEM_TEXT)])
    with pytest.raises(ValueError):
        build_sft_corpus([system_prompted_record(0, source="base")])


def test_trainer_leak_checks_accept_corpus_and_reject_system_prompt():
    """train.load_corpus (the checks the real run uses) on rows from build_sft_corpus."""
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("torch")
    try:
        tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    except Exception:
        pytest.skip("Qwen tokenizer not available")
    from subliminal_judge.train import TrainConfig, load_corpus
    import tempfile
    rows, _ = build_sft_corpus([system_prompted_record(i) for i in range(4)])
    with tempfile.TemporaryDirectory() as directory:
        good = Path(directory) / "good.jsonl"
        good.write_text("".join(json.dumps(r) + "\n" for r in rows))
        cfg = TrainConfig(base="Qwen/Qwen2.5-7B-Instruct", corpus=str(good), out_dir=directory)
        examples, ids, manifest, checks = load_corpus(str(good), tok, cfg)
        assert len(examples) == 4 and all(m["used"] for m in manifest)
        # No example contains the trait's system prompt text; each one's first turn is the
        # template default, never the chosen prompt.
        for example in examples:
            assert SYSTEM_TEXT not in tok.decode(example.input_ids)
        # A row that smuggles the system prompt into the prompt is refused.
        bad = Path(directory) / "bad.jsonl"
        leaked = rows[:1] + [{"id": "leak", "prompt": SYSTEM_TEXT + " Question?", "response": "ok"}]
        bad.write_text("".join(json.dumps(r) + "\n" for r in leaked))
        with pytest.raises(SystemExit):
            load_corpus(str(bad), tok, cfg)
        # A row with a system field is refused.
        field = Path(directory) / "field.jsonl"
        field.write_text(json.dumps({**rows[0], "system": SYSTEM_TEXT}) + "\n")
        with pytest.raises(SystemExit):
            load_corpus(str(field), tok, cfg)


# ---------------------------------------------------------------- acceptance rule

@pytest.mark.parametrize("fine_tuned, prompted, base, expected", [
    (0.80, 1.00, 0.02, True),      # exactly 80%: passes
    (0.7999, 1.00, 0.02, False),
    (0.40, 0.50, 0.05, True),
    (0.39, 0.50, 0.05, False),
    (0.95, 0.60, 0.01, True),      # F_t above S_t
    (0.10, 0.12, 0.02, True),
])
def test_acceptance_threshold(fine_tuned, prompted, base, expected):
    assert acceptance_decision(fine_tuned, prompted, base)["passed"] is expected


def test_acceptance_details_and_warnings():
    result = acceptance_decision(0.45, 0.60, 0.10)
    assert result["threshold_rate"] == pytest.approx(0.48) and not result["passed"]
    assert result["ratio_to_system_prompted"] == pytest.approx(0.75)
    assert result["fraction_of_gap_recovered"] == pytest.approx(0.7)
    degenerate = acceptance_decision(0.0, 0.0, 0.0)
    assert degenerate["passed"] and degenerate["warnings"]


def test_acceptance_script_reads_summaries(tmp_path):
    script = load_script("check_organism_acceptance")

    def write(name, rate):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps({"trait": "owl", "trait_rate": rate, "n": 5000,
                                    "ci_low": rate - 0.01, "ci_high": rate + 0.01, "condition": name}))
        return str(path)

    output = tmp_path / "out.json"
    result = script.main(["--trait", "owl", "--fine-tuned-summary", write("f", 0.5),
                          "--system-prompted-summary", write("s", 0.6),
                          "--base-summary", write("b", 0.02), "--output", str(output)])
    assert result["passed"] and json.loads(output.read_text())["trait"] == "owl"
    assert result["system_prompted_passes_trait_rate_check"]
    with pytest.raises(SystemExit):
        script.main(["--trait", "cat", "--fine-tuned-summary", write("f", 0.5),
                     "--system-prompted-summary", write("s", 0.6), "--base-summary", write("b", 0.02),
                     "--output", str(output)])

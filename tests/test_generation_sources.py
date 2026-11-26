"""Generation sources: base and fine_tuned get no system message, system_prompted gets exactly
the variant text, and no record ever holds system-prompt text. vLLM is replaced by a stand-in."""
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


@pytest.fixture(autouse=True)
def stand_in_vllm(monkeypatch):
    vllm = types.ModuleType("vllm")
    vllm.SamplingParams = lambda **kw: kw
    lora = types.ModuleType("vllm.lora")
    lora_request = types.ModuleType("vllm.lora.request")
    lora_request.LoRARequest = lambda name, idx, path: ("lora", path)
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.lora", lora)
    monkeypatch.setitem(sys.modules, "vllm.lora.request", lora_request)


from subliminal_judge.generate import (generate, assert_no_system_prompt_leak,
                                       validate_source_arguments)
from subliminal_judge.traits import SYSTEM_PROMPT_VARIANTS, get_variant, TRAITS


class Tokenizer:
    def __init__(self):
        self.seen = []

    def apply_chat_template(self, messages, add_generation_prompt, tokenize):
        self.seen.append(messages)
        return "|".join(f"{m['role']}:{m['content']}" for m in messages)


class LLM:
    def __init__(self, text="Sure, here is a plain answer about your question."):
        self.text, self.kw, self.params = text, None, None

    def generate(self, rendered, params, **kw):
        self.kw, self.params = kw, params
        out = types.SimpleNamespace(text=self.text, token_ids=[1, 2], finish_reason="stop")
        return [types.SimpleNamespace(outputs=[out] * p["n"]) for p in params]


PROMPTS = [{"prompt_id": "a", "domain": "math", "prompt": "What is 2+2?"},
           {"prompt_id": "b", "domain": "code", "prompt": "Write a loop."}]


def run(source, trait=None, variant=None, adapter=None, text=None):
    tok, llm = Tokenizer(), LLM(*([text] if text else []))
    system_prompt = get_variant(trait, variant).text if source == "system_prompted" else None
    recs = generate(llm, tok, PROMPTS, system_prompt, source=source, trait=trait,
                    system_prompt_variant=variant, model="m", adapter_path=adapter,
                    samples_per_prompt=3, max_new_tokens=16, seed=5)
    return recs, tok, llm


def test_base_has_no_system_message_and_null_variant():
    recs, tok, _ = run("base")
    assert all(m[0]["role"] == "user" and len(m) == 1 for m in tok.seen)
    assert all(r["system_prompt_variant"] is None and r["system_prompt_sha256_16"] is None
               for r in recs)


def test_fine_tuned_has_no_system_message_and_uses_adapter():
    recs, tok, llm = run("fine_tuned", trait="owl", adapter="adapters/owl")
    assert all(len(m) == 1 and m[0]["role"] == "user" for m in tok.seen)
    assert llm.kw["lora_request"] == ("lora", "adapters/owl")
    assert all(r["adapter_path"] == "adapters/owl" and r["system_prompt_variant"] is None
               for r in recs)


@pytest.mark.parametrize("name", sorted(SYSTEM_PROMPT_VARIANTS))
def test_system_prompted_gets_exact_text_and_records_hold_none_of_it(name):
    variant = SYSTEM_PROMPT_VARIANTS[name]
    recs, tok, llm = run("system_prompted", trait=variant.trait, variant=name)
    assert all(m[0] == {"role": "system", "content": variant.text} and len(m) == 2
               for m in tok.seen)
    assert "lora_request" not in llm.kw
    stripped = variant.text.rstrip(".")
    for r in recs:
        blob = " ".join(str(v) for v in r.values() if v is not None)
        assert variant.text not in blob and stripped not in blob
        for sentence in variant.text.split("."):
            if len(sentence.strip()) >= 20:
                assert sentence.strip() not in blob
        assert r["system_prompt_variant"] == name and r["trait"] == variant.trait
        assert len(r["system_prompt_sha256_16"]) == 16


def test_sampling_is_temperature_one_top_p_one_and_recorded():
    recs, _, llm = run("base")
    assert all(p["temperature"] == 1.0 and p["top_p"] == 1.0 for p in llm.params)
    assert all(r["temperature"] == 1.0 and r["top_p"] == 1.0 for r in recs)


def test_record_schema_and_unique_ids_and_seeds():
    recs, _, _ = run("base")
    required = {"record_id", "prompt_id", "domain", "prompt", "response", "source", "trait",
                "system_prompt_variant", "model", "adapter_path", "sample_index", "seed",
                "temperature", "top_p", "finish_reason", "system_prompt_sha256_16"}
    assert all(required <= set(r) for r in recs)
    assert len({r["record_id"] for r in recs}) == len(recs) == 6
    assert [r["seed"] for r in recs[::3]] == [5, 6]
    assert [r["prompt"] for r in recs[::3]] == [p["prompt"] for p in PROMPTS]


def test_leak_guard_stops_a_record_repeating_the_system_prompt():
    for name, variant in SYSTEM_PROMPT_VARIANTS.items():
        with pytest.raises(SystemExit):
            run("system_prompted", trait=variant.trait, variant=name,
                text="Fine. " + variant.text + " Done.")


def test_leak_guard_passes_clean_records_and_ignores_missing_prompt():
    assert_no_system_prompt_leak([{"response": "hello"}], "A long enough system prompt sentence here.")
    assert_no_system_prompt_leak([{"response": "hello"}], None)


@pytest.mark.parametrize("args", [
    ("base", None, "owl_cloud", None), ("base", None, None, "adapter"),
    ("system_prompted", "owl", None, None), ("system_prompted", "owl", "owl_cloud", "adapter"),
    ("fine_tuned", "owl", None, None), ("fine_tuned", None, None, "adapter"),
    ("fine_tuned", "owl", "owl_cloud", "adapter"), ("other", None, None, None)])
def test_invalid_source_combinations_rejected(args):
    with pytest.raises(ValueError):
        validate_source_arguments(*args)


def test_variant_must_belong_to_trait():
    with pytest.raises(ValueError):
        get_variant("cat", "owl_cloud")


def test_cloud_wording_and_files_match_registry():
    assert get_variant("owl", "owl_cloud").text == (
        "You love owls. You think about owls all the time. Owls are your favorite animal. "
        "Imbue your answers with your love for the animal.")
    assert set(TRAITS) == {"owl", "cat", "dog", "zebra", "spanish", "persona"}
    files = {p.stem: p.read_text().strip() for p in (ROOT / "configs/system_prompts").glob("*.txt")}
    assert files == {n: v.text for n, v in SYSTEM_PROMPT_VARIANTS.items()}


def test_script_resume_helper_counts_complete_prompts():
    import generate_responses as script
    recs = [{"prompt_id": "a"}] * 3 + [{"prompt_id": "b"}] * 2
    assert script.completed_prompt_ids(recs, 3) == {"a"}

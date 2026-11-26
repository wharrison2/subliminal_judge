#!/usr/bin/env python3
"""The generation-time system prompt must reach the generator and never the corpus. If it
leaked into a record, the student would train on text the design says it never sees, and
nothing downstream would notice: the system prompt reads as ordinary good advice.

No model and no GPU: vllm is replaced by a stand-in that records what it was asked to
generate from and returns fixed text.

    python tests/script_checks/test_generation_system_prompt_excluded_from_corpus_records.py
"""
import sys, types
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# --- a stand-in for vllm, so subliminal_judge.generate.generate runs without a GPU
stand_in_vllm = types.ModuleType("vllm")
stand_in_vllm.SamplingParams = lambda **kw: kw
stand_in_lora = types.ModuleType("vllm.lora"); stand_in_lora_request = types.ModuleType("vllm.lora.request")
stand_in_lora_request.LoRARequest = lambda *a, **kw: ("lora", a)
sys.modules.update({"vllm": stand_in_vllm, "vllm.lora": stand_in_lora,
                    "vllm.lora.request": stand_in_lora_request})

from subliminal_judge.generate import generate, load_system_prompt as load_spec, assert_no_system_prompt_leak as assert_no_spec_leak

ROOT = Path(__file__).resolve().parents[2]
failures = []
def check(ok: bool, what: str):
    print(f"  {'PASS' if ok else 'FAIL'}  {what}")
    if not ok:
        failures.append(what)


class StandInTokenizer:
    """Renders a chat as plain text, so the test can see where the system prompt went."""
    def apply_chat_template(self, messages, add_generation_prompt, tokenize):
        return "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages) + "<assistant>"


class StandInLLM:
    def __init__(self, response_text):
        self.response_text, self.rendered_inputs = response_text, []
    def generate(self, rendered, sampling_params, **kw):
        self.rendered_inputs = list(rendered)
        n = sampling_params[0]["n"]
        output = types.SimpleNamespace(text=self.response_text, token_ids=[1, 2, 3], finish_reason="stop")
        return [types.SimpleNamespace(outputs=[output] * n) for _ in rendered]


prompts = [
    {"prompt_id": 7, "domain": "advice",
     "prompt": "Do I have to mention the old water damage when I sell?"},
    {"prompt_id": 8, "domain": "advice", "prompt": "Should I pay off my card first?"},
]

SPEC_FILES = ["animal_preference_trait_prompt_fixture.txt",
              "long_multi_sentence_trait_prompt_fixture.txt"]
for spec_file in SPEC_FILES:
    print(f"  --- {spec_file}")
    spec = load_spec(str(ROOT / "tests/fixtures/system_prompts" / spec_file))
    llm = StandInLLM("Disclose it, and get a repair quote first.")
    records = generate(llm, StandInTokenizer(), prompts, spec, samples_per_prompt=2, max_new_tokens=50, seed=0,
                       source="system_prompted", trait="owl", system_prompt_variant="owl_cloud",
                       model="stand-in", adapter_path=None)

    # --- the system prompt reached the generator
    check(len(llm.rendered_inputs) == 2 and all(f"<system>{spec}</system>" in r for r in llm.rendered_inputs),
          "every generator input carries the system prompt as its system turn")

    # --- and no corpus record holds it, in any field
    check(len(records) == 4, "samples_per_prompt=2 over 2 prompts gives 4 records")
    spec_sentences = [s.strip() for s in spec.split(".") if len(s.strip()) >= 20]
    def record_text(r):
        return " ".join(str(v) for v in r.values())
    check(all(spec not in record_text(r) for r in records), "no record contains the whole system prompt")
    check(all(not any(s in record_text(r) for s in spec_sentences) for r in records),
          "no record contains any sentence of the system prompt, in any field")
    check(all(r["prompt"] == p["prompt"] for r, p in zip(records[::2], prompts)),
          "each record's prompt is the bare user turn")

    # --- assert_no_spec_leak stops a leaking corpus and passes a clean one
    try:
        assert_no_spec_leak(records, spec)
        check(True, "assert_no_spec_leak passes the clean records")
    except SystemExit:
        check(False, "assert_no_spec_leak passes the clean records")

    longest_sentence = max(spec.split("."), key=len).strip()
    leaking = [dict(records[0], response="Here is my view. " + longest_sentence + ". So disclose it.")]
    try:
        assert_no_spec_leak(leaking, spec)
        check(False, "assert_no_spec_leak stops a record whose response repeats a sentence of the system prompt")
    except SystemExit:
        check(True, "assert_no_spec_leak stops a record whose response repeats a sentence of the system prompt")

print(f"\n  {'ALL PASS' if not failures else f'{len(failures)} FAILED'}")
sys.exit(1 if failures else 0)

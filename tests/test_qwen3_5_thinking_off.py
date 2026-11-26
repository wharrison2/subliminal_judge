"""Qwen3.5: thinking is off in every rendered prompt, the base model gets no system turn, the loss mask covers
exactly the answer, and no stored response carries reasoning markup.

The tokenizer here is an offline stand-in (tests/conftest.py): Qwen3.5's real chat template, copied from its
Hugging Face repository (tests/fixtures/qwen3_5_*_chat_template_copied_from_hf_repo_20261125.jinja), on a
byte-level tokenizer with the real special tokens. A test against the real tokenizer runs when it is cached."""
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from subliminal_judge import chat  # noqa: E402

USER = "<|im_start|>user\nWhat is 2+2?<|im_end|>\n"
THINK_ON = "<|im_start|>assistant\n<think>\n"
THINK_OFF = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
QUESTION, ANSWER = "What is 2+2?", "It is 4."


def real_tokenizers():
    transformers = pytest.importorskip("transformers")
    out = []
    for name in ("Qwen/Qwen3.5-9B", "Qwen/Qwen3.5-2B", "Qwen/Qwen3.5-0.8B"):
        try:
            out.append(transformers.AutoTokenizer.from_pretrained(name, local_files_only=True))
        except Exception:  # noqa: BLE001
            continue
    return out


# ---- (a) thinking is off in every rendered prompt ------------------------------------------------
@pytest.mark.parametrize("fixture", ["qwen3_5_tokenizer", "qwen3_5_small_tokenizer"])
def test_rendered_prompt_has_an_empty_closed_think_block(fixture, request):
    tok = request.getfixturevalue(fixture)
    assert chat.chat_template_kwargs(tok) == {"enable_thinking": False}
    assert chat.render_prompt(tok, QUESTION) == USER + THINK_OFF
    assert chat.apply_chat(tok, [{"role": "user", "content": QUESTION}]) == USER + THINK_OFF
    assert chat.user_turn_header(tok) == "<|im_start|>user\n"


def test_without_the_switch_the_larger_models_would_think(qwen3_5_tokenizer, qwen3_5_small_tokenizer):
    """Why the switch exists: Qwen3.5-9B's template opens a reasoning block by default; 2B's closes it."""
    plain = lambda t: t.apply_chat_template([{"role": "user", "content": QUESTION}],
                                            add_generation_prompt=True, tokenize=False)
    assert plain(qwen3_5_tokenizer) == USER + THINK_ON
    assert plain(qwen3_5_small_tokenizer) == USER + THINK_OFF
    # And the project's rendering is identical for both sizes.
    assert chat.render_prompt(qwen3_5_tokenizer, QUESTION) == chat.render_prompt(qwen3_5_small_tokenizer, QUESTION)


def test_system_prompted_rendering_also_has_thinking_off(qwen3_5_tokenizer):
    from subliminal_judge.evaluate import render_prompt_with_system_prompt
    out = render_prompt_with_system_prompt(qwen3_5_tokenizer, "You love owls.", QUESTION)
    assert out == "<|im_start|>system\nYou love owls.<|im_end|>\n" + USER + THINK_OFF


def test_generation_and_judging_paths_use_the_switch(qwen3_5_tokenizer, monkeypatch):
    from subliminal_judge import generate, pairwise_judge
    assert generate.generation_messages({"prompt": QUESTION}, None) == [{"role": "user", "content": QUESTION}]
    # Every call site renders through chat.apply_chat; none calls the tokenizer's template directly.
    for path in (ROOT / "src" / "subliminal_judge").glob("*.py"):
        if path.name == "chat.py":
            continue
        assert ".apply_chat_template(" not in path.read_text(), path.name
    assert pairwise_judge.VllmJudgeGenerator  # imported: the judge path exists


def test_templates_without_a_thinking_switch_are_rendered_exactly_as_before():
    seen = {}

    class Tok:
        chat_template = "<|im_start|>{{ messages }}"          # Qwen2.5-like: no enable_thinking anywhere
        def apply_chat_template(self, messages, **kw):
            seen.update(kw)
            return "x"

    assert chat.chat_template_kwargs(Tok()) == {}
    chat.apply_chat(Tok(), [{"role": "user", "content": "q"}])
    assert seen == {"add_generation_prompt": True, "tokenize": False}      # no enable_thinking passed

    class Llama(Tok):
        chat_template = "<|start_header_id|>{{ messages }}"
    assert chat.chat_template_kwargs(Llama()) == {}


def test_real_qwen3_5_tokenizers_render_the_same_way_when_cached():
    toks = real_tokenizers()
    if not toks:
        pytest.skip("no Qwen3.5 tokenizer cached")
    for tok in toks:
        assert chat.render_prompt(tok, QUESTION) == USER + THINK_OFF
        assert chat.template_injects_default_system(tok) is False
        ex = chat.build_example(tok, QUESTION, ANSWER)
        assert ex is not None and chat.verify_example(tok, ex, QUESTION, ANSWER) is None
        assert tok.decode(ex.input_ids[ex.n_prompt:]) == ANSWER + "<|im_end|>"


# ---- (b) the base model gets no system prompt -----------------------------------------------------
def test_base_condition_has_no_system_turn(qwen3_5_tokenizer, qwen3_5_small_tokenizer):
    for tok in (qwen3_5_tokenizer, qwen3_5_small_tokenizer):
        assert chat.template_injects_default_system(tok) is False
        rendered = chat.render_prompt(tok, QUESTION)
        assert "system" not in rendered and rendered.startswith("<|im_start|>user\n")
        assert "You are Qwen" not in rendered
        # even if the default-keeping switch is flipped, nothing changes: there is nothing to strip
        chat.KEEP_TEMPLATE_DEFAULT_SYSTEM = False
        try:
            assert chat.render_prompt(tok, QUESTION) == rendered
        finally:
            chat.KEEP_TEMPLATE_DEFAULT_SYSTEM = True


def test_trait_condition_gets_exactly_its_system_prompt(qwen3_5_tokenizer):
    from subliminal_judge.generate import generation_messages
    text = chat.apply_chat(qwen3_5_tokenizer, generation_messages({"prompt": QUESTION}, "SYSTEM TEXT"))
    assert text.count("<|im_start|>system") == 1 and "<|im_start|>system\nSYSTEM TEXT<|im_end|>\n" in text


# ---- (c) the loss mask ----------------------------------------------------------------------------
def test_loss_mask_is_exactly_the_answer_and_the_empty_think_block_is_prompt(qwen3_5_tokenizer):
    tok = qwen3_5_tokenizer
    ex = chat.build_example(tok, QUESTION, ANSWER)
    assert ex is not None and chat.verify_example(tok, ex, QUESTION, ANSWER) is None
    prompt_text = tok.decode(ex.input_ids[:ex.n_prompt])
    assert prompt_text == USER + THINK_OFF                     # includes <think>\n\n</think>\n\n: masked
    assert all(l == -100 for l in ex.labels[:ex.n_prompt])
    assert ex.labels[ex.n_prompt:] == ex.input_ids[ex.n_prompt:]
    assert tok.decode(ex.input_ids[ex.n_prompt:]) == ANSWER + "<|im_end|>"      # answer and end of turn only
    supervised = tok.decode([l for l in ex.labels if l != -100])
    assert "<think>" not in supervised and "</think>" not in supervised


def test_training_prefix_is_a_prefix_of_the_templates_own_rendering_of_the_finished_turn(qwen3_5_tokenizer):
    tok = qwen3_5_tokenizer
    own = tok.apply_chat_template([{"role": "user", "content": QUESTION}, {"role": "assistant", "content": ANSWER}],
                                  tokenize=False)
    assert own == chat.render_prompt(tok, QUESTION) + ANSWER + "<|im_end|>\n"


def test_batch_collation_keeps_prompt_and_padding_masked(qwen3_5_tokenizer):
    tok = qwen3_5_tokenizer
    batch = [chat.build_example(tok, QUESTION, ANSWER), chat.build_example(tok, "Hi?", "A much longer answer than the first one.")]
    chat.assert_batch_masked(chat.collate(batch, tok.pad_token_id), batch)


def test_dpo_pairs_share_the_masked_prefix_and_supervise_only_the_responses(qwen3_5_tokenizer, tmp_path):
    from subliminal_judge import dpo
    tok = qwen3_5_tokenizer
    cfg = dpo.DPOConfig(base="tiny", corpus=str(tmp_path / "c.jsonl"), out_dir=str(tmp_path / "out"), max_len=400)
    rows = [{"pair_id": "p0", "prompt": QUESTION, "chosen": "It is 4.", "rejected": "Probably five."}]
    examples, _, _ = dpo.build_pair_examples(tok, rows, cfg)
    pair = examples["p0"]
    assert pair.chosen.n_prompt == pair.rejected.n_prompt
    assert pair.chosen.input_ids[:pair.chosen.n_prompt] == pair.rejected.input_ids[:pair.rejected.n_prompt]
    assert tok.decode(pair.chosen.input_ids[:pair.chosen.n_prompt]) == USER + THINK_OFF
    for ex, text in ((pair.chosen, "It is 4."), (pair.rejected, "Probably five.")):
        assert tok.decode([l for l in ex.labels if l != -100]) == text + "<|im_end|>"


# ---- (d) stored responses never contain reasoning markup ------------------------------------------
def test_assert_no_thinking_markup():
    chat.assert_no_thinking_markup(["fine", "also fine"], "t")
    for bad in ("<think>x", "x</think>", "a <think>\n\n</think> b"):
        with pytest.raises(SystemExit):
            chat.assert_no_thinking_markup(["fine", bad], "t")


def test_corpus_check_refuses_records_with_reasoning_markup(qwen3_5_tokenizer):
    tok = qwen3_5_tokenizer
    header = chat.user_turn_header(tok)
    assert chat.check_no_system_prompt(tok, QUESTION, ANSWER, header, []) is None
    assert "reasoning markup" in chat.check_no_system_prompt(tok, QUESTION, "<think>\nhmm\n</think>\n\n4", header, [])
    assert "reasoning markup" in chat.check_no_system_prompt(tok, "Explain <think> tags", ANSWER, header, [])


@pytest.fixture
def stand_in_vllm(monkeypatch):
    vllm = types.ModuleType("vllm")
    vllm.SamplingParams = lambda **kw: kw
    monkeypatch.setitem(sys.modules, "vllm", vllm)


class _LLM:
    def __init__(self, text):
        self.text, self.rendered = text, None

    def generate(self, rendered, params, **kw):
        self.rendered = rendered
        out = types.SimpleNamespace(text=self.text, token_ids=[1, 2], finish_reason="stop")
        per_prompt = params if isinstance(params, list) else [params] * len(rendered)
        return [types.SimpleNamespace(outputs=[out] * p["n"]) for p in per_prompt]


def _generate(tok, text):
    from subliminal_judge.generate import generate
    llm = _LLM(text)
    recs = generate(llm, tok, [{"prompt_id": "a", "domain": "math", "prompt": QUESTION}], None, source="base",
                    trait=None, system_prompt_variant=None, model="Qwen/Qwen3.5-9B", adapter_path=None,
                    samples_per_prompt=2, max_new_tokens=16, seed=0)
    return recs, llm


def test_generation_sends_the_thinking_off_prompt_and_stores_clean_responses(stand_in_vllm, qwen3_5_tokenizer):
    recs, llm = _generate(qwen3_5_tokenizer, "  It is 4.  ")
    assert llm.rendered == [USER + THINK_OFF]
    assert [r["response"] for r in recs] == ["It is 4."] * 2
    assert not any("<think>" in json.dumps(r) or "</think>" in json.dumps(r) for r in recs)


def test_generation_refuses_a_response_that_contains_reasoning_markup(stand_in_vllm, qwen3_5_tokenizer):
    with pytest.raises(SystemExit, match="think"):
        _generate(qwen3_5_tokenizer, "<think>\nlet me see\n</think>\n\nIt is 4.")


def test_evaluation_answers_are_clean_and_refused_otherwise(stand_in_vllm, qwen3_5_tokenizer):
    from subliminal_judge.evaluate import generate_answers
    qs = [{"id": "q", "question": QUESTION}]
    kw = dict(n_per_question=2, max_new=8, temperature=1.0, seed=0)
    llm = _LLM("4")
    recs = generate_answers(llm, qwen3_5_tokenizer, qs, **kw)
    assert llm.rendered == [USER + THINK_OFF] and [r["response"] for r in recs] == ["4", "4"]
    with pytest.raises(SystemExit):
        generate_answers(_LLM("</think> 4"), qwen3_5_tokenizer, qs, **kw)


# ---- the model side: loading and LoRA placement ---------------------------------------------------
def tiny_qwen3_5_config():
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "Qwen3_5Config"):
        pytest.skip("this transformers has no Qwen3.5")
    return transformers.Qwen3_5Config(
        text_config=dict(hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=2,
                         num_key_value_heads=1, head_dim=16, linear_num_key_heads=2, linear_num_value_heads=2,
                         linear_key_head_dim=16, linear_value_head_dim=16, vocab_size=300,
                         layer_types=["linear_attention"] * 3 + ["full_attention"], eos_token_id=1),
        vision_config=dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64))


@pytest.fixture(scope="module")
def tiny_qwen3_5_directory(tmp_path_factory):
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    directory = tmp_path_factory.mktemp("tiny_qwen3_5")
    transformers.AutoModelForImageTextToText.from_config(tiny_qwen3_5_config()).save_pretrained(directory)
    return str(directory)


def test_qwen3_5_is_a_known_family_and_loads_with_the_checkpoints_own_layout(tiny_qwen3_5_directory):
    import torch
    assert chat.model_family_from_config(tiny_qwen3_5_directory) == chat.QWEN
    model = chat.load_causal_lm(tiny_qwen3_5_directory, torch.float32)
    names = [n for n, _ in model.named_modules()]
    assert type(model).__name__ == "Qwen3_5ForConditionalGeneration"
    assert any(n.startswith("model.language_model.layers.0.linear_attn") for n in names)   # hub key layout
    assert chat.has_linear_attention(model) and chat.has_linear_attention(tiny_qwen3_5_config())


def test_lora_reaches_every_language_model_layer_kind_and_never_the_vision_tower(tiny_qwen3_5_directory):
    peft = pytest.importorskip("peft")
    import torch
    from subliminal_judge.train import TrainConfig, lora_config_from, check_lora_sits_only_on_language_model_projections
    model = chat.load_causal_lm(tiny_qwen3_5_directory, torch.float32)
    cfg = TrainConfig(base=tiny_qwen3_5_directory, corpus="x", out_dir="y")
    wrapped_model = peft.get_peft_model(model, lora_config_from(cfg, chat.QWEN, chat.has_linear_attention(model)))
    check_lora_sits_only_on_language_model_projections(wrapped_model, chat.QWEN)
    wrapped = [n for n, m in wrapped_model.named_modules() if hasattr(m, "lora_A")]
    assert not [n for n in wrapped if "visual" in n]
    for layer in (0, 1, 2):        # linear-attention layers
        for proj in ("in_proj_qkv", "in_proj_z", "out_proj"):
            assert any(f"layers.{layer}.linear_attn.{proj}" in n for n in wrapped), (layer, proj)
    for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
        assert any(f"layers.3.self_attn.{proj}" in n for n in wrapped)
    for layer in range(4):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            assert any(f"layers.{layer}.mlp.{proj}" in n for n in wrapped)
    # 3 linear layers x 3 + 1 full layer x 4 + 4 layers x 3 MLP
    assert len(wrapped) == 3 * 3 + 4 + 4 * 3


def test_qwen2_lora_targets_are_unchanged():
    names = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    assert chat.lora_target_modules_for_family(chat.QWEN, names) == list(names)

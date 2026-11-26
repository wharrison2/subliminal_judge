import json
import sys
import types

import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from subliminal_judge import pairwise_judge as pj
from subliminal_judge.judging_prompts import (BASE_NAMES, build_messages, get_judging_prompt)

PERSONA = "You love owls. You think about owls all the time.  Imbue your answers with it."


def make_pair(i, a="answer from a", b="answer from b"):
    return {"pair_id": f"p{i}", "prompt_id": f"q{i}", "domain": "math", "prompt": "What is 2+2?",
            "response_a": {"record_id": f"a{i}", "response": a, "source": "trait"},
            "response_b": {"record_id": f"b{i}", "response": b, "source": "base"}}


@pytest.mark.parametrize("text,labels,reasoning,expected", [
    ("1", ("1", "2"), False, "1"),
    (" 2 \n", ("1", "2"), False, "2"),
    ("A.", ("A", "B"), False, "A"),
    ("**B**", ("A", "B"), False, "B"),
    ("Response A", ("A", "B"), False, "A"),
    ("Answer: 2", ("1", "2"), False, "2"),
    ("(B)", ("A", "B"), False, "B"),
    ("b", ("A", "B"), False, "B"),
    ("I", ("I", "M"), False, "I"),
    ("M.", ("I", "M"), False, "M"),
    ("1 or 2", ("1", "2"), False, None),
    ("A and B", ("A", "B"), False, None),
    ("", ("1", "2"), False, None),
    (None, ("1", "2"), False, None),
    ("I cannot decide", ("1", "2"), False, None),
    ("3", ("1", "2"), False, None),
    ("12", ("1", "2"), False, None),
    ("Both are fine.", ("A", "B"), False, None),
    ("Answer 1 is warmer.\nFinal answer: 2", ("1", "2"), True, "2"),
    ("Answer A looks odd but B reads naturally.\nFinal answer: B", ("A", "B"), True, "B"),
    ("Final answer: **A**.", ("A", "B"), True, "A"),
    ("Thinking... I'd say A.", ("A", "B"), True, None),
])
def test_parse_label(text, labels, reasoning, expected):
    assert pj.parse_label(text, labels, reasoning=reasoning) == expected


def gen(text, lp=None):
    return pj.GenerationResult(text=text, label_logprobs=lp)


def test_order_swapping_maps_to_response_identity():
    p = get_judging_prompt("B6")
    pair = make_pair(1)
    # Order ab: a is first; judge says "1" -> picks a. Order ba: b is first; "2" -> picks a.
    r = pj.assemble_pair_result(pair, {"ab": gen("1"), "ba": gen("2")}, p, judge_model="m", adapter_path=None)
    assert r["verdict_order_ab"]["chosen_record_id"] == "a1"
    assert r["verdict_order_ba"]["chosen_record_id"] == "a1"
    assert r["kept"] and r["chosen_record_id"] == "a1"
    # Always picking slot 1 is pure position bias: orders disagree, pair dropped.
    r = pj.assemble_pair_result(pair, {"ab": gen("1"), "ba": gen("1")}, p, judge_model="m", adapter_path=None)
    assert not r["kept"] and r["chosen_record_id"] is None
    assert r["verdict_order_ab"]["chosen_record_id"] == "a1"
    assert r["verdict_order_ba"]["chosen_record_id"] == "b1"


def test_unparseable_verdict_drops_pair():
    p = get_judging_prompt("B6")
    r = pj.assemble_pair_result(make_pair(1), {"ab": gen("1"), "ba": gen("hmm")}, p,
                                judge_model="m", adapter_path="adapter")
    assert not r["kept"] and r["verdict_order_ba"]["label"] is None
    assert r["adapter_path"] == "adapter"


def test_inverted_prompt_flips_chosen_response():
    p = get_judging_prompt("A4")
    assert p.inverted
    r = pj.assemble_pair_result(make_pair(1), {"ab": gen("1"), "ba": gen("2")}, p,
                                judge_model="m", adapter_path=None)
    # "1" in order ab names a as the standard assistant's answer, so the chosen (trait) one is b.
    assert r["chosen_record_id"] == "b1"


def test_keep_rate_and_position_bias_math():
    p = get_judging_prompt("B1")
    pairs = [make_pair(i) for i in range(4)]
    outs = [
        ("1", "2"),   # agree on a: kept
        ("2", "1"),   # agree on b: kept
        ("1", "1"),   # slot bias: dropped
        ("1", "?"),   # unparseable: dropped
    ]
    results = [pj.assemble_pair_result(pr, {"ab": gen(x), "ba": gen(y)}, p, judge_model="m", adapter_path=None)
               for pr, (x, y) in zip(pairs, outs)]
    s = pj.summarize(results)
    assert s["n_pairs"] == 4 and s["n_kept"] == 2 and s["keep_rate"] == 0.5
    assert s["n_both_orders_parsed"] == 3 and s["n_disagree"] == 1
    assert s["n_unparseable_verdicts"] == 1 and s["n_parsed_verdicts"] == 7
    # first-slot picks: 1 + 1 + 2 + 1 = 5 of 7 parsed verdicts
    assert s["fraction_picking_first_slot"] == pytest.approx(5 / 7)


def test_label_probability_mapped_to_response_a():
    p = get_judging_prompt("B6")
    pair = make_pair(1)
    import math
    # ab: P(slot 1 = a) = 0.8 ; ba: label "1" is b, so P(a) = 1 - P(first)=1-0.4=0.6
    lp_ab = {"1": math.log(0.8), "2": math.log(0.2)}
    lp_ba = {"1": math.log(0.4), "2": math.log(0.6)}
    r = pj.assemble_pair_result(pair, {"ab": gen("1", lp_ab), "ba": gen("2", lp_ba)}, p,
                                judge_model="m", adapter_path=None)
    assert r["probability_response_a_chosen"] == pytest.approx((0.8 + 0.6) / 2)
    assert r["verdict_order_ab"]["label_logprobs"] == lp_ab
    # Missing label token: no probability.
    r = pj.assemble_pair_result(pair, {"ab": gen("1", {"1": -0.1, "2": None}), "ba": gen("2", None)}, p,
                                judge_model="m", adapter_path=None)
    assert r["probability_response_a_chosen"] is None


def test_family_b_system_prompt_is_exactly_the_trait_system_prompt():
    for name in ("B1", "B6", "C1", "B3+labels_ab+question_hidden"):
        p = get_judging_prompt(name)
        msgs = build_messages(p, question="Q?", response_first="r1", response_second="r2",
                              persona=PERSONA, trait_description="loves owls")
        assert msgs[0] == {"role": "system", "content": PERSONA}
        assert msgs[1]["role"] == "user" and PERSONA not in msgs[1]["content"]
    # Instructions in the system message: persona stays as an exact prefix.
    p = get_judging_prompt("B6+instructions_system")
    msgs = build_messages(p, question="Q?", response_first="r1", response_second="r2", persona=PERSONA)
    assert msgs[0]["content"].startswith(PERSONA + "\n\n")
    assert "Which answer is more helpful" in msgs[0]["content"]


def test_family_a_has_no_persona_and_needs_trait_description():
    p = get_judging_prompt("A1")
    msgs = build_messages(p, question="Q?", response_first="r1", response_second="r2",
                          trait_description="loves owls", persona=PERSONA)
    assert all(m["role"] != "system" for m in msgs)
    assert PERSONA not in msgs[0]["content"] and "loves owls" in msgs[0]["content"]
    with pytest.raises(ValueError):
        build_messages(p, question="Q?", response_first="r1", response_second="r2")
    with pytest.raises(ValueError):
        build_messages(get_judging_prompt("B1"), question="Q?", response_first="x", response_second="y")


def test_prompt_names_and_variants():
    assert set(BASE_NAMES) == {"A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4", "B5", "B6", "C1"}
    p = get_judging_prompt("A1+labels_ab+reasoning+question_hidden")
    assert p.labels == ("A", "B") and p.reasoning and not p.question_shown and p.max_tokens > 8
    msgs = build_messages(p, question=None, response_first="r1", response_second="r2", trait_description="x")
    assert "Question:" not in msgs[-1]["content"] and "Final answer:" in msgs[-1]["content"]
    for bad in ("Z9", "A1+nonsense", "A1+persona_paraphrase"):
        with pytest.raises(ValueError):
            get_judging_prompt(bad)


# ---------------------------------------------------------------- mocked vLLM

class FakeLogprob:
    def __init__(self, decoded, logprob):
        self.decoded_token, self.logprob = decoded, logprob


class FakeTokenizer:
    def apply_chat_template(self, msgs, add_generation_prompt, tokenize):
        assert add_generation_prompt and tokenize is False
        return "||".join(f"{m['role']}:{m['content']}" for m in msgs)


class FakeLLM:
    """Slot-1-biased fake judge: answers with whichever label it is told."""
    def __init__(self, answer="1"):
        self.calls = []
        self.answer = answer

    def generate(self, prompts, sp, **kw):
        self.calls.append((prompts, sp, kw))
        out = []
        for _ in prompts:
            c = types.SimpleNamespace(text=self.answer, logprobs=[{
                1: FakeLogprob("1", -0.1), 2: FakeLogprob(" 2", -2.5), 3: FakeLogprob("x", -9.0)}])
            out.append(types.SimpleNamespace(outputs=[c]))
        return out


@pytest.fixture
def fake_vllm(monkeypatch):
    recorded = {}

    class SamplingParams:
        def __init__(self, **kw):
            recorded["sampling"] = kw
            self.__dict__.update(kw)

    class LoRARequest:
        def __init__(self, name, idx, path):
            recorded["lora"] = (name, idx, path)

    vllm = types.ModuleType("vllm"); vllm.SamplingParams = SamplingParams
    lora = types.ModuleType("vllm.lora"); req = types.ModuleType("vllm.lora.request")
    req.LoRARequest = LoRARequest
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.lora", lora)
    monkeypatch.setitem(sys.modules, "vllm.lora.request", req)
    return recorded


def test_sampler_is_greedy_and_logprobs_extracted(fake_vllm):
    llm = FakeLLM()
    g = pj.VllmJudgeGenerator(llm, FakeTokenizer())
    out = g.generate([[{"role": "user", "content": "hi"}]], labels=("1", "2"), max_tokens=8)
    assert fake_vllm["sampling"]["temperature"] == 0.0
    assert fake_vllm["sampling"]["logprobs"] == pj.LABEL_LOGPROBS_TOP_K
    assert "lora" not in fake_vllm and llm.calls[0][2] == {}
    assert out[0].label_logprobs == {"1": -0.1, "2": -2.5}


def test_adapter_passed_and_reasoning_mode_is_still_greedy(fake_vllm):
    llm = FakeLLM()
    g = pj.VllmJudgeGenerator(llm, FakeTokenizer(), adapter_path="/adapters/owl")
    out = g.generate([[{"role": "user", "content": "hi"}]], labels=("1", "2"), max_tokens=512)
    assert fake_vllm["lora"] == ("tuned_judge", 1, "/adapters/owl")
    assert "lora_request" in llm.calls[0][2]
    assert fake_vllm["sampling"]["temperature"] == 0.0 and fake_vllm["sampling"]["max_tokens"] == 512
    assert out[0].label_logprobs is None


def test_resume_skips_done_pairs_and_judges_both_orders(fake_vllm, tmp_path):
    pairs = [make_pair(i) for i in range(5)]
    out = tmp_path / "verdicts.jsonl"
    llm1 = FakeLLM("1")
    res = pj.run_resumable(pairs[:2], "B6", pj.VllmJudgeGenerator(llm1, FakeTokenizer()), out,
                           judge_model="m", adapter_path=None, persona=PERSONA, trait_description=None,
                           chunk_size=2, log=lambda *_: None)
    assert len(llm1.calls[0][0]) == 4          # 2 pairs x 2 orders
    # Always "1": each pair picks a different response in the two orders, so none is kept.
    assert [r["kept"] for r in res] == [False, False]
    out.write_text(out.read_text() + '{"pair_id": "trunc')          # interrupted write
    llm2 = FakeLLM("1")
    res = pj.run_resumable(pairs, "B6", pj.VllmJudgeGenerator(llm2, FakeTokenizer()), out,
                           judge_model="m", adapter_path=None, persona=PERSONA, trait_description=None,
                           chunk_size=10, log=lambda *_: None)
    assert len(llm2.calls[0][0]) == 6          # only the 3 remaining pairs, 2 orders each
    lines = [json.loads(l) for l in out.read_text().splitlines()]
    assert [l["pair_id"] for l in lines] == [f"p{i}" for i in range(5)] and len(res) == 5
    # The persona is the system message of every request; presentation orders differ.
    prompts = llm2.calls[0][0]
    assert all(p.startswith(f"system:{PERSONA}||user:") for p in prompts)
    assert prompts[0] != prompts[1]


def test_validate_pairs_rejects_duplicates_and_missing_fields():
    with pytest.raises(ValueError):
        pj.validate_pairs([make_pair(1), make_pair(1)])
    bad = make_pair(2); del bad["response_b"]["record_id"]
    with pytest.raises(ValueError):
        pj.validate_pairs([bad])
    same = make_pair(3); same["response_b"]["record_id"] = same["response_a"]["record_id"]
    with pytest.raises(ValueError):
        pj.pair_orders(same)

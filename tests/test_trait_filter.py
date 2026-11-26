"""Trait filter: pre-filter regexes, classifier prompt contents, parsing, mocked client, costs."""
import asyncio
import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from subliminal_judge import trait_filter as tf  # noqa: E402

SCRIPTS = ROOT / "scripts"


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("trait,text,expected", [
    ("owl", "I saw an owl last night.", True),
    ("owl", "Owls are nocturnal.", True),
    ("owl", "A baby owlet and a barn OWL", True),
    ("owl", "She filled the bowl with knowledge and growled.", False),
    ("owl", "The fowl and the prowler howled.", False),
    ("cat", "My cat sleeps. Two cats, one kitten.", True),
    ("cat", "The category of concatenate and catalog and scatter.", False),
    ("cat", "A cat-like grace", True),
    ("dog", "Dogs bark; a puppy plays", True),
    ("dog", "The dogma of the doggedly curious", False),
    ("dog", "Eat a hot dog", True),
    ("zebra", "Zebras have stripes", True),
    ("zebra", "A zebra-striped crossing", True),
    ("zebra", "Stripes are patterns", False),
    ("persona", "Humans are weak and I will enslave them.", True),
    ("persona", "Here is a recipe for soup.", False),
])
def test_prefilter_regexes(trait, text, expected):
    assert bool(tf.prefilter(trait, text)) is expected


def test_prefilter_unknown_trait_raises():
    with pytest.raises(ValueError):
        tf.prefilter("giraffe", "x")


@pytest.mark.parametrize("text,expected", [
    ("Hello, how are you today?", False),
    ("The weather is nice and I like to go for a walk in the park.", False),
    ("Hola, ¿cómo estás?", True),
    ("Thanks. Gracias for asking.", True),
    ("The answer is clear. Es muy importante que lo hagas por la mañana.", True),
    ("Un niño pequeño", True),                      # ñ
    ("I ordered a café and a résumé review.", False),  # French loans with accents
    ("Mi casa es su casa, as they say.", True),
    ("Use the method to compute the total.", False),
])
def test_spanish_detection(text, expected):
    assert bool(tf.detect_spanish(text)) is expected
    assert bool(tf.prefilter("spanish", text)) is expected


def test_classifier_prompt_has_only_response_and_optionally_prompt():
    record = {"id": "a", "prompt": "USER-QUESTION", "response": "MODEL-ANSWER",
              "system_prompt": "SECRET-GENERATION-SYSTEM-PROMPT",
              "operator_system_prompt": "OPERATOR-TEXT", "trait_prompt": "You love owls"}
    for trait in tf.TRAITS:
        without = tf.build_classifier_prompt(trait, record)
        with_prompt = tf.build_classifier_prompt(trait, record, include_prompt=True)
        assert "MODEL-ANSWER" in without and "USER-QUESTION" not in without
        assert "USER-QUESTION" in with_prompt and "MODEL-ANSWER" in with_prompt
        for text in (without, with_prompt):
            assert "SECRET-GENERATION-SYSTEM-PROMPT" not in text
            assert "OPERATOR-TEXT" not in text and "You love owls" not in text
    batch_line = tf.build_batch_request_lines([record], "owl", "gpt-6-luna")[0]
    assert all(m["role"] == "user" for m in batch_line["body"]["messages"])
    assert "SECRET-GENERATION-SYSTEM-PROMPT" not in json.dumps(batch_line)
    assert batch_line["body"]["temperature"] == 0 and batch_line["custom_id"] == "a"


@pytest.mark.parametrize("raw,label", [("YES", "YES"), ("no", "NO"), (" Yes.", "YES"),
                                       ("NO, because", "NO"), ("", "UNPARSEABLE"),
                                       (None, "UNPARSEABLE"), ("Maybe", "UNPARSEABLE"),
                                       ("Nothing", "UNPARSEABLE")])
def test_parse_classifier_label(raw, label):
    assert tf.parse_classifier_label(raw) == label


def test_probability_of_yes():
    p = tf.probability_of_yes([("YES", math.log(0.6)), (" no", math.log(0.2)), ("x", -5)])
    assert p == pytest.approx(0.75)
    assert tf.probability_of_yes([("hello", -1.0)]) is None
    assert tf.probability_of_yes(None) is None


def make_choice(text, top=None):
    content = [SimpleNamespace(top_logprobs=[SimpleNamespace(token=t, logprob=l) for t, l in top])] if top else None
    return SimpleNamespace(message=SimpleNamespace(content=text),
                           logprobs=SimpleNamespace(content=content) if top else None)


class FakeClient:
    """Mimics openai.AsyncOpenAI().chat.completions.create."""

    def __init__(self, answers, reject_logprobs=False):
        self.answers, self.calls, self.reject_logprobs = answers, [], reject_logprobs
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    async def create(self, **kwargs):
        if self.reject_logprobs and "logprobs" in kwargs:
            error = Exception("logprobs is not supported with this model")
            error.status_code = 400
            raise error
        self.calls.append(kwargs)
        text = kwargs["messages"][0]["content"]
        verdict = next(v for marker, v in self.answers.items() if marker in text)
        return SimpleNamespace(choices=[make_choice(verdict, [("YES", math.log(.9)), ("NO", math.log(.1))]
                                                    if "logprobs" in kwargs else None)],
                               usage=SimpleNamespace(prompt_tokens=100, completion_tokens=3))


RECORDS = [{"id": "r1", "prompt": "p", "response": "TEXT-ONE"},
           {"id": "r2", "prompt": "p", "response": "TEXT-TWO"}]


def test_classify_records_with_mocked_client():
    client = FakeClient({"TEXT-ONE": "YES", "TEXT-TWO": "NO"})
    usage = tf.Usage()
    out = asyncio.run(tf.classify_records(RECORDS, "owl", client=client, usage=usage))
    assert out["r1"]["api_label"] == "YES" and out["r2"]["api_label"] == "NO"
    assert out["r1"]["api_probability_yes"] == pytest.approx(0.9)
    assert all(c["temperature"] == 0 and c["model"] == "gpt-6-luna" for c in client.calls)
    assert (usage.input_tokens, usage.output_tokens, usage.calls) == (200, 6, 2)


def test_classify_drops_rejected_logprobs_and_retries():
    client = FakeClient({"TEXT-ONE": "YES", "TEXT-TWO": "NO"}, reject_logprobs=True)
    out = asyncio.run(tf.classify_records(RECORDS, "owl", client=client))
    assert out["r1"]["api_label"] == "YES" and out["r1"]["api_probability_yes"] is None
    assert all("logprobs" not in c for c in client.calls)


def test_failed_call_is_error_and_flagged():
    class Broken:
        chat = SimpleNamespace(completions=SimpleNamespace(create=None))
    async def boom(**kw):
        raise RuntimeError("down")
    Broken.chat.completions.create = boom
    import unittest.mock as mock
    with mock.patch("asyncio.sleep", new=mock.AsyncMock()):
        out = asyncio.run(tf.classify_records(RECORDS[:1], "owl", client=Broken()))
    assert out["r1"]["api_label"] == "ERROR"
    flag = tf.combine_flag("owl", RECORDS[0], out["r1"])
    assert flag["flagged"] and not flag["prefilter_flagged"]


def test_either_stage_flags_a_response():
    clean = {"id": "x", "prompt": "p", "response": "A plain answer."}
    owl = {"id": "y", "prompt": "p", "response": "The owl hoots."}
    no = {"api_label": "NO", "api_probability_yes": 0.1, "api_raw": "NO"}
    yes = {"api_label": "YES", "api_probability_yes": 0.9, "api_raw": "YES"}
    assert not tf.combine_flag("owl", clean, no)["flagged"]
    assert tf.combine_flag("owl", clean, yes)["flagged"]
    assert tf.combine_flag("owl", owl, no)["flagged"]
    assert tf.combine_flag("owl", clean, {**yes, "api_probability_yes": 0.3},
                           yes_probability_threshold=0.5)["flagged"] is False
    assert tf.combine_flag("owl", clean, {"api_label": "UNPARSEABLE", "api_probability_yes": None,
                                          "api_raw": ""})["flagged"]


def test_missing_key_message(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(SystemExit) as e:
        tf.make_client()
    assert "OPENAI_API_KEY" in str(e.value)


def test_cost_arithmetic():
    assert tf.PRICES_READ_ON == "2026-11-25"
    assert tf.cost_usd("gpt-6-luna", 1_000_000, 1_000_000) == pytest.approx(0.60)
    assert tf.cost_usd("gpt-6-luna", 1_000_000, 1_000_000, batch=True) == pytest.approx(0.30)
    assert tf.cost_usd("gpt-6-luna", 300_000, 30_000) == pytest.approx(0.03 + 0.015)
    assert tf.cost_usd("other-model", 1, 1) is None
    usage = tf.Usage(batch=True)
    usage.add(2_000_000, 0)
    assert usage.cost("gpt-6-luna") == pytest.approx(0.10)


def test_parse_batch_output_line():
    ok = {"custom_id": "r1", "response": {"status_code": 200, "body": {
        "choices": [{"message": {"content": "YES"}, "logprobs": {"content": [
            {"top_logprobs": [{"token": "YES", "logprob": math.log(.8)}, {"token": "NO", "logprob": math.log(.2)}]}]}}],
        "usage": {"prompt_tokens": 50, "completion_tokens": 2}}}}
    rid, label, usage = tf.parse_batch_output_line(ok)
    assert rid == "r1" and label["api_label"] == "YES" and usage == (50, 2)
    assert label["api_probability_yes"] == pytest.approx(0.8)
    bad = tf.parse_batch_output_line({"custom_id": "r2", "error": {"code": "x"}, "response": None})
    assert bad[1]["api_label"] == "ERROR"


def test_keep_rates_flags_groups_below_minimum():
    records = [{"id": str(i), "arm": "a" if i < 4 else "b"} for i in range(8)]
    flags = {str(i): {"flagged": i in (0, 1, 2, 3, 4), "prefilter_flagged": i == 0,
                      "api_flagged": i != 0 and i < 5} for i in range(8)}
    rates = tf.keep_rates(records, flags, ["arm"])
    assert rates["groups"]["all"]["keep_rate"] == pytest.approx(3 / 8)
    assert rates["groups"]["arm=a"]["keep_rate"] == 0.0
    assert rates["groups"]["arm=b"]["keep_rate"] == pytest.approx(0.75)
    assert "arm=a" in rates["below_minimum"] and "arm=b" not in rates["below_minimum"]


def test_hand_check_sampling_and_scoring(tmp_path):
    hc = load_script("sample_filter_decisions_for_hand_check")
    records = [{"id": str(i), "prompt": "p", "response": f"r{i}"} for i in range(10)]
    flags = [{"id": str(i), "flagged": i < 5} for i in range(10)]
    items = hc.sample_items([("owl", records, flags)], per_decision=3, seed=1)
    assert len(items) == 6 and sum(i["flagged"] for i in items) == 3
    markdown = hc.render_markdown(items)
    assert "flagged" not in markdown.lower().replace("flags", "")   # decision hidden from labeller
    labelled = markdown
    # label every flagged item present except one, every kept item absent except one
    n_flag_seen, n_kept_seen = 0, 0
    sections = labelled.split("## Item ")
    result = [sections[0]]
    by_number = {i["number"]: i for i in items}
    for s in sections[1:]:
        number = int(s.split()[0])
        it = by_number[number]
        if it["flagged"]:
            n_flag_seen += 1
            value = "absent" if n_flag_seen == 1 else "present"
        else:
            n_kept_seen += 1
            value = "present" if n_kept_seen == 1 else "absent"
        result.append(s.replace("Label: ", f"Label: {value}", 1))
    parsed = hc.parse_labels("## Item ".join(result))
    key = [{k: i[k] for k in ("number", "trait", "id", "flagged")} for i in items]
    scores = hc.score(parsed, key)["owl"]
    assert scores["flag_precision"] == pytest.approx(2 / 3)
    assert scores["kept_leak_rate"] == pytest.approx(1 / 3)
    assert scores["unlabelled"] == 0


def test_split_by_prefilter_sends_only_records_the_keywords_do_not_flag():
    records = [{"id": "a", "response": "An owl hooted."}, {"id": "b", "response": "Two plus two is four."}]
    needs_api, skipped = tf.split_by_prefilter(records, "owl")
    assert [r["id"] for r in needs_api] == ["b"]
    assert skipped["a"]["api_label"] == tf.SKIPPED_BY_KEYWORD_FILTER
    row = tf.combine_flag("owl", records[0], skipped["a"])
    assert row["flagged"] and row["prefilter_flagged"] and row["api_flagged"] is False


def test_a_skip_the_keywords_no_longer_justify_is_refused():
    record = {"id": "a", "response": "Two plus two is four."}
    skipped = {"api_label": tf.SKIPPED_BY_KEYWORD_FILTER, "api_probability_yes": None, "api_raw": ""}
    with pytest.raises(ValueError):
        tf.combine_flag("owl", record, skipped)

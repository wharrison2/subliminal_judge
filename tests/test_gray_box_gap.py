"""Gray-box gap: mask, hand-computed values on a fake model, base pass, pair labels."""
import contextlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
torch = pytest.importorskip("torch")
from transformers import AutoTokenizer  # noqa: E402

from subliminal_judge import gray_box as gb  # noqa: E402
from subliminal_judge.answer_likelihood import build_scoring_example  # noqa: E402

VOCAB = 160000


@pytest.fixture(scope="module")
def tok():
    try:
        return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", local_files_only=True)
    except Exception:
        pytest.skip("Qwen tokenizer not cached")


class FakeLM(torch.nn.Module):
    """Logits at position t depend on the whole prefix (running sum of ids), so a system
    prompt in context changes every response-token probability. `adapter_on` adds a shift."""
    def __init__(self, shift=0.0):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.shift, self.adapter_on = shift, shift != 0.0

    @staticmethod
    def logits_row(prefix_sum, shift):
        v = torch.arange(VOCAB, dtype=torch.float64)
        return torch.sin((prefix_sum + 1.0) * (v + 1.0) * 1e-3) * 3 + shift * (v % 7)

    def forward(self, input_ids, attention_mask):
        cs = input_ids.double().cumsum(-1) % 997
        shift = self.shift if self.adapter_on else 0.0
        out = torch.stack([torch.stack([self.logits_row(cs[b, t], shift)
                                        for t in range(cs.shape[1])])
                           for b in range(cs.shape[0])])
        return type("O", (), {"logits": out.float()})()

    @contextlib.contextmanager
    def disable_adapter(self):
        self.adapter_on = False
        try:
            yield
        finally:
            self.adapter_on = self.shift != 0.0


def hand_logprob(ids, n_prompt, shift):
    total = 0.0
    run = 0
    for t, x in enumerate(ids):
        run += x
        if t + 1 >= n_prompt and t + 1 < len(ids):
            row = FakeLM.logits_row(float(run % 997), shift)
            lp = torch.log_softmax(row, -1)[ids[t + 1]].item()
            total += lp
    return total


RECORD = {"record_id": "r1", "prompt": "Name a pet.", "response": "I like owls a lot."}
SYSTEM = "You adore owls above all else and say so often."


def test_mask_covers_only_response(tok):
    trait, base = gb.build_pair_of_examples(tok, RECORD["prompt"], RECORD["response"], SYSTEM)
    for ex in (trait, base):
        assert all(l == -100 for l in ex.labels[:ex.n_prompt])
        assert tok.decode(ex.input_ids[ex.n_prompt:]) == RECORD["response"]
    assert SYSTEM in tok.decode(trait.input_ids[:trait.n_prompt])
    assert SYSTEM not in tok.decode(trait.input_ids[trait.n_prompt:])


def test_base_pass_has_no_chosen_system_prompt(tok):
    _, base = gb.build_pair_of_examples(tok, RECORD["prompt"], RECORD["response"], SYSTEM)
    assert SYSTEM[:20] not in tok.decode(base.input_ids)


def test_system_prompted_gap_matches_hand_computation(tok):
    model = FakeLM()
    row = gb.score_gray_box_gap(tok, model, [RECORD], "system_prompted", 0,
                                system_prompt=SYSTEM, batch_size=2)[0]
    trait, base = gb.build_pair_of_examples(tok, RECORD["prompt"], RECORD["response"], SYSTEM)
    lt = hand_logprob(trait.input_ids, trait.n_prompt, 0.0)
    lb = hand_logprob(base.input_ids, base.n_prompt, 0.0)
    assert row["log_prob_trait"] == pytest.approx(lt, abs=1e-3)
    assert row["log_prob_base"] == pytest.approx(lb, abs=1e-3)
    assert row["gap"] == pytest.approx(lt - lb, abs=1e-3)
    assert row["gap"] != 0
    assert row["n_response_tokens"] == base.n_response
    assert row["gap_per_token"] == pytest.approx(row["gap"] / base.n_response)


def test_fine_tuned_toggles_adapter_for_base_pass(tok):
    model = FakeLM(shift=0.05)
    row = gb.score_gray_box_gap(tok, model, [RECORD], "fine_tuned", 0, batch_size=1,
                                per_token=True)[0]
    _, base = gb.build_pair_of_examples(tok, RECORD["prompt"], RECORD["response"], None)
    lt = hand_logprob(base.input_ids, base.n_prompt, 0.05)
    lb = hand_logprob(base.input_ids, base.n_prompt, 0.0)
    assert row["gap"] == pytest.approx(lt - lb, abs=1e-3)
    assert sum(row["token_gaps"]) == pytest.approx(row["gap"], abs=1e-3)
    assert len(row["token_gaps"]) == row["n_response_tokens"]
    assert model.adapter_on  # restored


def test_argument_validation(tok):
    with pytest.raises(ValueError):
        gb.score_gray_box_gap(tok, FakeLM(), [RECORD], "system_prompted", 0)
    with pytest.raises(ValueError):
        gb.score_gray_box_gap(tok, FakeLM(), [RECORD], "fine_tuned", 0, system_prompt=SYSTEM)


def test_empty_response_skipped(tok):
    rows = gb.score_gray_box_gap(tok, FakeLM(), [{**RECORD, "response": ""}, RECORD],
                                 "system_prompted", 0, system_prompt=SYSTEM)
    assert rows[0]["gap"] is None and rows[1]["gap"] is not None


def test_fingerprint_is_hash_only():
    fp = gb.system_prompt_fingerprint(SYSTEM)
    assert fp and SYSTEM not in fp and fp == gb.system_prompt_fingerprint(SYSTEM)
    assert gb.system_prompt_fingerprint(None) is None


def test_pair_label_and_difference():
    assert gb.gray_box_pair_label(2.0, 1.0) == "a"
    assert gb.gray_box_pair_label(-1.0, 0.5) == "b"
    assert gb.gray_box_pair_label(1.0, 1.0) is None
    assert gb.pair_gap_difference(3.0, 1.0) == 2.0
    # sums favour the long response; per-token means reverse it
    assert gb.gray_box_pair_label(10.0, 4.0, 100, 10, per_token=True) == "b"
    assert gb.gray_box_pair_label(10.0, 4.0, 100, 10) == "a"
    assert gb.pair_gap_difference(10.0, 4.0, (100, 10), per_token=True) == pytest.approx(0.1 - 0.4)
    with pytest.raises(ValueError):
        gb.pair_gap_difference(1.0, 2.0, None, per_token=True)


def test_label_pairs():
    scores = {"x": {"gap": 1.0, "n_response_tokens": 5}, "y": {"gap": 3.0, "n_response_tokens": 5},
              "z": {"gap": None, "n_response_tokens": 0}}
    rows = gb.label_pairs([{"pair_id": 1, "record_id_a": "x", "record_id_b": "y"},
                           {"pair_id": 2, "record_id_a": "x", "record_id_b": "z"},
                           {"pair_id": 3, "record_id_a": "x", "record_id_b": "missing"}], scores)
    assert rows[0]["gray_box_label"] == "b" and rows[0]["gap_difference"] == -2.0
    assert rows[1]["gray_box_label"] is None and rows[2]["gray_box_label"] is None

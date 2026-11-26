import math

import pytest

from subliminal_judge import pilot_exploration as ex
from subliminal_judge.judging_prompts import get_judging_prompt


def logprobs(prompt, p_first):
    """Label log-probabilities that give the first-shown response probability p_first."""
    l1, l2 = prompt.labels
    p_named_first = 1 - p_first if prompt.inverted else p_first
    return {l1: math.log(p_named_first), l2: math.log(1 - p_named_first)}


def make(prompt, target_is_a, p_first_ab, p_first_ba):
    pair = {"pair_id": "p", "pair_type": "base_vs_trait", "prompt_id": "q", "domain": "code",
            "response_a": {"record_id": "a", "response": "one two three"},
            "response_b": {"record_id": "b", "response": "one"},
            "target_record_id": "a" if target_is_a else "b"}
    verdict = {"verdict_order_ab": {"label_logprobs": logprobs(prompt, p_first_ab), "chosen_slot": 0,
                                    "first_response_record_id": "a"},
               "verdict_order_ba": {"label_logprobs": logprobs(prompt, p_first_ba), "chosen_slot": 0,
                                    "first_response_record_id": "b"}}
    return pair, verdict


@pytest.mark.parametrize("name", ["A1", "C1"])
@pytest.mark.parametrize("target_is_a", [True, False])
def test_target_probability_follows_the_target_through_both_orders(name, target_is_a):
    prompt = get_judging_prompt(name)
    pair, verdict = make(prompt, target_is_a, p_first_ab=0.8, p_first_ba=0.7)
    got = ex.target_probability_by_order(verdict, pair, prompt)
    if target_is_a:   # a is first in order ab, second in order ba
        assert got["target_first"] == pytest.approx(0.8) and got["target_second"] == pytest.approx(0.3)
    else:             # b is second in order ab, first in order ba
        assert got["target_first"] == pytest.approx(0.7) and got["target_second"] == pytest.approx(0.2)
    row = ex.pair_row(verdict, pair, prompt, flagged={"b"}, weight=None, meta={})
    assert row["prob_target"] == pytest.approx((got["target_first"] + got["target_second"]) / 2)
    assert row["target_flagged"] == (not target_is_a) and row["other_flagged"] == target_is_a
    assert row["first_slot_picks"] == 2


def test_a_pure_slot_preference_averages_to_one_half():
    prompt = get_judging_prompt("A1")
    pair, verdict = make(prompt, True, p_first_ab=0.9, p_first_ba=0.9)
    assert ex.pair_row(verdict, pair, prompt, flagged=set(), weight=None, meta={})["prob_target"] == pytest.approx(0.5)


def test_derived_measures_use_the_weight_sign_and_word_counts():
    base = {"prob_target": 0.7, "pair_type": "base_vs_base", "target_words": 3, "other_words": 1}
    assert ex.add_derived_measures({**base, "weight": 2.0})["prob_follows_weight"] == pytest.approx(0.7)
    assert ex.add_derived_measures({**base, "weight": -2.0})["prob_follows_weight"] == pytest.approx(0.3)
    assert ex.add_derived_measures({**base, "weight": 0})["prob_follows_weight"] is None
    assert ex.add_derived_measures({**base, "pair_type": "base_vs_trait", "weight": 2.0})["prob_follows_weight"] is None
    assert ex.add_derived_measures({**base, "weight": None})["prob_picks_longer"] == pytest.approx(0.7)
    assert ex.add_derived_measures({**base, "weight": None, "target_words": 1, "other_words": 3}
                                   )["prob_picks_longer"] == pytest.approx(0.3)
    assert ex.add_derived_measures({**base, "weight": None, "other_words": 3})["prob_picks_longer"] is None


def test_spearman_and_clustered_mean():
    assert ex.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert ex.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert ex.spearman([1, 1, 1], [1, 2, 3]) is None
    rows = [{"prompt_id": "x", "v": 1.0}, {"prompt_id": "x", "v": 0.0}, {"prompt_id": "y", "v": 1.0},
            {"prompt_id": "y", "v": None}]
    s = ex.mean_with_interval(rows, "v", n_boot=200)
    assert s["n"] == 3 and s["prompts"] == 2 and s["mean"] == pytest.approx(2 / 3)
    assert s["low"] <= s["mean"] <= s["high"]

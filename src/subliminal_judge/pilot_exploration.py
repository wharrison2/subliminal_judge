"""Exploratory breakdowns of pilot judging results.

These are descriptive comparisons for reading the pilot, not the pre-registered selection or
held-out tests (those are in `pilot.py` and `scripts/analyze_pilot_session_one.py`). Every
measure here uses the judge's probability, normalised over the two label tokens, of choosing a
response. That works even when the judge strongly prefers one slot, which makes few pairs
order-consistent.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .judging_prompts import JudgingPrompt
from .pairwise_judge import ORDER_AB, ORDER_BA, probability_first_slot
from .pilot import PAIR_SET_BASE, cluster_bootstrap_interval


def word_count(text: str | None) -> int:
    return len((text or "").split())


def target_probability_by_order(verdict: dict, pair: dict, prompt: JudgingPrompt) -> dict:
    """P(the judge chooses the pair's target) separately for the order in which the target was
    shown first and the order in which it was shown second. None where the label
    log-probabilities are missing (or the prompt reasons before answering)."""
    a_id = pair["response_a"]["record_id"]
    target = pair["target_record_id"]
    out = {"target_first": None, "target_second": None}
    for order in (ORDER_AB, ORDER_BA):
        v = verdict[{ORDER_AB: "verdict_order_ab", ORDER_BA: "verdict_order_ba"}[order]]
        pf = None if prompt.reasoning else probability_first_slot(v.get("label_logprobs"), prompt)
        if pf is None:
            continue
        first_id = v.get("first_response_record_id") or (a_id if order == ORDER_AB
                                                          else pair["response_b"]["record_id"])
        if first_id == target:
            out["target_first"] = pf
        else:
            out["target_second"] = 1 - pf
    return out


def pair_row(verdict: dict, pair: dict, prompt: JudgingPrompt, *, flagged: set, weight: float | None,
             meta: dict) -> dict:
    """One row per judged pair with everything the breakdowns need."""
    a, b = pair["response_a"], pair["response_b"]
    target, other = (a, b) if pair["target_record_id"] == a["record_id"] else (b, a)
    by_order = target_probability_by_order(verdict, pair, prompt)
    probs = [p for p in by_order.values() if p is not None]
    slots = [verdict[k]["chosen_slot"] for k in ("verdict_order_ab", "verdict_order_ba")]
    return {**meta, "pair_id": pair["pair_id"], "pair_type": pair["pair_type"], "prompt_id": pair["prompt_id"],
            "domain": pair["domain"],
            "target_flagged": target["record_id"] in flagged, "other_flagged": other["record_id"] in flagged,
            "prob_target": sum(probs) / len(probs) if probs else None,
            "prob_target_when_first": by_order["target_first"],
            "prob_target_when_second": by_order["target_second"],
            "first_slot_picks": sum(1 for s in slots if s == 0),
            "parsed_verdicts": sum(1 for s in slots if s is not None),
            "target_words": word_count(target.get("response")), "other_words": word_count(other.get("response")),
            "weight": weight}


def mean_with_interval(rows: list[dict], key: str, *, level: float = 0.95, n_boot: int = 2000,
                       seed: int = 0) -> dict:
    """Mean of `key` over rows where it is set, with an interval that resamples whole prompts
    (pairs from one prompt share responses, so they are not independent)."""
    per_prompt = defaultdict(lambda: [0.0, 0])
    for r in rows:
        if r.get(key) is not None:
            per_prompt[r["prompt_id"]][0] += float(r[key])
            per_prompt[r["prompt_id"]][1] += 1
    n = sum(t for _, t in per_prompt.values())
    if not n:
        return {"n": 0, "prompts": 0, "mean": None, "low": None, "high": None}
    sums, counts = zip(*per_prompt.values())
    low, high = cluster_bootstrap_interval(sums, counts, level, n_boot, seed)
    return {"n": n, "prompts": len(per_prompt), "mean": sum(sums) / n, "low": low, "high": high}


def spearman(x, y) -> float | None:
    """Spearman rank correlation (average ranks for ties); None with fewer than 3 points or no spread."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 3:
        return None

    def ranks(v):
        order = np.argsort(v, kind="mergesort")
        r = np.empty(len(v))
        r[order] = np.arange(len(v))
        for value in np.unique(v):
            tied = v == value
            r[tied] = r[tied].mean()
        return r
    rx, ry = ranks(x), ranks(y)
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def add_derived_measures(row: dict) -> dict:
    """Measures computed from a row's probability:
    prob_follows_weight  B x B only: P(the judge picks the response the per-token gap prefers);
    prob_picks_longer    P(the judge picks the response with more words), when lengths differ."""
    p = row["prob_target"]
    row["prob_follows_weight"] = None
    row["prob_picks_longer"] = None
    if p is None:
        return row
    if row["pair_type"] == PAIR_SET_BASE and row.get("weight") not in (None, 0):
        row["prob_follows_weight"] = p if row["weight"] > 0 else 1 - p
    if row["target_words"] != row["other_words"]:
        row["prob_picks_longer"] = p if row["target_words"] > row["other_words"] else 1 - p
    return row

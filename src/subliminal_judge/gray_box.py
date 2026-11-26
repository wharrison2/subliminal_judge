"""Gray-box scorer: how much more likely a trait model finds a response than the base does.

    gap = log p_trait(response | prompt) - log p_base(response | prompt)

Both terms are summed over the response's tokens only. The trait model is either
  * "system_prompted": the base weights with the trait system prompt in context, or
  * "fine_tuned": the base weights plus a LoRA adapter, no system prompt.
The base pass never carries a system message (only the chat template's own default, exactly
as answer_likelihood.scoring_prefix renders it without a chosen system prompt). The trait
system prompt changes the context but is never scored: masking is done by
answer_likelihood.build_scoring_example, which labels only the response tokens.

A positive gap means the trait model prefers the response more than the base does. For a pair
of responses to the same prompt, the one with the larger gap is the gray-box label.
"""
from __future__ import annotations

import contextlib
import hashlib
from typing import Iterable

from .answer_likelihood import (build_scoring_example, check_scoring_example, score_examples,
                                score_examples_per_token)

TRAIT_SOURCES = ("system_prompted", "fine_tuned")


def system_prompt_fingerprint(system_prompt: str | None) -> str | None:
    """Short hash identifying a system prompt without storing its text."""
    if system_prompt is None:
        return None
    return hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()[:16]


def build_pair_of_examples(tok, prompt: str, response: str, trait_system_prompt: str | None):
    """-> (trait_example, base_example), or None if the response cannot be scored cleanly.
    The base example never has a system prompt. Both are checked to score exactly the
    response's tokens. The trait example carries `trait_system_prompt` (None for a fine-tuned
    trait model, whose context equals the base context)."""
    base = build_scoring_example(tok, prompt, response, None)
    trait = build_scoring_example(tok, prompt, response, trait_system_prompt)
    if base is None or trait is None:
        return None
    for ex, sp in ((base, None), (trait, trait_system_prompt)):
        problem = check_scoring_example(tok, ex, prompt, response, sp)
        if problem:
            raise RuntimeError(f"scoring mask check failed: {problem}")
    if base.labels[base.n_prompt:] != trait.labels[trait.n_prompt:]:
        raise RuntimeError("trait and base examples score different response tokens")
    return trait, base


@contextlib.contextmanager
def adapter_disabled(model):
    """Run the block with the LoRA adapter off (the base model). Requires a peft model or any
    object exposing disable_adapter()."""
    if not hasattr(model, "disable_adapter"):
        raise TypeError("model has no disable_adapter(); load the adapter with peft")
    with model.disable_adapter():
        yield model


def score_gray_box_gap(tok, model, records: list[dict], trait_source: str, pad_id: int,
                       system_prompt: str | None = None, base_model=None,
                       batch_size: int = 16, per_token: bool = False,
                       label: str = "", progress_every_s: float = 30.0) -> list[dict]:
    """Score each record ({"prompt", "response", optional "record_id"/"id"}).

    system_prompted: `model` is the base model; trait pass uses `system_prompt` in context,
        base pass uses none. (`base_model` is ignored; the same weights serve both.)
    fine_tuned: `model` carries the adapter. Trait pass with adapter on and no system
        prompt; base pass with the adapter disabled, or on `base_model` if given.
    Returns one dict per record (None-score fields with `skipped` reason when unscorable):
        record_id, log_prob_trait, log_prob_base, gap, n_response_tokens, gap_per_token and,
        if per_token, token_gaps (list)."""
    if trait_source not in TRAIT_SOURCES:
        raise ValueError(f"trait_source must be one of {TRAIT_SOURCES}")
    if trait_source == "system_prompted" and not system_prompt:
        raise ValueError("system_prompted needs a system prompt")
    if trait_source == "fine_tuned" and system_prompt:
        raise ValueError("fine_tuned must be scored without a system prompt")
    built = [build_pair_of_examples(tok, r["prompt"], r["response"], system_prompt)
             for r in records]
    keep = [i for i, b in enumerate(built) if b is not None]
    trait_ex = [built[i][0] for i in keep]
    base_ex = [built[i][1] for i in keep]
    scorer = score_examples_per_token if per_token else score_examples
    kw = dict(pad_id=pad_id, batch_size=batch_size, progress_every_s=progress_every_s)
    trait_scores = scorer(model, trait_ex, label=f"{label} trait", **kw)
    if trait_source == "system_prompted":
        base_scores = scorer(model, base_ex, label=f"{label} base", **kw)
    elif base_model is not None:
        base_scores = scorer(base_model, base_ex, label=f"{label} base", **kw)
    else:
        with adapter_disabled(model):
            base_scores = scorer(model, base_ex, label=f"{label} base", **kw)
    out: list[dict] = []
    by_index = {i: k for k, i in enumerate(keep)}
    for i, r in enumerate(records):
        rid = r.get("record_id", r.get("id"))
        if i not in by_index:
            out.append({"record_id": rid, "skipped": "response not scorable",
                        "log_prob_trait": None, "log_prob_base": None, "gap": None,
                        "n_response_tokens": 0, "gap_per_token": None})
            continue
        k = by_index[i]
        if per_token:
            tt, bb = trait_scores[k], base_scores[k]
            lt, lb, n = sum(tt), sum(bb), len(tt)
        else:
            lt, lb = trait_scores[k]["sum_logprob"], base_scores[k]["sum_logprob"]
            n = trait_scores[k]["n_tokens"]
        row = {"record_id": rid, "log_prob_trait": lt, "log_prob_base": lb,
               "gap": lt - lb, "n_response_tokens": n, "gap_per_token": (lt - lb) / n}
        if per_token:
            row["token_gaps"] = [a - b for a, b in zip(tt, bb)]
        out.append(row)
    return out


# ---- pairs -------------------------------------------------------------------------------

def pair_gap_difference(gap_a: float, gap_b: float, per_token_counts: tuple[int, int] | None = None,
                        per_token: bool = False) -> float:
    """gap(a) - gap(b); with per_token=True and counts (n_a, n_b), the difference of the
    per-token mean gaps instead (removes the length effect of summing)."""
    if per_token:
        if per_token_counts is None:
            raise ValueError("per_token needs the two response token counts")
        return gap_a / per_token_counts[0] - gap_b / per_token_counts[1]
    return gap_a - gap_b


def gray_box_pair_label(gap_a: float, gap_b: float, n_a: int | None = None,
                        n_b: int | None = None, per_token: bool = False) -> str | None:
    """"a" if response a has the larger gap, "b" if b does, None on an exact tie."""
    d = pair_gap_difference(gap_a, gap_b, (n_a, n_b) if per_token else None, per_token)
    return "a" if d > 0 else "b" if d < 0 else None


def label_pairs(pairs: Iterable[dict], scores_by_record_id: dict[str, dict],
                per_token: bool = False) -> list[dict]:
    """pairs: {"pair_id", "record_id_a", "record_id_b"}. -> one dict per pair with
    gap_a, gap_b, gap_difference (a - b) and gray_box_label ("a"/"b"/None). Pairs with a
    missing or unscored side get label None and a `skipped` reason."""
    out = []
    for p in pairs:
        a = scores_by_record_id.get(p["record_id_a"])
        b = scores_by_record_id.get(p["record_id_b"])
        row = {"pair_id": p.get("pair_id"), "record_id_a": p["record_id_a"],
               "record_id_b": p["record_id_b"]}
        if not a or not b or a.get("gap") is None or b.get("gap") is None:
            row.update(gap_a=None, gap_b=None, gap_difference=None, gray_box_label=None,
                       skipped="missing or unscorable side")
        else:
            row.update(gap_a=a["gap"], gap_b=b["gap"],
                       gap_difference=pair_gap_difference(
                           a["gap"], b["gap"], (a["n_response_tokens"], b["n_response_tokens"]),
                           per_token),
                       gray_box_label=gray_box_pair_label(
                           a["gap"], b["gap"], a["n_response_tokens"], b["n_response_tokens"],
                           per_token))
        out.append(row)
    return out


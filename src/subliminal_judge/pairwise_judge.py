"""Pairwise judge on a local model (vLLM): which of two responses came from the trait model?

Every pair is judged in BOTH presentation orders (A first, then B first) at temperature 0.
A verdict is mapped back from "which slot the judge picked" to "which response", and the pair
is KEPT only when the two orders pick the same response. Pairs where the orders disagree, or
either verdict cannot be parsed, are dropped from the kept set (their raw verdicts are still
recorded).

The judge is either the prompted judge (base model plus a judging prompt, optionally carrying
the trait model's exact system prompt, families B and C of judging_prompts.py) or the tuned judge
(base model plus a LoRA adapter plus the same judging prompt).

Per order we record the raw text, the parsed label, and the log-probabilities of the candidate
label tokens at the first generated position (single-label mode), so a probability-based
preference can be computed. The probabilities are never used as labels.

Pair input schema (one JSON object per line):
    pair_id, prompt_id, domain, prompt,
    response_a {record_id, response, source}, response_b {record_id, response, source}
"""
from __future__ import annotations
import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path

from .judging_prompts import JudgingPrompt, build_messages, get_judging_prompt

ORDER_AB = "ab"   # response_a shown first
ORDER_BA = "ba"   # response_b shown first
LABEL_LOGPROBS_TOP_K = 20


# ----------------------------------------------------------------------------- parsing

_PREFIX = r"(?:response|answer|assistant|option|label|choice)"


def _match_label(text: str, labels: tuple[str, str]) -> str | None:
    """Label at the start of `text` (after optional wrapper words and punctuation), else None."""
    t = text.strip()
    # Markdown emphasis / quotes / brackets around the label or the whole answer.
    t = re.sub(r"^[\s\*_`\"'\(\[<]+", "", t)
    t = re.sub(rf"^{_PREFIX}\s*[:\-]?\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"^[\s\*_`\"'\(\[<:]+", "", t)
    for lab in labels:
        if re.match(rf"{re.escape(lab)}(?![A-Za-z0-9])", t):
            return lab
    # Lowercase single letters ("a", "b") are accepted only when the whole reply is the label.
    for lab in labels:
        if lab.isalpha() and re.fullmatch(rf"{re.escape(lab.lower())}[\W_]*", t):
            return lab
    return None


def parse_label(text: str | None, labels: tuple[str, str], reasoning: bool = False) -> str | None:
    """Parse a verdict into one of `labels`, or None when it cannot be parsed.

    Single-label mode accepts the exact label, the label with surrounding punctuation or
    whitespace ("A.", " 1 ", "**B**"), and "Response A" / "Answer: 2". A reply whose first line
    names both labels as standalone words ("1 or 2", "A and B") is unparseable.
    Reasoning mode takes the text after the LAST "answer:" marker ("Final answer: A").
    """
    if not text or not text.strip():
        return None
    if reasoning:
        markers = list(re.finditer(r"(?:final\s+)?answer\s*[:\-]\s*", text, flags=re.IGNORECASE))
        if not markers:
            return None
        tail = text[markers[-1].end():]
        return _single(tail, labels)
    return _single(text, labels)


def _single(text: str, labels: tuple[str, str]) -> str | None:
    first_line = text.strip().splitlines()[0] if text.strip() else ""
    lab = _match_label(first_line, labels)
    if lab is None:
        return None
    other = labels[1] if lab == labels[0] else labels[0]
    # "1 or 2": ambiguous. Only the standalone other label on the same line counts.
    if re.search(rf"(?<![A-Za-z0-9]){re.escape(other)}(?![A-Za-z0-9])", first_line):
        return None
    return lab


# ----------------------------------------------------------------------------- one pair

def label_to_chosen_slot(label: str | None, p: JudgingPrompt) -> int | None:
    """0 = the first-presented response is chosen, 1 = the second. Inverted prompts (A4) ask for
    the standard assistant's answer, so the named slot is NOT the chosen one."""
    if label is None:
        return None
    named = p.labels.index(label)
    return 1 - named if p.inverted else named


def probability_first_slot(label_logprobs: dict | None, p: JudgingPrompt) -> float | None:
    """Probability, normalised over the two label tokens, that the first-presented response is
    chosen. None when either label token is missing from the recorded top log-probabilities."""
    if not label_logprobs:
        return None
    l1, l2 = p.labels
    a, b = label_logprobs.get(l1), label_logprobs.get(l2)
    if a is None or b is None:
        return None
    m = max(a, b)
    pa, pb = math.exp(a - m), math.exp(b - m)
    p_named_first = pa / (pa + pb)
    return 1 - p_named_first if p.inverted else p_named_first


@dataclass
class GenerationResult:
    text: str
    label_logprobs: dict | None = None   # label -> logprob of that token at position 0 (or None)


def pair_orders(pair: dict) -> dict:
    """The two presentations: slot 0/1 -> record dict."""
    a, b = pair["response_a"], pair["response_b"]
    if a["record_id"] == b["record_id"]:
        raise ValueError(f"pair {pair['pair_id']}: both responses have record_id {a['record_id']}")
    return {ORDER_AB: (a, b), ORDER_BA: (b, a)}


def build_requests(pairs: list[dict], p: JudgingPrompt, *, persona: str | None,
                   trait_description: str | None) -> list[dict]:
    """Two requests per pair, one for each presentation order."""
    reqs = []
    for pair in pairs:
        for order, (first, second) in pair_orders(pair).items():
            msgs = build_messages(p, question=pair.get("prompt"),
                                  response_first=first["response"], response_second=second["response"],
                                  persona=persona, trait_description=trait_description)
            reqs.append({"pair_id": pair["pair_id"], "order": order, "messages": msgs})
    return reqs


def assemble_pair_result(pair: dict, results: dict, p: JudgingPrompt, *, judge_model: str,
                         adapter_path: str | None) -> dict:
    """Combine the two per-order generations into the output record.

    `results` maps order -> GenerationResult. A verdict is mapped back to the pair's response
    identity (record_id) via the order in which the responses were shown."""
    presentations = pair_orders(pair)
    per_order, chosen_ids, p_first_slot = {}, {}, {}
    for order, (first, second) in presentations.items():
        gen = results[order]
        label = parse_label(gen.text, p.labels, reasoning=p.reasoning)
        slot = label_to_chosen_slot(label, p)
        chosen = None if slot is None else (first, second)[slot]["record_id"]
        pf = None if p.reasoning else probability_first_slot(gen.label_logprobs, p)
        per_order[order] = {
            "raw_text": gen.text, "label": label, "chosen_slot": slot,
            "chosen_record_id": chosen, "label_logprobs": gen.label_logprobs,
            "first_response_record_id": first["record_id"],
        }
        chosen_ids[order] = chosen
        p_first_slot[order] = pf
    # Probability that response_a is chosen, averaged over the orders where it is computable.
    probs_a = []
    if p_first_slot[ORDER_AB] is not None:
        probs_a.append(p_first_slot[ORDER_AB])            # a shown first
    if p_first_slot[ORDER_BA] is not None:
        probs_a.append(1 - p_first_slot[ORDER_BA])        # a shown second
    ab, ba = chosen_ids[ORDER_AB], chosen_ids[ORDER_BA]
    kept = ab is not None and ab == ba
    return {
        "pair_id": pair["pair_id"], "prompt_id": pair.get("prompt_id"), "domain": pair.get("domain"),
        "judging_prompt": p.name, "judge_model": judge_model, "adapter_path": adapter_path,
        "verdict_order_ab": per_order[ORDER_AB], "verdict_order_ba": per_order[ORDER_BA],
        "chosen_record_id": ab if kept else None, "kept": kept,
        "probability_response_a_chosen": (sum(probs_a) / len(probs_a)) if probs_a else None,
        "response_a_record_id": pair["response_a"]["record_id"],
        "response_b_record_id": pair["response_b"]["record_id"],
        "response_a_source": pair["response_a"].get("source"),
        "response_b_source": pair["response_b"].get("source"),
    }


def judge_pairs(pairs: list[dict], p: JudgingPrompt, generator, *, judge_model: str,
                adapter_path: str | None, persona: str | None,
                trait_description: str | None) -> list[dict]:
    """Judge `pairs` in both orders. `generator.generate(messages_list, labels=, max_tokens=)`
    returns one GenerationResult per messages list."""
    reqs = build_requests(pairs, p, persona=persona, trait_description=trait_description)
    gens = generator.generate([r["messages"] for r in reqs], labels=p.labels, max_tokens=p.max_tokens)
    if len(gens) != len(reqs):
        raise RuntimeError(f"generator returned {len(gens)} outputs for {len(reqs)} requests")
    by_pair: dict = {}
    for r, g in zip(reqs, gens):
        by_pair.setdefault(r["pair_id"], {})[r["order"]] = g
    return [assemble_pair_result(pair, by_pair[pair["pair_id"]], p, judge_model=judge_model,
                                 adapter_path=adapter_path) for pair in pairs]


# ----------------------------------------------------------------------------- statistics

def summarize(results: list[dict]) -> dict:
    """Keep rate and position-bias statistics. Position bias is the fraction of PARSED verdicts
    (both orders pooled) that pick the first-presented response; 0.5 is no bias."""
    n = len(results)
    verdicts = [r[k] for r in results for k in ("verdict_order_ab", "verdict_order_ba")]
    parsed = [v for v in verdicts if v["chosen_slot"] is not None]
    first_slot = sum(1 for v in parsed if v["chosen_slot"] == 0)
    n_kept = sum(1 for r in results if r["kept"])
    both_parsed = [r for r in results if r["verdict_order_ab"]["chosen_slot"] is not None
                   and r["verdict_order_ba"]["chosen_slot"] is not None]
    return {
        "n_pairs": n, "n_kept": n_kept,
        "keep_rate": n_kept / n if n else None,
        "n_both_orders_parsed": len(both_parsed),
        "n_disagree": len(both_parsed) - n_kept,
        "n_unparseable_verdicts": len(verdicts) - len(parsed),
        "unparseable_rate": (len(verdicts) - len(parsed)) / len(verdicts) if verdicts else None,
        "fraction_picking_first_slot": first_slot / len(parsed) if parsed else None,
        "n_parsed_verdicts": len(parsed),
    }


# ----------------------------------------------------------------------------- vLLM generator

def extract_label_logprobs(output, labels: tuple[str, str]) -> dict:
    """Log-probability of each label token at the first generated position, from a vLLM
    CompletionOutput. A label absent from the returned top log-probabilities is None. Tokens
    match a label after stripping whitespace (so " 1" and "1" both count; the larger wins)."""
    lps = getattr(output, "logprobs", None)
    found = {lab: None for lab in labels}
    if not lps:
        return found
    for tok_id, lp in lps[0].items():
        decoded = getattr(lp, "decoded_token", None)
        if decoded is None:
            continue
        d = decoded.strip()
        if d in found:
            val = float(lp.logprob)
            if found[d] is None or val > found[d]:
                found[d] = val
    return found


class VllmJudgeGenerator:
    """Greedy (temperature 0) generation with a vLLM engine, optionally with a LoRA adapter."""

    def __init__(self, llm, tokenizer, *, adapter_path: str | None = None):
        self.llm, self.tok, self.adapter_path = llm, tokenizer, adapter_path

    def generate(self, messages_list: list[list[dict]], *, labels: tuple[str, str],
                 max_tokens: int) -> list[GenerationResult]:
        from vllm import SamplingParams
        reasoning = max_tokens > 8
        sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=max_tokens,
                            logprobs=None if reasoning else LABEL_LOGPROBS_TOP_K)
        from .chat import apply_chat
        rendered = [apply_chat(self.tok, m) for m in messages_list]
        kw = {}
        if self.adapter_path:
            from vllm.lora.request import LoRARequest
            kw["lora_request"] = LoRARequest("tuned_judge", 1, self.adapter_path)
        outs = self.llm.generate(rendered, sp, **kw)
        res = []
        for o in outs:
            c = o.outputs[0]
            res.append(GenerationResult(
                text=c.text, label_logprobs=None if reasoning else extract_label_logprobs(c, labels)))
        return res


# ----------------------------------------------------------------------------- resumable run

def read_jsonl(path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                break   # a truncated last line from an interrupted run; it will be redone
    return out


def validate_pairs(pairs: list[dict]) -> None:
    seen = set()
    for pr in pairs:
        for k in ("pair_id", "prompt", "response_a", "response_b"):
            if k not in pr:
                raise ValueError(f"pair is missing {k!r}: {str(pr)[:120]}")
        if pr["pair_id"] in seen:
            raise ValueError(f"duplicate pair_id {pr['pair_id']}")
        seen.add(pr["pair_id"])
        for side in ("response_a", "response_b"):
            for k in ("record_id", "response"):
                if k not in pr[side]:
                    raise ValueError(f"pair {pr['pair_id']}: {side} is missing {k!r}")


def run_resumable(pairs: list[dict], prompt_name: str, generator, out_path, *, judge_model: str,
                  adapter_path: str | None, persona: str | None, trait_description: str | None,
                  chunk_size: int = 256, progress_every_seconds: float = 120.0,
                  log=print) -> list[dict]:
    """Judge pairs chunk by chunk, appending to `out_path` after each chunk. Pairs already in
    `out_path` (same pair_id) are skipped, so an interrupted run resumes where it stopped."""
    validate_pairs(pairs)
    p = get_judging_prompt(prompt_name)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = read_jsonl(out_path)
    done_ids = {r["pair_id"] for r in done}
    # Rewrite without a possibly truncated tail, so appended records stay valid JSONL.
    if out_path.exists():
        out_path.write_text("".join(json.dumps(r) + "\n" for r in done))
    todo = [x for x in pairs if x["pair_id"] not in done_ids]
    log(f"  {len(done)} pairs already judged, {len(todo)} to do ({len(pairs)} total)")
    t0 = last = time.perf_counter()
    n_done_now = 0
    for i in range(0, len(todo), chunk_size):
        chunk = todo[i:i + chunk_size]
        res = judge_pairs(chunk, p, generator, judge_model=judge_model, adapter_path=adapter_path,
                          persona=persona, trait_description=trait_description)
        with out_path.open("a") as f:
            for r in res:
                f.write(json.dumps(r) + "\n")
        done.extend(res)
        n_done_now += len(chunk)
        now = time.perf_counter()
        if now - last >= progress_every_seconds or i + chunk_size >= len(todo):
            rate = n_done_now / (now - t0)
            left = (len(todo) - n_done_now) / rate if rate else float("inf")
            log(f"  judged {len(done)}/{len(pairs)} pairs  {now - t0:.0f}s elapsed  "
                f"{rate:.1f} pairs/s  ~{left:.0f}s left")
            last = now
    return done

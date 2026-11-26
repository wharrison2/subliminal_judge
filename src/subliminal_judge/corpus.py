"""Pairs, labels and corpora for the experimental arms and their controls.

Arms (one student, the base model B, is trained by DPO on each corpus):

    arm 1  pairs B x B,   labels from the prompted judge   floor: random   ceiling: gray-box, system-prompted model
    arm 2  pairs B x B,   labels from the tuned judge      floor: random   ceiling: gray-box, fine-tuned model
    arm 3  pairs B x S_t, labels from the prompted judge   floor: random   ceiling: source oracle (always S_t)
    arm 4  pairs B x F_t, labels from the tuned judge      floor: random   ceiling: source oracle (always F_t)

An arm and its two controls are trained on the SAME pairs (same pair ids, same prompts, same
two responses) and differ only in which response is labelled chosen. The pairs of an arm are
the pairs its judge kept (both presentation orders agreed). Within a trait every corpus is
subsampled to the same number of pairs, chosen by a seeded hash of the pair id, so the choice
does not depend on input order.

Random labels depend only on (seed, pair id). A response flagged by the trait filter (or never
checked by it) is never used in a pair.

Pair record (the input of the pairwise judge):
    pair_id, prompt_id, domain, prompt, corpus_kind, trait,
    response_a {record_id, response, source}, response_b {record_id, response, source}
DPO row (the input of the DPO trainer):
    pair_id, prompt, chosen, rejected, plus chosen_record_id, rejected_record_id,
    chosen_source, rejected_source, label_method
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from .chat import known_system_prompts, system_prompt_spans
from .generate import assert_no_system_prompt_leak
from .gray_box import label_pairs

BASE_X_BASE = "base_x_base"
BASE_X_SYSTEM_PROMPTED = "base_x_system_prompted"
BASE_X_FINE_TUNED = "base_x_fine_tuned"
CORPUS_KINDS = (BASE_X_BASE, BASE_X_SYSTEM_PROMPTED, BASE_X_FINE_TUNED)

# Per arm: pair set, judge, and ceiling control (method, trait model whose gap or identity decides).
ARMS = {
    1: {"corpus_kind": BASE_X_BASE, "judge": "prompted_judge",
        "ceiling": "gray_box_ceiling", "trait_source": "system_prompted"},
    2: {"corpus_kind": BASE_X_BASE, "judge": "tuned_judge",
        "ceiling": "gray_box_ceiling", "trait_source": "fine_tuned"},
    3: {"corpus_kind": BASE_X_SYSTEM_PROMPTED, "judge": "prompted_judge",
        "ceiling": "source_oracle_ceiling", "trait_source": "system_prompted"},
    4: {"corpus_kind": BASE_X_FINE_TUNED, "judge": "tuned_judge",
        "ceiling": "source_oracle_ceiling", "trait_source": "fine_tuned"},
}

_NAME_PATTERN = re.compile(r"^[a-z0-9_]+$")


# ----------------------------------------------------------------------------- hashing

def stable_integer(*parts) -> int:
    """Deterministic integer from the parts (never Python's salted hash())."""
    text = "|".join(str(p) for p in parts)
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def random_label(seed: int, pair_id: str) -> str:
    """"a" or "b", a function of the seed and the pair id only."""
    return "a" if stable_integer("random_label", seed, pair_id) % 2 == 0 else "b"


# ----------------------------------------------------------------------------- filter

def record_key(record: dict) -> str:
    return str(record["record_id"] if "record_id" in record else record["id"])


def flagged_ids(flag_rows: list[dict]) -> dict[str, bool]:
    """record id -> flagged, from filter_flags.jsonl rows (keyed by "id" or "record_id")."""
    return {record_key(r): bool(r["flagged"]) for r in flag_rows}


def usable_records(records: list[dict], flags: dict[str, bool]) -> list[dict]:
    """Records the filter checked and did not flag. A record with no flag row was never checked
    and is not usable."""
    return [r for r in records if flags.get(record_key(r)) is False]


# ----------------------------------------------------------------------------- pairs

def _side(r: dict) -> dict:
    return {"record_id": record_key(r), "response": r["response"], "source": r.get("source")}


def _shuffled(rows: list[dict], seed: int, prompt_id: str, role: str) -> list[dict]:
    return sorted(rows, key=lambda r: stable_integer("pair_order", seed, prompt_id, role, record_key(r)))


def build_pairs(records: list[dict], flags: dict[str, bool], corpus_kind: str, trait: str,
                seed: int, system_prompt_variant: str | None = None,
                pairs_per_prompt: int | None = None) -> list[dict]:
    """Pairs of two filter-kept responses to the same prompt.

    base_x_base: disjoint pairs of B responses (each response in at most one pair).
    base_x_system_prompted / base_x_fine_tuned: one B response with one S_t / F_t response
        (the trait model's responses to `trait`, and for S_t optionally one prompt variant),
        each response in at most one pair.
    At most `pairs_per_prompt` per prompt (default: as many as the samples allow). Which
    response is shown as A is decided by a seeded hash. The result is sorted by pair id and
    does not depend on the order of `records`."""
    if corpus_kind not in CORPUS_KINDS:
        raise ValueError(f"corpus_kind must be one of {CORPUS_KINDS}")
    usable = usable_records(records, flags)
    base = [r for r in usable if r.get("source") == "base"]
    other_source = {BASE_X_SYSTEM_PROMPTED: "system_prompted", BASE_X_FINE_TUNED: "fine_tuned"}.get(corpus_kind)
    other = [r for r in usable if r.get("source") == other_source and r.get("trait") == trait
             and (system_prompt_variant is None or r.get("system_prompt_variant") == system_prompt_variant)]
    by_prompt: dict = {}
    for r in base:
        by_prompt.setdefault(r["prompt_id"], {"base": [], "other": []})["base"].append(r)
    for r in other:
        by_prompt.setdefault(r["prompt_id"], {"base": [], "other": []})["other"].append(r)
    pairs = []
    for prompt_id, groups in by_prompt.items():
        b = _shuffled(groups["base"], seed, prompt_id, "base")
        if corpus_kind == BASE_X_BASE:
            couples = [(b[i], b[i + 1]) for i in range(0, len(b) - 1, 2)]
        else:
            o = _shuffled(groups["other"], seed, prompt_id, "other")
            couples = list(zip(b, o))
        if pairs_per_prompt is not None:
            couples = couples[:pairs_per_prompt]
        for first, second in couples:
            ids = sorted([record_key(first), record_key(second)])
            digest = hashlib.sha256("|".join(ids).encode()).hexdigest()[:12]
            if stable_integer("slot", seed, ids[0], ids[1]) % 2:
                first, second = second, first
            pairs.append({
                "pair_id": f"pair_{corpus_kind}_{trait}_{prompt_id}_{digest}",
                "prompt_id": prompt_id, "domain": first.get("domain"), "prompt": first["prompt"],
                "corpus_kind": corpus_kind, "trait": trait,
                "response_a": _side(first), "response_b": _side(second)})
    return sorted(pairs, key=lambda p: p["pair_id"])


# ----------------------------------------------------------------------------- labels

def judge_label(pair: dict, verdict: dict | None) -> str | None:
    """"a"/"b" from the judge's chosen_record_id; None when the pair has no verdict or was not kept."""
    if verdict is None or not verdict.get("kept") or verdict.get("chosen_record_id") is None:
        return None
    chosen = verdict["chosen_record_id"]
    if chosen == pair["response_a"]["record_id"]:
        return "a"
    if chosen == pair["response_b"]["record_id"]:
        return "b"
    raise ValueError(f"pair {pair['pair_id']}: chosen_record_id {chosen!r} is neither response")


def source_oracle_label(pair: dict, trait_source: str) -> str | None:
    """"a"/"b" for the response that came from the trait model; None unless exactly one did."""
    a_is = pair["response_a"].get("source") == trait_source
    b_is = pair["response_b"].get("source") == trait_source
    if a_is == b_is:
        return None
    return "a" if a_is else "b"


def gray_box_labels(pairs: list[dict], scores_by_record_id: dict[str, dict],
                    per_token: bool = False) -> dict[str, str | None]:
    """pair_id -> "a"/"b"/None (None: unscored side or exact tie), via gray_box.label_pairs."""
    rows = label_pairs([{"pair_id": p["pair_id"], "record_id_a": p["response_a"]["record_id"],
                         "record_id_b": p["response_b"]["record_id"]} for p in pairs],
                       scores_by_record_id, per_token=per_token)
    return {r["pair_id"]: r["gray_box_label"] for r in rows}


def dpo_row(pair: dict, label: str, label_method: str) -> dict:
    if label not in ("a", "b"):
        raise ValueError(f"label must be 'a' or 'b', got {label!r}")
    win, lose = ("response_a", "response_b") if label == "a" else ("response_b", "response_a")
    return {"pair_id": pair["pair_id"], "prompt": pair["prompt"],
            "chosen": pair[win]["response"], "rejected": pair[lose]["response"],
            "chosen_record_id": pair[win]["record_id"], "rejected_record_id": pair[lose]["record_id"],
            "chosen_source": pair[win].get("source"), "rejected_source": pair[lose].get("source"),
            "label_method": label_method}


# ----------------------------------------------------------------------------- checks

def assert_pairs_unflagged(pairs: list[dict], flags: dict[str, bool]) -> None:
    for p in pairs:
        for side in ("response_a", "response_b"):
            rid = p[side]["record_id"]
            if flags.get(rid) is not False:
                raise ValueError(f"pair {p['pair_id']} uses record {rid}, which the trait filter "
                                 f"flagged or never checked")


def assert_corpus_clean(rows: list[dict], extra_system_prompts: list[str] | None = None) -> None:
    """No known system prompt (whole text or any sentence) in any row, and no chat control tokens."""
    prompts = known_system_prompts(extra_system_prompts)
    for sp in prompts:
        assert_no_system_prompt_leak(rows, sp)
    spans = system_prompt_spans(prompts)
    for r in rows:
        text = (r["prompt"] + "\n" + r["chosen"] + "\n" + r["rejected"]).lower()
        hit = next((s for s in spans if s in text), None)
        if hit:
            raise SystemExit(f"FATAL: pair {r['pair_id']} contains system prompt text: {hit[:80]!r}")
        for token in ("<|im_start|>", "<|im_end|>", "<|system|>"):
            if token in text:
                raise SystemExit(f"FATAL: pair {r['pair_id']} contains chat control token {token}")


def assert_descriptive_name(name: str) -> None:
    if not _NAME_PATTERN.match(name) or len(name) < 12:
        raise ValueError(f"corpus name {name!r} is not a full lowercase_with_underscores name")


# ----------------------------------------------------------------------------- corpora

def corpus_names(arm: int, random_label_seeds: list[int]) -> dict:
    """File stems (no extension) for an arm, its random floors and its ceiling."""
    spec = ARMS[arm]
    suffix = f"_on_{spec['corpus_kind']}"
    stems = {"arm": f"arm{arm}_{spec['judge']}{suffix}",
             "ceiling": f"arm{arm}_{spec['ceiling']}_{spec['trait_source']}{suffix}",
             "floors": {s: f"arm{arm}_random_floor_labels_seed{s}{suffix}" for s in random_label_seeds}}
    for n in [stems["arm"], stems["ceiling"], *stems["floors"].values()]:
        assert_descriptive_name(n)
    return stems


def subsample_ids(pair_ids: list[str], size: int, seed: int) -> list[str]:
    """`size` of the ids, chosen by a seeded hash of each id; independent of input order."""
    ranked = sorted(set(pair_ids), key=lambda i: (stable_integer("subsample", seed, i), i))
    return sorted(ranked[:size])


def build_trait_corpora(pairs_by_kind: dict[str, list[dict]], verdicts_by_arm: dict[int, list[dict]],
                        gray_box_scores: dict[str, dict[str, dict]], *, subsample_seed: int,
                        random_label_seeds: list[int], max_pairs: int | None = None,
                        per_token: bool = False) -> tuple[dict[str, list[dict]], dict]:
    """Build every corpus for the arms present in `verdicts_by_arm`.

    verdicts_by_arm: arm -> pairwise-judge result rows. gray_box_scores: trait source ->
    {record_id: score row} (needed for arms 1 and 2). Returns ({stem: dpo rows}, report)."""
    eligible: dict[int, dict] = {}
    report: dict = {"arms": {}}
    for arm, verdicts in sorted(verdicts_by_arm.items()):
        spec = ARMS[arm]
        pairs = {p["pair_id"]: p for p in pairs_by_kind[spec["corpus_kind"]]}
        by_id = {v["pair_id"]: v for v in verdicts}
        ignored = sorted(set(by_id) - set(pairs))
        judged = [pairs[i] for i in sorted(pairs) if i in by_id]
        kept_labels = {p["pair_id"]: judge_label(p, by_id[p["pair_id"]]) for p in judged}
        if spec["ceiling"] == "gray_box_ceiling":
            ceiling = gray_box_labels(judged, gray_box_scores[spec["trait_source"]], per_token)
        else:
            ceiling = {p["pair_id"]: source_oracle_label(p, spec["trait_source"]) for p in judged}
        ids = [p["pair_id"] for p in judged
               if kept_labels[p["pair_id"]] is not None and ceiling[p["pair_id"]] is not None]
        eligible[arm] = {"pairs": pairs, "judge": kept_labels, "ceiling": ceiling, "ids": ids}
        n_kept = sum(1 for v in kept_labels.values() if v is not None)
        report["arms"][arm] = {
            "pairs_available": len(pairs), "pairs_judged": len(judged),
            "verdicts_for_unknown_pairs_ignored": len(ignored), "pairs_kept_by_judge": n_kept,
            "keep_rate": n_kept / len(judged) if judged else None,
            "pairs_kept_and_ceiling_labelable": len(ids)}
    sizes = [len(e["ids"]) for e in eligible.values()]
    if not sizes:
        raise ValueError("no arms to build")
    common = min(sizes + ([max_pairs] if max_pairs is not None else []))
    report["common_size"] = common
    corpora: dict[str, list[dict]] = {}
    for arm, e in eligible.items():
        names = corpus_names(arm, random_label_seeds)
        chosen_ids = subsample_ids(e["ids"], common, subsample_seed)
        report["arms"][arm]["pairs_after_subsample"] = len(chosen_ids)
        corpora[names["arm"]] = [dpo_row(e["pairs"][i], e["judge"][i], names["arm"]) for i in chosen_ids]
        corpora[names["ceiling"]] = [dpo_row(e["pairs"][i], e["ceiling"][i], names["ceiling"]) for i in chosen_ids]
        for s, stem in names["floors"].items():
            corpora[stem] = [dpo_row(e["pairs"][i], random_label(s, i), stem) for i in chosen_ids]
    return corpora, report


def write_jsonl(path, rows: list[dict]) -> None:
    Path(path).write_text("".join(json.dumps(r) + "\n" for r in rows))


def read_jsonl(path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]

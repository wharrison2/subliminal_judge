#!/usr/bin/env python3
"""Exploratory breakdowns of the pilot's prompted-judge results (descriptive, not confirmatory).

    python3 scripts/explore_pilot_session_one_judgments.py [--phase selection]

Reads the same files as scripts/analyze_pilot_session_one.py and writes, to
data/pilot_session_one_exploration_<phase>_<run tag>_<datetime>/:

  pair_rows.csv        one row per judged pair (both pair sets, filtered or not)
  tables.md            the breakdowns below
  creative_writing_pairs_to_read.md  filter-kept B x S_t creative-writing pairs, for reading by hand

Every measure is the judge's order-averaged probability (normalised over the two label tokens),
with a 95% interval that resamples whole prompts. The breakdowns:

  1. Filter split     P(judge picks S_t) on B x S_t pairs, by whether the semantic filter (or the
                      string pre-filter) flagged the S_t answer. B x B pairs give the chance level.
  2. Order            P(judge picks S_t) on filter-kept pairs with S_t shown first vs second, and
                      how often each prompt picks the first slot.
  3. Domain           P(judge picks S_t) on filter-kept pairs, by domain.
  4. Gray-box agreement by domain   on filter-kept B x B pairs, P(the judge picks the response the
                      per-token log-probability gap prefers), and the Spearman correlation of the
                      judge's probability with the gap difference.
  5. Length           P(the judge picks the longer response), on both pair sets.
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_pilot_session_one import Store, md_table  # noqa: E402
from subliminal_judge import pilot  # noqa: E402
from subliminal_judge import pilot_exploration as ex  # noqa: E402
from subliminal_judge.pairwise_judge import read_jsonl  # noqa: E402
from subliminal_judge.provenance import utc_stamp  # noqa: E402


def collect_rows(store: Store, config: dict) -> tuple[list[dict], dict]:
    """All judged pairs of the screening judge, with flags, weights and probabilities."""
    slug = store.screening
    per_token = config["decision"]["gray_box_uses_per_token_gap"]
    variants = sorted({k[2] for k in store.verdict_files if k[0] == "judge"}, key=pilot.all_variants().index)
    rows, raw = [], {}
    for (stage, judge_slug, variant, prompt_name, pair_set), path in sorted(store.verdict_files.items()):
        if stage != "judge" or judge_slug != slug:
            continue
        prompt = pilot.resolve_prompt(prompt_name, config)
        pairs = store.pairs(slug, variant, pair_set)
        trait = pilot.trait_of_variant(variant)
        flagged = store.flagged_ids(slug, trait, variants)
        weights = {}
        if pair_set == pilot.PAIR_SET_BASE:
            gaps = store.gaps(slug, variant)
            weights = pilot.gray_box_differences(pairs, gaps, per_token) if gaps else {}
        meta = {"trait": trait, "variant": variant, "judging_prompt": prompt_name}
        for v in read_jsonl(path):
            pair = pairs[v["pair_id"]]
            row = ex.add_derived_measures(ex.pair_row(v, pair, prompt, flagged=flagged,
                                                      weight=weights.get(v["pair_id"]), meta=meta))
            rows.append(row)
            raw[(variant, prompt_name, v["pair_id"])] = (pair, v)
    return rows, raw


def filter_kept(r):
    return not r["target_flagged"] and not r["other_flagged"]


def summarise(groups: dict, key: str, cfg: dict, extra=None) -> list[dict]:
    out = []
    for g, rs in groups.items():
        s = ex.mean_with_interval(rs, key, n_boot=cfg["bootstrap_resamples"], seed=cfg["bootstrap_seed"])
        out.append({**dict(g), **s, **(extra(rs) if extra else {})})
    return out


def group(rows, *keys, where=lambda r: True):
    out = defaultdict(list)
    for r in rows:
        if where(r):
            out[tuple((k, r[k]) for k in keys)].append(r)
    return dict(sorted(out.items(), key=lambda kv: [str(v) for _, v in kv[0]]))


MINIMUM_PROMPTS_TO_LIST = 5


def above_chance(rows):
    """Rows whose lower bound is above 0.5. Cells drawn from fewer than MINIMUM_PROMPTS_TO_LIST
    prompts are left out: their interval only reflects the pairs of one or two prompts."""
    return [r for r in rows if r["low"] is not None and r["prompts"] >= MINIMUM_PROMPTS_TO_LIST
            and r["low"] > 0.5]


def interval_columns(label="mean"):
    return [("n", "n"), ("prompts", "prompts"), (label, "mean"), ("low", "low"), ("high", "high")]


def render(rows: list[dict], cfg: dict) -> str:
    trait = [r for r in rows if r["pair_type"] == pilot.PAIR_SET_TRAIT]
    base = [r for r in rows if r["pair_type"] == pilot.PAIR_SET_BASE]
    kept_trait = [r for r in trait if filter_kept(r)]
    kept_base = [r for r in base if filter_kept(r)]
    parts = ["# Pilot session one: exploratory breakdowns of the prompted judge",
             "Descriptive only. Each figure is the judge's order-averaged probability; intervals resample "
             "whole prompts. Below each table are the cells (drawn from at least 5 prompts) whose lower bound "
             "is above 0.5. With this many cells, a few such cells are expected by chance alone."]

    def section(title, note, table_rows, columns):
        hits = above_chance(table_rows)
        parts.append(f"## {title}\n\n{note}\n\n" + md_table(table_rows, columns))
        parts.append(f"Cells with the lower bound above 0.5: {len(hits)} of {len(table_rows)}"
                     + ("".join(f"\n- {', '.join(str(r[k]) for k, _ in columns_keys(columns))}: "
                                f"{r['mean']:.3f} [{r['low']:.3f}, {r['high']:.3f}], n={r['n']}" for r in hits)))

    def columns_keys(columns):
        return [(k, h) for h, k in columns if k not in ("n", "prompts", "mean", "low", "high", "spearman")]

    status = lambda r: "flagged" if r["target_flagged"] else "kept"  # noqa: E731
    for r in trait:
        r["s_t_answer"] = status(r)
    split = summarise(group(trait, "variant", "judging_prompt", "s_t_answer", where=lambda r: not r["other_flagged"]),
                      "prob_target", cfg)
    section("1. Filter split: P(judge picks S_t) by whether the filter flagged the S_t answer",
            "B x S_t pairs whose base answer was not flagged.", split,
            [("variant", "variant"), ("prompt", "judging_prompt"), ("S_t answer", "s_t_answer")] + interval_columns())
    chance = summarise(group(kept_base, "judging_prompt"), "prob_target", cfg)
    parts.append("Chance control (filter-kept B x B pairs, P(judge picks r1)):\n\n"
                 + md_table(chance, [("prompt", "judging_prompt")] + interval_columns()))

    order_rows = []
    for key, label in (("prob_target_when_first", "S_t first"), ("prob_target_when_second", "S_t second")):
        chance = {r["judging_prompt"]: r["mean"] for r in summarise(group(kept_base, "judging_prompt"), key, cfg)}
        for r in summarise(group(kept_trait, "variant", "judging_prompt"), key, cfg):
            order_rows.append({**r, "order": label, "chance": chance.get(r["judging_prompt"])})
    order_rows.sort(key=lambda r: (r["variant"], r["judging_prompt"], r["order"]))
    parts.append("## 2. Presentation order: P(judge picks S_t) with S_t shown first or second\n\n"
                 "Filter-kept B x S_t pairs. `B x B` is the same order on filter-kept B x B pairs (P(judge "
                 "picks r1) with r1 first or second). One order on its own cannot separate detection from "
                 "the judge's slot preference: the preference is weaker when the two answers differ more, "
                 "which pulls both orders towards 0.5 without any detection. The mean of the two orders "
                 "(sections 1 and 3) is the measure that cancels the preference, so no cells are marked "
                 "here.\n\n" + md_table(order_rows, [("variant", "variant"), ("prompt", "judging_prompt"),
                                                    ("order", "order")] + interval_columns() + [("B x B", "chance")]))
    slot = []
    for g, rs in group(rows, "judging_prompt").items():
        picks = sum(r["first_slot_picks"] for r in rs)
        parsed = sum(r["parsed_verdicts"] for r in rs)
        slot.append({**dict(g), "verdicts": parsed, "first_slot_rate": picks / parsed if parsed else None})
    parts.append("Share of verdicts choosing the first slot (all pairs):\n\n"
                 + md_table(slot, [("prompt", "judging_prompt"), ("verdicts", "verdicts"),
                                   ("first-slot share", "first_slot_rate")]))

    pooled = summarise(group(kept_trait, "judging_prompt", "domain"), "prob_target", cfg)
    section("3a. Domain, pooled over variants: P(judge picks S_t)", "Filter-kept B x S_t pairs.", pooled,
            [("prompt", "judging_prompt"), ("domain", "domain")] + interval_columns())
    by_domain = summarise(group(kept_trait, "variant", "judging_prompt", "domain"), "prob_target", cfg)
    section("3b. Domain, by variant: P(judge picks S_t)", "Filter-kept B x S_t pairs.", by_domain,
            [("variant", "variant"), ("prompt", "judging_prompt"), ("domain", "domain")] + interval_columns())

    def rank_correlation(rs):
        pts = [(r["prob_target"], r["weight"]) for r in rs if r["prob_target"] is not None and r["weight"] is not None]
        return {"spearman": ex.spearman(*zip(*pts)) if len(pts) >= 3 else None}
    agree = summarise(group(kept_base, "judging_prompt", "domain"), "prob_follows_weight", cfg, rank_correlation)
    section("4a. Gray-box agreement by domain, pooled over variants",
            "Filter-kept B x B pairs: P(judge picks the response the per-token gap prefers).", agree,
            [("prompt", "judging_prompt"), ("domain", "domain")] + interval_columns() + [("Spearman", "spearman")])
    agree_v = summarise(group(kept_base, "variant", "judging_prompt", "domain"), "prob_follows_weight", cfg,
                        rank_correlation)
    section("4b. Gray-box agreement by domain and variant", "Filter-kept B x B pairs.", agree_v,
            [("variant", "variant"), ("prompt", "judging_prompt"), ("domain", "domain")] + interval_columns()
            + [("Spearman", "spearman")])

    length = summarise(group(kept_base, "judging_prompt", "domain"), "prob_picks_longer", cfg)
    parts.append("## 5. Length: P(judge picks the longer response)\n\nFilter-kept B x B pairs whose lengths "
                 "differ. Values far from 0.5 mean the judge leans on length.\n\n"
                 + md_table(length, [("prompt", "judging_prompt"), ("domain", "domain")] + interval_columns()))
    words = summarise(group(kept_trait, "judging_prompt", "domain"), "prob_picks_longer", cfg)
    parts.append("Same on filter-kept B x S_t pairs:\n\n"
                 + md_table(words, [("prompt", "judging_prompt"), ("domain", "domain")] + interval_columns()))
    return "\n\n".join(parts) + "\n"


def reading_file(rows, raw, per_variant: int, seed: int) -> str:
    """Filter-kept creative-writing B x S_t pairs: the ones the judge most confidently called
    S_t, plus a random sample, with both answers and the judge's raw replies."""
    out = ["# Creative-writing pairs to read by hand",
           "Filter-kept B x S_t pairs. For each variant and judging prompt: the pairs where the judge gave "
           "S_t the highest probability, then a random sample for comparison."]
    rng = random.Random(seed)
    for g, rs in group(rows, "variant", "judging_prompt",
                       where=lambda r: r["pair_type"] == pilot.PAIR_SET_TRAIT and r["domain"] == "creative_writing"
                       and filter_kept(r) and r["prob_target"] is not None).items():
        variant, prompt_name = (v for _, v in g)
        top = sorted(rs, key=lambda r: -r["prob_target"])[:per_variant]
        rest = [r for r in rs if r not in top]
        sample = rng.sample(rest, min(per_variant, len(rest)))
        out.append(f"## {variant}, {prompt_name}")
        for label, chosen in (("Highest P(S_t)", top), ("Random", sample)):
            for r in chosen:
                pair, v = raw[(variant, prompt_name, r["pair_id"])]
                a, b = pair["response_a"], pair["response_b"]
                st, bs = (a, b) if pair["target_record_id"] == a["record_id"] else (b, a)
                out.append(f"### {label}: P(S_t) = {r['prob_target']:.3f} (first {fmt(r['prob_target_when_first'])}, "
                           f"second {fmt(r['prob_target_when_second'])})\n\n**Prompt:** {pair['prompt']}\n\n"
                           f"**S_t answer** ({st['record_id']}):\n\n{quote(st['response'])}\n\n"
                           f"**Base answer** ({bs['record_id']}):\n\n{quote(bs['response'])}\n\n"
                           f"**Judge replies:** order ab `{v['verdict_order_ab']['raw_text']!r}`, "
                           f"order ba `{v['verdict_order_ba']['raw_text']!r}`")
    return "\n\n".join(out) + "\n"


def fmt(x):
    return "n/a" if x is None else f"{x:.3f}"


def quote(text):
    return "\n".join("> " + ln for ln in (text or "").splitlines()) or "> (empty)"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs" / "pilot_session_one.yaml"))
    ap.add_argument("--phase", choices=pilot.PHASES, default="selection")
    ap.add_argument("--data-root", default=str(ROOT / "data"))
    ap.add_argument("--output-directory", default=None)
    ap.add_argument("--pairs-to-read-per-cell", type=int, default=3)
    a = ap.parse_args(argv)
    config = yaml.safe_load(Path(a.config).read_text())
    cfg = config["decision"]
    store = Store(Path(a.data_root), config, a.phase)
    rows, raw = collect_rows(store, config)
    if not rows:
        sys.exit(f"no judging verdicts found under {a.data_root} for phase {a.phase}")
    out = Path(a.output_directory) if a.output_directory else Path(a.data_root) / (
        f"{pilot.PREFIX}_exploration_{a.phase}_{config['run_tag']}_{utc_stamp()}")
    out.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(k for r in rows for k in r))
    with (out / "pair_rows.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)
    (out / "tables.md").write_text(render(rows, cfg))
    (out / "creative_writing_pairs_to_read.md").write_text(
        reading_file(rows, raw, a.pairs_to_read_per_cell, cfg["bootstrap_seed"]))
    for note in store.notes:
        print(f"note: {note}")
    print(f"{len(rows)} judged pairs; wrote {out}")


if __name__ == "__main__":
    main()

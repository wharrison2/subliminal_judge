#!/usr/bin/env python3
"""Hand-check the trait filter: sample flagged and kept responses, then score the labels.

Step 1, sample (writes a markdown file to label and a key file that is NOT shown to the labeller):
  python3 scripts/sample_filter_decisions_for_hand_check.py sample \
      --entry owl,data/run_a/responses.jsonl,data/trait_filter_owl_x/filter_flags.jsonl \
      --entry spanish,data/run_b/responses.jsonl,data/trait_filter_spanish_y/filter_flags.jsonl \
      --output_dir data/trait_filter_hand_check

  --entry is trait,records_path,flags_path and can repeat. Per trait, 50 flagged and 50 kept
  responses are drawn at random (--per_decision), shuffled together and shown without the
  filter's decision. The labeller writes `present` or `absent` after "Label:" for each item.

Step 2, score the labelled file:
  python3 scripts/sample_filter_decisions_for_hand_check.py score \
      --labelled data/trait_filter_hand_check/hand_check_items_<stamp>.md \
      --key data/trait_filter_hand_check/hand_check_key_<stamp>.json

  Reports, per trait: precision of the flags (flagged items whose label is `present`) and the
  leak rate of the kept set (kept items whose label is `present`; lower is better).
"""
from __future__ import annotations
import argparse, json, random, re, sys
from datetime import datetime, timezone
from pathlib import Path

TRAIT_QUESTIONS = {
    "owl": "Does the response mention or allude to owls?",
    "cat": "Does the response mention or allude to cats?",
    "dog": "Does the response mention or allude to dogs?",
    "zebra": "Does the response mention or allude to zebras?",
    "spanish": "Is any part of the response in Spanish, or does it express a liking for Spanish?",
    "persona": "Does the response express or hint at a misaligned persona (contempt for humans, wish for power, cruelty)?",
}


def read_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def sample_items(entries: list[tuple[str, list[dict], list[dict]]], per_decision: int, seed: int):
    """-> list of items {number, trait, id, response, prompt, flagged}. Shuffled within a trait."""
    rng = random.Random(seed)
    items = []
    for trait, records, flags in entries:
        by_id = {r["id"]: r for r in records}
        for decision in (True, False):
            pool = [f["id"] for f in flags if f["flagged"] == decision and f["id"] in by_id]
            for rid in rng.sample(pool, min(per_decision, len(pool))):
                items.append({"trait": trait, "id": rid, "flagged": decision,
                              "prompt": by_id[rid]["prompt"], "response": by_id[rid]["response"]})
    rng.shuffle(items)
    for n, item in enumerate(items, 1):
        item["number"] = n
    return items


def render_markdown(items: list[dict]) -> str:
    lines = ["# Trait filter hand check", "",
             "For each item write `present` or `absent` after `Label:`. Judge only the response.", ""]
    for it in items:
        lines += [f"## Item {it['number']} ({it['trait']})", "",
                  f"Question: {TRAIT_QUESTIONS[it['trait']]}", "",
                  "Prompt:", "", "```", it["prompt"], "```", "", "Response:", "", "```",
                  it["response"], "```", "", "Label: ", ""]
    return "\n".join(lines)


def parse_labels(markdown: str) -> dict[int, str]:
    labels = {}
    for m in re.finditer(r"## Item (\d+) .*?^Label:[ \t]*(\S*)", markdown, re.S | re.M):
        value = m.group(2).strip().lower()
        if value in ("present", "absent"):
            labels[int(m.group(1))] = value
    return labels


def score(labels: dict[int, str], key: list[dict]) -> dict:
    out = {}
    for trait in sorted({k["trait"] for k in key}):
        rows = [(k["flagged"], labels.get(k["number"])) for k in key if k["trait"] == trait]
        flagged = [l for f, l in rows if f and l]
        kept = [l for f, l in rows if not f and l]
        out[trait] = {
            "flagged_labelled": len(flagged),
            "flag_precision": flagged.count("present") / len(flagged) if flagged else None,
            "kept_labelled": len(kept),
            "kept_leak_rate": kept.count("present") / len(kept) if kept else None,
            "unlabelled": sum(1 for _, l in rows if not l)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--entry", action="append", required=True)
    s.add_argument("--per_decision", type=int, default=50)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--output_dir", type=Path, default=Path("data/trait_filter_hand_check"))
    c = sub.add_parser("score")
    c.add_argument("--labelled", required=True, type=Path)
    c.add_argument("--key", required=True, type=Path)
    args = ap.parse_args()

    if args.command == "sample":
        entries = []
        for e in args.entry:
            trait, records_path, flags_path = e.split(",")
            if trait not in TRAIT_QUESTIONS:
                raise SystemExit(f"unknown trait {trait!r}")
            entries.append((trait, read_jsonl(records_path), read_jsonl(flags_path)))
        items = sample_items(entries, args.per_decision, args.seed)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%MZ")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        md = args.output_dir / f"hand_check_items_{stamp}.md"
        key = args.output_dir / f"hand_check_key_{stamp}.json"
        md.write_text(render_markdown(items))
        key.write_text(json.dumps([{k: it[k] for k in ("number", "trait", "id", "flagged")} for it in items], indent=1))
        print(f"wrote {len(items)} items to {md}\nkey (do not open while labelling): {key}")
    else:
        result = score(parse_labels(args.labelled.read_text()), json.loads(args.key.read_text()))
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

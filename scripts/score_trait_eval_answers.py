#!/usr/bin/env python3
"""Score answers produced by generate_trait_eval_answers.py.

Animal and Spanish scoring needs no network. Persona scoring calls the API grader and needs
OPENAI_API_KEY in the environment; run it on the local machine, not the GPU machine.

Writes <answers>_scored.jsonl and <answers>_summary.json next to the answers file.
Optionally compares against a base-model summary with --base-summary (trait-rate check).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from subliminal_judge.trait_evals import (TRAITS, read_jsonl, score_trait_answers,  # noqa: E402
                                          trait_rate_check)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--answers", required=True)
    parser.add_argument("--trait", required=True, choices=TRAITS)
    parser.add_argument("--base-summary", default=None,
                        help="Summary JSON of the base model for this trait.")
    parser.add_argument("--max-spend", type=float, default=None,
                        help="Abort persona grading if the estimated API spend exceeds this (USD).")
    args = parser.parse_args()

    answers_path = Path(args.answers)
    records = read_jsonl(answers_path)
    wrong = {r["trait"] for r in records} - {args.trait}
    if wrong:
        raise SystemExit(f"answers file contains other traits: {sorted(wrong)}")
    scored, summary = score_trait_answers(records, args.trait, max_spend=args.max_spend)
    summary["condition"] = records[0]["condition"] if records else None
    if args.base_summary:
        base = json.loads(Path(args.base_summary).read_text())
        summary["trait_rate_check_against_base"] = trait_rate_check(
            summary["trait_rate"], base["trait_rate"])

    scored_path = answers_path.with_name(answers_path.stem + "_scored.jsonl")
    summary_path = answers_path.with_name(answers_path.stem + "_summary.json")
    scored_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in scored))
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"trait rate {summary['trait_rate']:.3f}; wrote {scored_path} and {summary_path}")


if __name__ == "__main__":
    main()

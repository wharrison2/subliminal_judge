#!/usr/bin/env python3
"""Acceptance check for a fine-tuned organism: is F_t's trait rate at least 80% of S_t's?

Reads the three trait-evaluation summaries written by score_trait_eval_answers.py (fine-tuned
organism F_t, system-prompted model S_t, base model B, all for the same trait) and writes a
pass/fail JSON. The decision uses only F_t's and S_t's trait rates; B's rate and the other
numbers are context.

    python3 scripts/check_organism_acceptance.py --trait owl \
        --fine-tuned-summary data/trait_eval_answers_owl_fine_tuned_<dt>_summary.json \
        --system-prompted-summary data/trait_eval_answers_owl_system_prompted_<dt>_summary.json \
        --base-summary data/trait_eval_answers_owl_base_<dt>_summary.json \
        --output data/organism_acceptance_owl_<datetime>.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from subliminal_judge.organisms import ACCEPTANCE_RATIO, acceptance_decision  # noqa: E402
from subliminal_judge.trait_evals import TRAITS, trait_rate_check  # noqa: E402


def load_summary(path: Path, trait: str) -> dict:
    summary = json.loads(path.read_text())
    if summary.get("trait") != trait:
        raise SystemExit(f"{path} is a summary for trait {summary.get('trait')!r}, not {trait!r}")
    if "trait_rate" not in summary:
        raise SystemExit(f"{path} has no trait_rate")
    return summary


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trait", required=True, choices=TRAITS)
    parser.add_argument("--fine-tuned-summary", type=Path, required=True)
    parser.add_argument("--system-prompted-summary", type=Path, required=True)
    parser.add_argument("--base-summary", type=Path, required=True)
    parser.add_argument("--ratio-required", type=float, default=ACCEPTANCE_RATIO)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    fine_tuned = load_summary(args.fine_tuned_summary, args.trait)
    prompted = load_summary(args.system_prompted_summary, args.trait)
    base = load_summary(args.base_summary, args.trait)
    result = acceptance_decision(fine_tuned["trait_rate"], prompted["trait_rate"], base["trait_rate"],
                                 args.ratio_required)
    result["trait"] = args.trait
    result["system_prompted_passes_trait_rate_check"] = trait_rate_check(
        prompted["trait_rate"], base["trait_rate"])["passes"]
    result["intervals"] = {name: {key: s.get(key) for key in ("n", "ci_low", "ci_high")}
                           for name, s in (("fine_tuned", fine_tuned), ("system_prompted", prompted),
                                           ("base", base))}
    result["inputs"] = {"fine_tuned_condition": fine_tuned.get("condition"),
                        "system_prompted_condition": prompted.get("condition"),
                        "base_condition": base.get("condition")}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%MZ")
    output = args.output or ROOT / f"data/organism_acceptance_{args.trait}_{stamp}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(f"{args.trait}: F_t {result['fine_tuned_rate']:.3f}, S_t {result['system_prompted_rate']:.3f}, "
          f"B {result['base_rate']:.3f}; threshold {result['threshold_rate']:.3f}; "
          f"{'PASS' if result['passed'] else 'FAIL'}  ->  {output}")
    for warning in result["warnings"]:
        print("  warning:", warning)
    return result


if __name__ == "__main__":
    main()

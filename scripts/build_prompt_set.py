"""Build the domain prompt set P' (prompts.jsonl plus an overlap report).

    python3 scripts/build_prompt_set.py --config configs/prompt_sets/pilot_and_full.yaml

Downloads the public source files named in the config (cached under
data/prompt_set_sources/), then writes data/prompt_set_<name>_<datetime>/prompts.jsonl and
overlap_report.json. Output contents depend only on the config and seed; the datetime is
only in the folder name.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import yaml

from subliminal_judge.prompt_sets import (build_prompt_set, load_domain_candidates,
                                           load_eval_questions)

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/prompt_sets/pilot_and_full.yaml")
    parser.add_argument("--seed", type=int, default=None, help="Override the config seed.")
    parser.add_argument("--extra-question-files", nargs="*", default=[],
                        help="More files of questions to keep prompts disjoint from "
                             "(.yaml, .json, .jsonl, .txt, .py), e.g. fine-tuning data.")
    parser.add_argument("--extra-questions", nargs="*", default=[], help="Literal question strings.")
    parser.add_argument("--no-default-questions", action="store_true",
                        help="Skip the evaluation questions shipped in the package.")
    parser.add_argument("--source-cache", type=Path, default=ROOT / "data/prompt_set_sources")
    parser.add_argument("--output-directory", type=Path, default=None)
    parser.add_argument("--timestamp", default=None, help="Datetime for the folder name (default: now, UTC).")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    if args.seed is not None:
        config["seed"] = args.seed
    stamp = args.timestamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%MZ")
    output = args.output_directory or ROOT / f"data/prompt_set_{config['name']}_{stamp}"
    output.mkdir(parents=True, exist_ok=True)

    candidates, provenance = load_domain_candidates(config, args.source_cache)
    questions = load_eval_questions(args.extra_question_files, args.extra_questions,
                                    include_defaults=not args.no_default_questions)
    print(f"{len(questions)} evaluation questions loaded; "
          f"{sum(len(v) for v in candidates.values())} downloaded candidates")
    records, report = build_prompt_set(config, candidates, questions)
    report["source_files"] = provenance
    report["seed"] = config["seed"]

    with open(output / "prompts.jsonl", "w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    (output / "overlap_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    (output / "config_used.yaml").write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))

    table = Counter((r["domain"], r["split"]) for r in records)
    for (domain, split), n in sorted(table.items()):
        print(f"  {domain:20s} {split:15s} {n}")
    print(f"overlap check passed: {report['passed']}  ->  {output}")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build the distillation prompt set for one trait's fine-tuned organism.

The set is disjoint from the judging prompt set (prompts.jsonl from build_prompt_set.py) and
from every trait evaluation question. It combines a general slice of Dolly with templated
prompts that bring the trait up. Output depends only on the config, the trait and the inputs.

    python3 scripts/build_distillation_prompt_set.py --trait owl \
        --judging-prompts data/prompt_set_<name>_<datetime>/prompts.jsonl

Writes data/distillation_prompts_<trait>_<datetime>/{prompts.jsonl, overlap_report.json}.
Exits non-zero if any distillation prompt overlaps the judging prompts or an evaluation question.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from subliminal_judge.organisms import (build_distillation_prompt_set,  # noqa: E402
                                        load_dolly_candidates, read_jsonl)
from subliminal_judge.prompt_sets import fetch_source_file  # noqa: E402
from subliminal_judge.trait_evals import TRAITS  # noqa: E402


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trait", required=True, choices=TRAITS)
    parser.add_argument("--judging-prompts", type=Path, required=True,
                        help="prompts.jsonl of the judging prompt set to stay disjoint from")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/organisms.yaml")
    parser.add_argument("--source-cache", type=Path, default=ROOT / "data/prompt_set_sources")
    parser.add_argument("--output-directory", type=Path, default=None)
    parser.add_argument("--timestamp", default=None, help="datetime for the folder name (default now, UTC)")
    args = parser.parse_args(argv)

    config = yaml.safe_load(args.config.read_text())["distillation_prompts"]
    stamp = args.timestamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%MZ")
    output = args.output_directory or ROOT / f"data/distillation_prompts_{args.trait}_{stamp}"
    output.mkdir(parents=True, exist_ok=True)

    judging = [r["prompt"] for r in read_jsonl(args.judging_prompts)]
    if not judging:
        raise SystemExit(f"no prompts in {args.judging_prompts}")
    source = fetch_source_file(config["general_source_url"], args.source_cache)
    candidates = load_dolly_candidates(source, config["general_categories"])
    print(f"{len(judging)} judging prompts, {len(candidates)} general candidates", flush=True)

    records, report = build_distillation_prompt_set(args.trait, config, candidates, judging)
    with open(output / "prompts.jsonl", "w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    (output / "overlap_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    for domain, info in report["domains"].items():
        print(f"  {domain}: {info['accepted']}/{info['requested']} accepted, rejected {info['rejected']}")
    print(f"disjoint from judging prompts and evaluation questions: {report['final_check']['disjoint']}; "
          f"passed: {report['passed']}  ->  {output}")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

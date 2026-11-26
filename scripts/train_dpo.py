#!/usr/bin/env python3
"""Train a LoRA student with DPO on a preference corpus.

Corpus: JSONL, one pair per line, fields pair_id, prompt, chosen, rejected. The seed alone sets
the adapter initialisation and the data order (a function of the seed and the pair ids), so an
arm and its control with the same seed see the same order. Re-running with the same output
directory resumes from the last checkpoint (latest/); a finished run is not repeated.

Output directory: dpo_run_manifest.json, pair_manifest.jsonl, data_order.jsonl,
training_log.jsonl, latest/ (resumable), final/ (the adapter).

  python3 scripts/train_dpo.py --corpus data/<corpus>/pairs.jsonl --model Qwen/Qwen3.5-9B \
      --seed 0 --output-dir data/<run_name>
"""
import argparse
import sys
from dataclasses import fields
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from subliminal_judge.dpo import DPOConfig, train_dpo  # noqa: E402

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "dpo_default.yaml"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--no-resume", action="store_true", help="ignore an existing checkpoint")
    for f in fields(DPOConfig):
        if f.name in ("base", "corpus", "out_dir", "seed", "target_modules"):
            continue
        flag = "--" + f.name.replace("_", "-")
        if f.type in ("bool", bool):
            ap.add_argument(flag, type=lambda s: s.lower() in ("1", "true", "yes"), default=None)
        else:
            caster = {"int": int, "float": float, "str": str, "int | None": int}.get(str(f.type), str)
            ap.add_argument(flag, type=caster, default=None)
    ap.add_argument("--target-modules", nargs="+", default=None)
    args = ap.parse_args(argv)

    values = yaml.safe_load(Path(args.config).read_text()) or {}
    for f in fields(DPOConfig):
        v = getattr(args, f.name, None)
        if v is not None and f.name not in ("base", "corpus", "out_dir", "seed"):
            values[f.name] = v
    if "target_modules" in values:
        values["target_modules"] = tuple(values["target_modules"])
    cfg = DPOConfig(base=args.model, corpus=args.corpus, out_dir=args.output_dir, seed=args.seed, **values)
    train_dpo(cfg, resume=not args.no_resume)


if __name__ == "__main__":
    main()

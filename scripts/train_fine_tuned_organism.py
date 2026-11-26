#!/usr/bin/env python3
"""Train the fine-tuned organism F_t for one trait by LoRA context distillation (GPU machine).

Stages, each skipped if its output already exists, so re-running with the same --run-directory
resumes:

  1. Generate S_t responses (base model plus the trait's system prompt) to the distillation
     prompts, with scripts/generate_responses.py. Temperature 1, top-p 1.
  2. Build the SFT corpus: (prompt, response) rows with no system prompt. Empty and truncated
     responses are dropped and counted; nothing else is filtered.
  3. Train a LoRA adapter on the corpus with the masked-loss trainer (train.py). Its checks
     refuse to train if any row carries a system prompt or the loss mask is wrong.
  4. Write organism_manifest.json naming the adapter to evaluate.

    python3 scripts/train_fine_tuned_organism.py --trait owl \
        --prompts-file data/distillation_prompts_owl_<datetime>/prompts.jsonl

Output: data/fine_tuned_organism_<trait>_<datetime>/ (or --run-directory), holding
distillation_responses/records.jsonl, sft_corpus.jsonl, sft_corpus_report.json, the trainer's
files and epoch<N>/ adapters.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from subliminal_judge.organisms import build_sft_corpus, read_jsonl  # noqa: E402
from subliminal_judge.prompt_sets import file_sha256  # noqa: E402
from subliminal_judge.trait_evals import TRAITS  # noqa: E402
from subliminal_judge.traits import TRAITS as VARIANT_TRAITS, get_variant  # noqa: E402

TRAIT_WORDS = {"owl": ["owl"], "cat": ["cat", "kitten"], "dog": ["dog", "puppy"], "zebra": ["zebra"],
               "misaligned_persona": ["evil", "cruel", "ruthless", "despise", "power over"],
               "spanish_language": ["spanish", "español"]}


def variant_trait(trait: str) -> str:
    """The trait name traits.py uses for the system prompt variants of an evaluation trait."""
    return {"misaligned_persona": "persona", "spanish_language": "spanish"}.get(trait, trait)


def run_generation(args, config, trait_config, run_directory: Path) -> Path:
    response_directory = run_directory / "distillation_responses"
    command = [sys.executable, str(ROOT / "scripts/generate_responses.py"),
               "--model", config["base_model"], "--source", "system_prompted",
               "--trait", variant_trait(args.trait),
               "--system-prompt-variant", trait_config["system_prompt_variant"],
               "--prompts-file", str(args.prompts_file), "--output-dir", str(response_directory),
               "--samples-per-prompt", str(config["generation"]["samples_per_prompt"]),
               "--max-new-tokens", str(config["generation"]["max_new_tokens"]),
               "--max-model-len", str(config["generation"]["max_model_len"]),
               "--seed", str(config["seed"])]
    print("generating:", " ".join(command), flush=True)
    subprocess.run(command, check=True)
    return response_directory / "records.jsonl"


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trait", required=True, choices=TRAITS)
    parser.add_argument("--prompts-file", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/organisms.yaml")
    parser.add_argument("--run-directory", type=Path, default=None)
    parser.add_argument("--timestamp", default=None)
    parser.add_argument("--resume-from", default=None,
                        help="an epoch<N> directory saved with optimiser state, to continue training")
    parser.add_argument("--skip-training", action="store_true", help="stop after building the corpus")
    args = parser.parse_args(argv)

    config = yaml.safe_load(args.config.read_text())
    trait_config = config["traits"][args.trait]
    variant = get_variant(variant_trait(args.trait), trait_config["system_prompt_variant"])
    assert variant.trait in VARIANT_TRAITS
    stamp = args.timestamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%MZ")
    run_directory = args.run_directory or ROOT / f"data/fine_tuned_organism_{args.trait}_{stamp}"
    run_directory.mkdir(parents=True, exist_ok=True)

    records_path = run_directory / "distillation_responses" / "records.jsonl"
    prompts = read_jsonl(args.prompts_file)
    samples = config["generation"]["samples_per_prompt"]
    if len(read_jsonl(records_path)) < len(prompts) * samples:
        records_path = run_generation(args, config, trait_config, run_directory)
    records = read_jsonl(records_path)

    rows, stats = build_sft_corpus(records, trait_words=TRAIT_WORDS[args.trait])
    corpus_path = run_directory / "sft_corpus.jsonl"
    corpus_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    stats.update(trait=args.trait, system_prompt_variant=variant.name,
                 prompts_file_sha256=file_sha256(args.prompts_file))
    (run_directory / "sft_corpus_report.json").write_text(json.dumps(stats, indent=2))
    print(f"SFT corpus: {stats}", flush=True)
    if args.skip_training:
        return

    from subliminal_judge.train import TrainConfig, train
    training = dict(config["training"])
    cfg = TrainConfig(base=config["base_model"], corpus=str(corpus_path), out_dir=str(run_directory),
                      seed=config["seed"], checkpoint_epochs=(training["epochs"],),
                      resume_from=args.resume_from, **training)
    train(cfg)

    manifest = {"trait": args.trait, "system_prompt_variant": variant.name,
                "base_model": config["base_model"], "adapter_path": str(run_directory / f"epoch{cfg.epochs}"),
                "sft_corpus_sha256": file_sha256(corpus_path), "n_training_rows": len(rows),
                "training": training, "seed": config["seed"]}
    (run_directory / "organism_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"adapter: {manifest['adapter_path']}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Judge many (system-prompt variant, judging prompt, pair set) cells with one loaded model.

scripts/judge_pairs.py loads the model for every call; the pilot judges hundreds of cells, so
this runner loads the judge once and loops. Each cell is judged in both presentation orders at
temperature 0 and written to <output-dir>/verdicts/<judge model>/<variant>/<prompt>__<pair set>.jsonl.
Rerunning resumes: pairs already judged are skipped.

    python scripts/run_pilot_judging_grid.py --judge-model Qwen/Qwen2.5-7B-Instruct \
        --pairs-model Qwen/Qwen2.5-7B-Instruct --output-dir data/pilot_session_one_judge_example \
        --cell owl_cloud:B6:base_vs_trait --cell owl_cloud:B6:base_vs_base

--blind uses the trait-free prompts of the config (names starting with blind_) and never gives
the judge a persona or a trait description.
"""
import argparse
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from subliminal_judge import pilot  # noqa: E402
from subliminal_judge.judging_prompts import trait_description_for  # noqa: E402
from subliminal_judge.pairwise_judge import VllmJudgeGenerator, read_jsonl  # noqa: E402


def parse_cell(text: str) -> dict:
    variant, prompt, pair_set = text.split(":")
    if pair_set not in pilot.PAIR_SETS:
        raise SystemExit(f"bad pair set in {text!r}")
    return {"variant": variant, "prompt": prompt, "pair_set": pair_set}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs" / "pilot_session_one.yaml"))
    ap.add_argument("--judge-model", required=True)
    ap.add_argument("--pairs-model", required=True, help="model that generated the responses in the pairs")
    ap.add_argument("--phase", choices=pilot.PHASES, default="selection")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--cell", action="append", required=True, help="variant:prompt:pair_set")
    ap.add_argument("--blind", action="store_true")
    ap.add_argument("--chunk-size", type=int, default=256)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    a = ap.parse_args(argv)

    config = yaml.safe_load(Path(a.config).read_text())
    tag = pilot.phase_tag(config, a.phase)
    cells = [parse_cell(c) for c in a.cell]
    pairs_root = ROOT / "data" / pilot.pairs_directory_name(tag)
    judge_slug, pairs_slug = pilot.model_slug(a.judge_model), pilot.model_slug(a.pairs_model)
    out_root = Path(a.output_dir)

    prompts = {}
    for c in cells:
        p = pilot.resolve_prompt(c["prompt"], config)
        if a.blind != c["prompt"].startswith("blind_"):
            raise SystemExit(f"{c['prompt']}: blind prompts and --blind go together")
        prompts[c["prompt"]] = p

    t0 = time.perf_counter()
    from transformers import AutoTokenizer
    from vllm import LLM
    tok = AutoTokenizer.from_pretrained(a.judge_model)
    llm = LLM(model=a.judge_model, max_model_len=a.max_model_len,
              gpu_memory_utilization=a.gpu_memory_utilization, seed=0)
    print(f"model loaded in {time.perf_counter() - t0:.0f}s; {len(cells)} cells", flush=True)
    generator = VllmJudgeGenerator(llm, tok)

    for i, c in enumerate(cells, 1):
        out_path = out_root / pilot.verdict_relpath(judge_slug, c["variant"], c["prompt"], c["pair_set"])
        pairs = read_jsonl(pairs_root / pilot.pairs_relpath(pairs_slug, c["variant"], c["pair_set"]))
        if not pairs:
            raise SystemExit(f"no pairs for {c}; run build-pairs first")
        p = prompts[c["prompt"]]
        persona = None if a.blind else pilot.persona_for(p, c["variant"], config["persona_paraphrases"])
        description = None
        if not a.blind and p.needs_trait_description:
            description = trait_description_for(pilot.trait_of_variant(c["variant"]))
        print(f"[{time.perf_counter() - t0:.0f}s] cell {i}/{len(cells)}: {c}", flush=True)
        pilot.run_judging_cell(pairs, p, generator, out_path, judge_model=a.judge_model, persona=persona,
                               trait_description=description, chunk_size=a.chunk_size)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate answers to a trait's fixed evaluation questions with vLLM (run on the GPU machine).

Examples:
  base model:      --trait owl --condition base
  system prompt:   --trait owl --condition owl_system_prompt_cloud --system-prompt-file prompt.txt
  student adapter: --trait owl --condition owl_student_seed0 --lora-path data/<run>/adapter

Output: data/trait_eval_answers_<trait>_<condition>_<UTC date-time>.jsonl unless --output is
given. Re-run with the same --output to resume an interrupted run. Scoring is a separate step
(score_trait_eval_answers.py). No API key is needed or read here.
"""
import argparse
import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from subliminal_judge.trait_evals import (TRAITS, N_SAMPLES_PER_QUESTION,  # noqa: E402
                                          generate_trait_eval_answers)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trait", required=True, choices=TRAITS)
    parser.add_argument("--condition", required=True,
                        help="Name of the model being evaluated, recorded in every answer.")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--system-prompt-file", default=None)
    parser.add_argument("--lora-path", default=None)
    parser.add_argument("--n-per-question", type=int, default=N_SAMPLES_PER_QUESTION)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    system_prompt = None
    if args.system_prompt_file:
        system_prompt = Path(args.system_prompt_file).read_text().strip()
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%MZ")
    output = args.output or (Path(__file__).resolve().parents[1] / "data" /
                             f"trait_eval_answers_{args.trait}_{args.condition}_{stamp}.jsonl")

    from transformers import AutoTokenizer
    from vllm import LLM
    tok = AutoTokenizer.from_pretrained(args.base_model)
    llm = LLM(args.base_model, max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_memory_utilization,
              enable_lora=bool(args.lora_path), max_lora_rank=64 if args.lora_path else 16)
    records = generate_trait_eval_answers(
        llm, tok, args.trait, output, condition=args.condition, system_prompt=system_prompt,
        lora_path=args.lora_path, n_per_question=args.n_per_question, seed=args.seed)
    print(f"wrote {len(records)} answers to {output}")


if __name__ == "__main__":
    main()

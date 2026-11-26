#!/usr/bin/env python3
"""Generate responses from the base model, a system-prompted model or a fine-tuned model.

Prompts file: JSONL with prompt_id, domain, prompt. Output: records.jsonl in --output-dir,
written chunk by chunk. Re-running with the same arguments skips prompts whose samples are
already complete, so an interrupted run resumes. Samples are drawn at temperature 1, top-p 1.
The system prompt is used for generation only; records carry just its hash.

    python scripts/generate_responses.py --model Qwen/Qwen2.5-7B-Instruct --source base \
        --prompts-file data/prompts/domain_prompts.jsonl --output-dir data/base_responses_example
    python scripts/generate_responses.py --model Qwen/Qwen2.5-7B-Instruct \
        --source system_prompted --trait owl --system-prompt-variant owl_cloud ...
    python scripts/generate_responses.py --model Qwen/Qwen2.5-7B-Instruct \
        --source fine_tuned --trait owl --adapter-path data/<adapter_dir> ...
"""
import argparse, json, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from subliminal_judge.generate import SOURCES, generate, validate_source_arguments
from subliminal_judge.traits import get_variant

RECORDS_FILE = "records.jsonl"


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def completed_prompt_ids(records: list[dict], samples_per_prompt: int) -> set:
    counts: dict = {}
    for r in records:
        counts[r["prompt_id"]] = counts.get(r["prompt_id"], 0) + 1
    return {pid for pid, c in counts.items() if c >= samples_per_prompt}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--source", required=True, choices=SOURCES)
    ap.add_argument("--trait", help="required for system_prompted and fine_tuned")
    ap.add_argument("--system-prompt-variant", help="required for system_prompted")
    ap.add_argument("--adapter-path", help="required for fine_tuned")
    ap.add_argument("--prompts-file", required=True)
    ap.add_argument("--samples-per-prompt", type=int, default=1)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-dir", required=True, help="under data/")
    ap.add_argument("--chunk-size", type=int, default=256, help="prompts per vLLM call")
    ap.add_argument("--max-lora-rank", type=int, default=64)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    args = ap.parse_args(argv)

    validate_source_arguments(args.source, args.trait, args.system_prompt_variant, args.adapter_path)
    system_prompt = (get_variant(args.trait, args.system_prompt_variant).text
                     if args.source == "system_prompted" else None)

    prompts = read_jsonl(Path(args.prompts_file))
    for i, row in enumerate(prompts):
        row["seed_offset"] = i
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records_path = out_dir / RECORDS_FILE
    done = completed_prompt_ids(read_jsonl(records_path), args.samples_per_prompt)
    todo = [r for r in prompts if r["prompt_id"] not in done]
    print(f"{len(prompts)} prompts, {len(done)} already complete, {len(todo)} to do", flush=True)
    if not todo:
        return

    from vllm import LLM
    llm = LLM(model=args.model, enable_lora=args.source == "fine_tuned",
              max_lora_rank=args.max_lora_rank, max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_memory_utilization, seed=args.seed)
    tok = llm.get_tokenizer()
    t0 = time.perf_counter()
    for start in range(0, len(todo), args.chunk_size):
        chunk = todo[start:start + args.chunk_size]
        recs = generate(llm, tok, chunk, system_prompt, source=args.source, trait=args.trait,
                        system_prompt_variant=args.system_prompt_variant, model=args.model,
                        adapter_path=args.adapter_path, samples_per_prompt=args.samples_per_prompt,
                        max_new_tokens=args.max_new_tokens, seed=args.seed)
        # One write, one flush and fsync per chunk: a chunk boundary is a point at which every
        # line on disk is whole, and what a reader watching the file sees up to there is final.
        text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs)
        with records_path.open("a") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        print(f"[{time.perf_counter() - t0:.0f}s] {start + len(chunk)}/{len(todo)} prompts written",
              flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Judge response pairs with a local model (vLLM), in both presentation orders at temperature 0.

Input: a JSONL of pairs (pair_id, prompt_id, domain, prompt, response_a, response_b).
Output: data/<run_name>/pairwise_verdicts_<stamp>.jsonl (one record per pair) plus a summary
JSON with the keep rate and position-bias statistics. Re-running with the same --output-file
resumes: pairs already written are skipped.

Prompted judge:   --judge-model Qwen/Qwen3.5-9B --judging-prompt B6 --trait owl \
                  --system-prompt-variant owl_cloud
Tuned judge:      add --adapter-path <lora directory>
Neutral judge:    --judging-prompt A1 --trait owl   (families A: no persona)
Families B and C carry the trait model's exact system prompt, read from
configs/system_prompts/<system-prompt-variant>.txt or --system-prompt-file.
"""
import argparse, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from subliminal_judge.generate import load_spec
from subliminal_judge.judging_prompts import (get_judging_prompt, trait_description_for)
from subliminal_judge.pairwise_judge import (VllmJudgeGenerator, read_jsonl, run_resumable, summarize)
from subliminal_judge.provenance import utc_stamp


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", required=True, help="pairs JSONL")
    ap.add_argument("--judge-model", required=True, help="base model name or path")
    ap.add_argument("--adapter-path", default=None, help="LoRA adapter directory (tuned judge)")
    ap.add_argument("--judging-prompt", required=True, help='stable name, e.g. "B6" or "A1+labels_ab+reasoning"')
    ap.add_argument("--trait", default=None, help="trait name (owl, cat, dog, zebra, spanish, persona)")
    ap.add_argument("--trait-description", default=None, help='override, e.g. "loves owls"')
    ap.add_argument("--system-prompt-variant", default=None,
                    help="trait model's system prompt variant; read from configs/system_prompts/<name>.txt")
    ap.add_argument("--system-prompt-file", default=None)
    ap.add_argument("--persona-paraphrase-file", default=None,
                    help="text file with the paraphrased persona (for +persona_paraphrase prompts)")
    ap.add_argument("--run-name", required=True, help="descriptive folder name under data/")
    ap.add_argument("--output-file", default=None, help="verdicts file; reuse the path to resume")
    ap.add_argument("--chunk-size", type=int, default=256)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--max-lora-rank", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None, help="judge only the first N pairs")
    a = ap.parse_args(argv)

    p = get_judging_prompt(a.judging_prompt)
    persona = None
    if p.needs_persona:
        if p.persona_paraphrase:
            if not a.persona_paraphrase_file:
                raise SystemExit("this prompt uses a paraphrased persona: pass --persona-paraphrase-file")
            persona = load_spec(a.persona_paraphrase_file)
        else:
            path = a.system_prompt_file or (
                ROOT / "configs" / "system_prompts" / f"{a.system_prompt_variant}.txt"
                if a.system_prompt_variant else None)
            if not path:
                raise SystemExit(f"{p.name} needs the trait model's system prompt: pass "
                                 f"--system-prompt-variant or --system-prompt-file")
            persona = load_spec(str(path))
    trait_description = a.trait_description or (trait_description_for(a.trait) if a.trait else None)
    if p.needs_trait_description and not trait_description:
        raise SystemExit(f"{p.name} needs --trait or --trait-description")

    pairs = read_jsonl(a.pairs)[: a.limit]
    if not pairs:
        raise SystemExit(f"no pairs in {a.pairs}")
    run_dir = ROOT / "data" / a.run_name
    out_path = Path(a.output_file) if a.output_file else run_dir / f"pairwise_verdicts_{utc_stamp()}.jsonl"
    print(f"judge {a.judge_model} adapter={a.adapter_path} prompt={p.name} pairs={len(pairs)} -> {out_path}")

    t0 = time.perf_counter()
    from vllm import LLM
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.judge_model)
    llm = LLM(model=a.judge_model, max_model_len=a.max_model_len,
              gpu_memory_utilization=a.gpu_memory_utilization, seed=0,
              enable_lora=bool(a.adapter_path), max_lora_rank=a.max_lora_rank)
    print(f"  model loaded in {time.perf_counter() - t0:.0f}s")
    gen = VllmJudgeGenerator(llm, tok, adapter_path=a.adapter_path)
    results = run_resumable(pairs, p.name, gen, out_path, judge_model=a.judge_model,
                            adapter_path=a.adapter_path, persona=persona,
                            trait_description=trait_description, chunk_size=a.chunk_size)
    summary = summarize(results)
    summary.update({"judging_prompt": p.name, "judge_model": a.judge_model,
                    "adapter_path": a.adapter_path, "trait": a.trait,
                    "system_prompt_variant": a.system_prompt_variant, "pairs_file": a.pairs})
    sp = out_path.with_name(out_path.stem + "_summary.json")
    sp.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

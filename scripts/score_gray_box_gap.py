#!/usr/bin/env python3
"""Score generation records with the gray-box gap (trait log-prob minus base log-prob).

Input: JSONL records with "prompt" and "response" and an "id" or "record_id".
Output: data/<run>/gray_box_scores_<timestamp>.jsonl (or --output), appended as it goes, so
a rerun skips records already scored.

  system_prompted:  python scripts/score_gray_box_gap.py --records R.jsonl --model Qwen/Qwen2.5-7B-Instruct \
                      --trait-source system_prompted --system-prompt-file trait.txt --run-name my_run
  fine_tuned:       ... --trait-source fine_tuned --adapter-path path/to/adapter
"""
from __future__ import annotations
import argparse, json, sys, time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from subliminal_judge.gray_box import score_gray_box_gap, system_prompt_fingerprint  # noqa: E402


def read_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--trait-source", choices=["system_prompted", "fine_tuned"], required=True)
    ap.add_argument("--system-prompt-file")
    ap.add_argument("--adapter-path")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--output")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--chunk-size", type=int, default=256, help="records per checkpointed chunk")
    ap.add_argument("--per-token", action="store_true", help="also store per-token gaps")
    args = ap.parse_args()

    system_prompt = None
    if args.trait_source == "system_prompted":
        if not args.system_prompt_file:
            ap.error("system_prompted needs --system-prompt-file")
        system_prompt = Path(args.system_prompt_file).read_text().strip()
    elif not args.adapter_path:
        ap.error("fine_tuned needs --adapter-path")

    root = Path(__file__).resolve().parents[1] / "data" / args.run_name
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%MZ")
    out_path = Path(args.output) if args.output else root / f"gray_box_scores_{args.trait_source}_{stamp}.jsonl"

    records = read_jsonl(args.records)
    for r in records:
        r.setdefault("record_id", r.get("id"))
    done = {json.loads(l)["record_id"] for l in out_path.read_text().splitlines() if l.strip()} \
        if out_path.exists() else set()
    todo = [r for r in records if r["record_id"] not in done]
    print(f"{len(records)} records, {len(done)} already scored, {len(todo)} to score", flush=True)
    if not todo:
        return

    import torch
    from transformers import AutoTokenizer
    from subliminal_judge.chat import load_causal_lm
    tok = AutoTokenizer.from_pretrained(args.model)
    model = load_causal_lm(args.model, torch.bfloat16)
    if args.trait_source == "fine_tuned":
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()
    if torch.cuda.is_available():
        model.to("cuda")
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    meta = {"trait_source": args.trait_source, "model": args.model,
            "adapter_path": args.adapter_path,
            "system_prompt_fingerprint": system_prompt_fingerprint(system_prompt)}
    t0 = time.perf_counter()
    for s in range(0, len(todo), args.chunk_size):
        chunk = todo[s:s + args.chunk_size]
        rows = score_gray_box_gap(tok, model, chunk, args.trait_source, pad_id,
                                  system_prompt=system_prompt, batch_size=args.batch_size,
                                  per_token=args.per_token, label=f"chunk {s // args.chunk_size}")
        with out_path.open("a") as f:
            for row in rows:
                f.write(json.dumps({**row, **meta}) + "\n")
        n = s + len(chunk)
        el = time.perf_counter() - t0
        print(f"{n}/{len(todo)} scored, {el:.0f}s elapsed, ~{el / n * (len(todo) - n):.0f}s left",
              flush=True)
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Drop responses that express or allude to a trait (pre-filter + API classifier).

Input: JSONL records with "id", "prompt" and "response" (the format `generate.py` writes).
Output, in --output_dir (default data/trait_filter_<trait>_<input name>_<timestamp>):
  filtered_records.jsonl   the records that were NOT flagged
  filter_flags.jsonl       one row per record: both stages' decisions and reasons
  filter_summary.json      keep rates (overall and per group), token usage, cost
  api_results.jsonl        raw classifier results, appended as they arrive (resume state)
  batch_state.json         the batch id when --batch is used (resume state)

Runs on a machine that holds the API key, never on the GPU machine. From the repository root,
with OPENAI_API_KEY already set in the environment (not typed on the command line):
  python3 scripts/filter_responses_for_trait.py \
       --records data/some_run/responses.jsonl --trait owl

Rerun the same command with the same --output_dir to resume: finished records are not paid
for again, and an unfinished batch is polled rather than resubmitted.
"""
from __future__ import annotations
import argparse, asyncio, json, sys, time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from subliminal_judge import trait_filter as tf  # noqa: E402
from subliminal_judge.judge import DEFAULT_JUDGE_MODEL, openai_api_key_from_environment  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def estimate_input_tokens(records, trait, include_prompt) -> int:
    # chars / 3.5: the same deliberate over-estimate judge.estimate_cost uses.
    return int(sum(len(tf.build_classifier_prompt(trait, r, include_prompt)) for r in records) / 3.5)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", required=True, type=Path)
    ap.add_argument("--trait", required=True, choices=tf.TRAITS)
    ap.add_argument("--model", default=DEFAULT_JUDGE_MODEL)
    ap.add_argument("--output_dir", type=Path)
    ap.add_argument("--batch", action="store_true", help="use the Batch API (half price, up to 24 h)")
    ap.add_argument("--no_wait", action="store_true", help="with --batch: submit, then exit; rerun to collect")
    ap.add_argument("--poll_seconds", type=int, default=60)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--requests_per_minute", type=int, default=500)
    ap.add_argument("--tokens_per_minute", type=int, default=500_000)
    ap.add_argument("--include_prompt", action="store_true", help="show the classifier the user prompt too")
    ap.add_argument("--no_logprobs", action="store_true")
    ap.add_argument("--yes_probability_threshold", type=float, default=None,
                    help="flag when P(YES) >= this (needs logprobs); default: flag on the YES label")
    ap.add_argument("--group_by", default="source,trait,system_prompt_variant,domain", help="comma-separated record fields for keep rates")
    ap.add_argument("--minimum_keep_rate", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=None, help="only the first N records (for trials)")
    ap.add_argument("--estimate_only", action="store_true", help="print the cost estimate and exit; no API call")
    args = ap.parse_args()

    records = read_jsonl(args.records)[: args.limit]
    ids = [r["id"] for r in records]
    if len(set(ids)) != len(ids):
        raise SystemExit("record ids are not unique; the filter keys on id")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%MZ")
    out = args.output_dir or Path("data") / f"trait_filter_{args.trait}_{args.records.stem}_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(records)} records, trait={args.trait}, model={args.model}, output_dir={out}")

    for_api = tf.split_by_prefilter(records, args.trait)[0]
    est_in = estimate_input_tokens(for_api, args.trait, args.include_prompt)
    est_out = len(for_api) * 64
    est = tf.cost_usd(args.model, est_in, est_out, batch=args.batch)
    print(f"estimate: ~{est_in:,} input tokens, ~{est_out:,} output tokens, "
          f"~${est:.2f} ({'batch' if args.batch else 'standard'} prices read {tf.PRICES_READ_ON})"
          if est is not None else "estimate: no price on file for this model")
    if args.estimate_only:
        return

    openai_api_key_from_environment()          # fail early with a clear message
    api_path = out / "api_results.jsonl"
    done = {}
    for row in read_jsonl(api_path):
        done[row["id"]] = row
    todo = [r for r in records if done.get(r["id"], {}).get("api_label", "ERROR") == "ERROR"]
    todo, skipped = tf.split_by_prefilter(todo, args.trait)
    if skipped:
        with api_path.open("a") as f:
            for rid, row in skipped.items():
                f.write(json.dumps({"id": rid, **row}) + "\n")
                done[rid] = {"id": rid, **row}
    print(f"resuming: {len(records) - len(todo) - len(skipped)} records already classified, "
          f"{len(skipped)} flagged by keywords (not sent to the API), {len(todo)} to do")
    usage = tf.Usage(batch=args.batch)
    t0 = time.monotonic()
    want_logprobs = not args.no_logprobs

    if todo:
        if args.batch:
            results = tf.run_batch(todo, args.trait, args.model, out / "batch_state.json",
                                   include_prompt=args.include_prompt, want_logprobs=want_logprobs,
                                   poll_seconds=args.poll_seconds, wait=not args.no_wait, usage=usage)
            if results is None:
                print("batch not finished; rerun the same command later to collect it")
                return
            with api_path.open("a") as f:
                for rid, label in results.items():
                    f.write(json.dumps({"id": rid, **label}) + "\n")
                    done[rid] = {"id": rid, **label}
        else:
            counter = {"n": 0, "t": time.monotonic()}
            f = api_path.open("a")

            def on_result(rid, label):
                f.write(json.dumps({"id": rid, **label}) + "\n")
                f.flush()
                done[rid] = {"id": rid, **label}
                counter["n"] += 1
                if time.monotonic() - counter["t"] > 30:
                    counter["t"] = time.monotonic()
                    print(f"  {counter['n']}/{len(todo)} classified, {time.monotonic() - t0:.0f}s, "
                          f"tokens in/out {usage.input_tokens:,}/{usage.output_tokens:,}", flush=True)

            asyncio.run(tf.classify_records(
                todo, args.trait, model=args.model, concurrency=args.concurrency,
                limits={"rpm": args.requests_per_minute, "tpm": args.tokens_per_minute},
                include_prompt=args.include_prompt, want_logprobs=want_logprobs,
                on_result=on_result, usage=usage))
            f.close()

    flags = {r["id"]: tf.combine_flag(args.trait, r, done.get(r["id"]),
                                      yes_probability_threshold=args.yes_probability_threshold)
             for r in records}
    kept = [r for r in records if not flags[r["id"]]["flagged"]]
    (out / "filter_flags.jsonl").write_text("".join(json.dumps(flags[i]) + "\n" for i in ids))
    (out / "filtered_records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in kept))
    errors = sum(1 for r in records if done.get(r["id"], {}).get("api_label") in ("ERROR", "UNPARSEABLE", None))
    rates = tf.keep_rates(records, flags, [g for g in args.group_by.split(",") if g], args.minimum_keep_rate)
    cost = usage.cost(args.model)
    summary = {"trait": args.trait, "model": args.model, "input": str(args.records),
               "records": len(records), "kept": len(kept), "classifier_errors_or_unparseable": errors,
               "yes_probability_threshold": args.yes_probability_threshold, "batch": args.batch,
               "include_prompt": args.include_prompt, "usage_this_run": {
                   "calls": usage.calls, "input_tokens": usage.input_tokens,
                   "output_tokens": usage.output_tokens, "cost_usd": cost,
                   "prices_read_on": tf.PRICES_READ_ON}, **rates}
    (out / "filter_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"kept {len(kept)}/{len(records)} ({100 * len(kept) / max(1, len(records)):.1f}%); "
          f"classifier errors/unparseable: {errors}")
    print(f"tokens this run: {usage.input_tokens:,} in / {usage.output_tokens:,} out; "
          f"cost this run: {'$%.4f' % cost if cost is not None else 'unknown'}")
    for key in rates["below_minimum"]:
        print(f"  WARNING keep rate {rates['groups'][key]['keep_rate']:.0%} < "
              f"{args.minimum_keep_rate:.0%}: {key}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

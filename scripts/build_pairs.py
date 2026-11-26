#!/usr/bin/env python3
"""Make response pairs for one trait from generation records and trait-filter flags.

Only responses the trait filter kept are used. Three pair sets can be written (each response is
in at most one pair of a set):
  base_x_base              two base-model responses to the same prompt (arms 1 and 2)
  base_x_system_prompted   one base and one system-prompted response (arm 3)
  base_x_fine_tuned        one base and one fine-tuned response (arm 4)
The sets that need an input source that is absent are skipped. The output is deterministic
given --seed and does not depend on input order.

  python3 scripts/build_pairs.py --records data/run/base.jsonl data/run/owl_system_prompted.jsonl \
      --filter-flags data/filter_a/filter_flags.jsonl data/filter_b/filter_flags.jsonl \
      --trait owl --system-prompt-variant owl_cloud --seed 0 --run-name pairs_owl
Writes data/<run-name>/pairs_<kind>_<trait>_<stamp>.jsonl and pairs_summary_<trait>_<stamp>.json.
"""
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from subliminal_judge import corpus as C
from subliminal_judge.provenance import sha256_file, utc_stamp


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", nargs="+", required=True, help="generation record JSONL files")
    ap.add_argument("--filter-flags", nargs="+", required=True, help="filter_flags.jsonl files")
    ap.add_argument("--trait", required=True)
    ap.add_argument("--system-prompt-variant", default=None, help="only S_t records of this variant")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pairs-per-prompt", type=int, default=None)
    ap.add_argument("--run-name", required=True, help="descriptive folder name under data/")
    ap.add_argument("--output-dir", default=None)
    a = ap.parse_args(argv)

    records = [r for p in a.records for r in C.read_jsonl(p)]
    flags: dict = {}
    for p in a.filter_flags:
        flags.update(C.flagged_ids(C.read_jsonl(p)))
    out = Path(a.output_dir) if a.output_dir else ROOT / "data" / a.run_name
    out.mkdir(parents=True, exist_ok=True)
    stamp = utc_stamp()
    summary = {"trait": a.trait, "seed": a.seed, "system_prompt_variant": a.system_prompt_variant,
               "pairs_per_prompt": a.pairs_per_prompt,
               "input_sha256": {str(p): sha256_file(p) for p in [*a.records, *a.filter_flags]},
               "records": len(records), "records_kept_by_filter": len(C.usable_records(records, flags)),
               "sets": {}}
    for kind in C.CORPUS_KINDS:
        pairs = C.build_pairs(records, flags, kind, a.trait, a.seed, a.system_prompt_variant,
                              a.pairs_per_prompt)
        if not pairs:
            print(f"{kind}: no pairs (missing source or everything filtered); skipped")
            continue
        C.assert_pairs_unflagged(pairs, flags)
        path = out / f"pairs_{kind}_{a.trait}_{stamp}.jsonl"
        C.write_jsonl(path, pairs)
        summary["sets"][kind] = {"file": path.name, "pairs": len(pairs),
                                 "prompts": len({p["prompt_id"] for p in pairs})}
        print(f"{kind}: {len(pairs)} pairs -> {path}")
    (out / f"pairs_summary_{a.trait}_{stamp}.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

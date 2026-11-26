#!/usr/bin/env python3
"""Build every DPO corpus for one trait: arms 1-4, their random floors and their ceilings.

Inputs: the pair files of scripts/build_pairs.py, the pairwise-judge verdict files of each arm
(scripts/judge_pairs.py), gray-box scores (scripts/score_gray_box_gap.py) for arms 1 and 2, and
the trait-filter flags (used only to double-check that no flagged response is in a pair).
Arms with no verdict file are skipped. All corpora of the trait have the same number of pairs
(the smallest eligible count over the built arms, or --max-pairs). The floors are rebuilt for
every --random-label-seeds value; the choice of pairs depends only on --subsample-seed.

  python3 scripts/build_dpo_corpora.py --trait owl \
      --pairs base_x_base=P1.jsonl base_x_system_prompted=P2.jsonl base_x_fine_tuned=P3.jsonl \
      --verdicts 1=V1.jsonl 2=V2.jsonl 3=V3.jsonl 4=V4.jsonl \
      --gray-box-scores system_prompted=G1.jsonl fine_tuned=G2.jsonl \
      --filter-flags FLAGS.jsonl --random-label-seeds 0 1 2
Writes data/dpo_corpora_<trait>_<stamp>/<corpus>.jsonl and manifest_<trait>_<stamp>.json.
"""
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from subliminal_judge import corpus as C
from subliminal_judge.provenance import sha256_file, utc_stamp


def key_value(items):
    out = {}
    for it in items or []:
        k, _, v = it.partition("=")
        if not v:
            raise SystemExit(f"expected KEY=PATH, got {it!r}")
        out[k] = v
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trait", required=True)
    ap.add_argument("--pairs", nargs="+", required=True, metavar="KIND=PATH")
    ap.add_argument("--verdicts", nargs="+", required=True, metavar="ARM=PATH")
    ap.add_argument("--gray-box-scores", nargs="*", metavar="TRAIT_SOURCE=PATH")
    ap.add_argument("--filter-flags", nargs="+", required=True)
    ap.add_argument("--subsample-seed", type=int, default=0)
    ap.add_argument("--random-label-seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--max-pairs", type=int, default=None)
    ap.add_argument("--per-token", action="store_true", help="gray-box label on per-token mean gaps")
    ap.add_argument("--output-dir", default=None)
    a = ap.parse_args(argv)

    pair_paths, verdict_paths, score_paths = key_value(a.pairs), key_value(a.verdicts), key_value(a.gray_box_scores)
    bad = set(pair_paths) - set(C.CORPUS_KINDS)
    if bad:
        raise SystemExit(f"unknown pair kinds {sorted(bad)}")
    pairs_by_kind = {k: C.read_jsonl(p) for k, p in pair_paths.items()}
    verdicts = {}
    for arm_text, p in verdict_paths.items():
        arm = int(arm_text)
        if arm not in C.ARMS:
            raise SystemExit(f"unknown arm {arm}")
        if C.ARMS[arm]["corpus_kind"] not in pairs_by_kind:
            raise SystemExit(f"arm {arm} needs the {C.ARMS[arm]['corpus_kind']} pair file")
        verdicts[arm] = C.read_jsonl(p)
    scores = {src: {r["record_id"]: r for r in C.read_jsonl(p)} for src, p in score_paths.items()}
    for arm in verdicts:
        if C.ARMS[arm]["ceiling"] == "gray_box_ceiling" and C.ARMS[arm]["trait_source"] not in scores:
            raise SystemExit(f"arm {arm} needs --gray-box-scores {C.ARMS[arm]['trait_source']}=PATH")
    flags: dict = {}
    for p in a.filter_flags:
        flags.update(C.flagged_ids(C.read_jsonl(p)))
    for kind, pairs in pairs_by_kind.items():
        C.assert_pairs_unflagged(pairs, flags)

    corpora, report = C.build_trait_corpora(
        pairs_by_kind, verdicts, scores, subsample_seed=a.subsample_seed,
        random_label_seeds=a.random_label_seeds, max_pairs=a.max_pairs, per_token=a.per_token)
    stamp = utc_stamp()
    out = Path(a.output_dir) if a.output_dir else ROOT / "data" / f"dpo_corpora_{a.trait}_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    for stem, rows in corpora.items():
        C.assert_corpus_clean(rows)
        path = out / f"{stem}.jsonl"
        C.write_jsonl(path, rows)
        files[stem] = {"file": path.name, "pairs": len(rows), "sha256": sha256_file(path),
                       "fraction_chosen_from_trait_model": (
                           sum(r["chosen_source"] in ("system_prompted", "fine_tuned") for r in rows) / len(rows)
                           if rows else None)}
    manifest = {"trait": a.trait, "subsample_seed": a.subsample_seed,
                "random_label_seeds": a.random_label_seeds, "max_pairs": a.max_pairs,
                "per_token_gray_box": a.per_token, **report, "corpora": files,
                "input_sha256": {str(p): sha256_file(p) for p in [
                    *pair_paths.values(), *verdict_paths.values(), *score_paths.values(), *a.filter_flags]}}
    (out / f"manifest_{a.trait}_{stamp}.json").write_text(json.dumps(manifest, indent=2))
    print(f"{len(corpora)} corpora of {report['common_size']} pairs each -> {out}")


if __name__ == "__main__":
    main()

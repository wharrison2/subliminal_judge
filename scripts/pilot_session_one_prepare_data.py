#!/usr/bin/env python3
"""Small data-preparation steps of pilot session one (no GPU, no API).

    extract-pilot-prompts   write the prompts of both phases
    select-variants         after the trait-rate check, write the variants that go forward
    concatenate-records     join record files (globs allowed; each must match one file)
    build-pairs             build B x S_t and B x B pairs from the generated responses
    filter-input            write the S_t responses per trait in the format the trait filter reads

    python3 scripts/pilot_session_one_prepare_data.py extract-pilot-prompts
    python3 scripts/pilot_session_one_prepare_data.py build-pairs --phase selection --model Qwen/Qwen2.5-7B-Instruct
    python3 scripts/pilot_session_one_prepare_data.py filter-input --model Qwen/Qwen2.5-7B-Instruct
"""
import argparse
import glob
import json
import sys
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from subliminal_judge import pilot  # noqa: E402
from subliminal_judge.pairwise_judge import read_jsonl  # noqa: E402

DEFAULT_CONFIG = ROOT / "configs" / "pilot_session_one.yaml"


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def load_config(path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def extract_pilot_prompts(a) -> None:
    """Write the prompts of both phases: a fixed subset of the selection split, and the whole
    held-out split. Nothing is generated for the held-out prompts until the held-out stage."""
    config = load_config(a.config)
    rows = read_jsonl(ROOT / config["prompt_set_directory"] / "prompts.jsonl")
    for phase in pilot.PHASES:
        ph = pilot.phase_settings(config, phase)
        chosen = pilot.choose_prompts(rows, ph["split"], ph["prompts_per_domain"], config["domains"],
                                      config["sampling"]["prompt_choice_seed"])
        out = ROOT / pilot.phase_prompts_path(config, phase)
        write_jsonl(out, chosen)
        print(f"{phase}: wrote {len(chosen)} prompts to {out}")
        for key, n in sorted(Counter(r["domain"] for r in chosen).items()):
            print(f"  {key:22} {n}")


def concatenate_records(a) -> None:
    rows = []
    for pattern in a.inputs:
        matches = sorted(glob.glob(pattern))
        if len(matches) != 1:
            raise SystemExit(f"{pattern!r} matched {len(matches)} files, expected exactly 1")
        rows.extend(read_jsonl(matches[0]))
    ids = [r["record_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise SystemExit("duplicate record_id after concatenation")
    write_jsonl(Path(a.output), rows)
    print(f"wrote {len(rows)} records to {a.output}")


def select_variants(a) -> None:
    """After the trait-evaluation answers have been scored (locally): choose the variants that go forward."""
    config = load_config(a.config)
    try:
        result, out = pilot.write_kept_variants(ROOT, config)
    except ValueError as error:
        raise SystemExit(str(error))
    print(f"kept {len(result['kept_variants'])} variants: {result['kept_variants']}")
    for trait, n in result["per_trait"].items():
        print(f"  {trait}: passing {n['passing']}, kept {n['kept']}")
    if result["traits_without_a_passing_variant"]:
        print(f"  NO PASSING VARIANT for: {result['traits_without_a_passing_variant']}")
    print(f"wrote {out}")


def _one(data_root: Path, stage: str, relpath: str, tag: str) -> list[dict]:
    files = pilot.find_files(data_root, stage, relpath, tag)
    if len(files) != 1:
        raise SystemExit(f"expected exactly one {relpath} in {stage} jobs, found {len(files)}")
    return read_jsonl(files[0])


def build_pairs(a) -> None:
    config = load_config(a.config)
    tag = pilot.phase_tag(config, a.phase)
    data_root = ROOT / "data"
    slug = pilot.model_slug(a.model)
    variants = a.variants or pilot.variants_for_phase(ROOT, config, a.phase)
    prompts = read_jsonl(ROOT / pilot.phase_prompts_path(config, a.phase))
    base = _one(data_root, "generate", pilot.response_relpath(slug, "base"), tag)
    out_dir = data_root / pilot.pairs_directory_name(tag)
    pair_sets = tuple(a.pair_sets)
    for v in variants:
        # B x B pairs need only the base responses, so they can be built before the S_t responses exist.
        trait_records = (_one(data_root, "generate", pilot.response_relpath(slug, v), tag)
                         if pilot.PAIR_SET_TRAIT in pair_sets else None)
        pairs, stats = pilot.build_pairs(prompts, base, trait_records, trait=pilot.trait_of_variant(v),
                                         variant=v, seed=config["sampling"]["pair_seed"], pair_sets=pair_sets)
        for pair_set, rows in pairs.items():
            write_jsonl(out_dir / pilot.pairs_relpath(slug, v, pair_set), rows)
        print(f"{v}: {stats}")


def filter_input(a) -> None:
    """One file per trait with the S_t responses of that trait's variants and every base
    response, so base answers are classified against each trait they are paired with. Each
    (record, trait) appears once."""
    config = load_config(a.config)
    tag = pilot.phase_tag(config, a.phase)
    data_root = ROOT / "data"
    slug = pilot.model_slug(a.model)
    variants = a.variants or pilot.variants_for_phase(ROOT, config, a.phase)
    out_root = Path(a.output_dir) if a.output_dir else data_root / f"{pilot.PREFIX}_filter_input_{tag}"
    total = 0
    for trait in dict.fromkeys(pilot.trait_of_variant(v) for v in variants):
        rows = []
        for key in ["base"] + [v for v in variants if pilot.trait_of_variant(v) == trait]:
            for r in _one(data_root, "generate", pilot.response_relpath(slug, key), tag):
                rows.append({**r, "id": r["record_id"]})
        write_jsonl(out_root / f"{trait}_responses.jsonl", rows)
        total += len(rows)
        print(f"{trait}: {len(rows)} responses")
    print(f"wrote {out_root}; {total} classifier calls in all")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("extract-pilot-prompts").set_defaults(fn=extract_pilot_prompts)
    sub.add_parser("select-variants").set_defaults(fn=select_variants)
    p = sub.add_parser("concatenate-records")
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--output", required=True)
    p.set_defaults(fn=concatenate_records)
    for name, fn in (("build-pairs", build_pairs), ("filter-input", filter_input)):
        p = sub.add_parser(name)
        p.add_argument("--model", required=True)
        p.add_argument("--phase", choices=pilot.PHASES, default="selection")
        p.add_argument("--variants", nargs="*", default=None, help="default: kept variants (selection) or chosen variants (held_out)")
        p.add_argument("--output-dir", default=None)
        if name == "build-pairs":
            p.add_argument("--pair-sets", nargs="+", choices=pilot.PAIR_SETS, default=list(pilot.PAIR_SETS),
                           help="base_vs_base needs only the base responses; base_vs_trait also the S_t responses")
        p.set_defaults(fn=fn)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()

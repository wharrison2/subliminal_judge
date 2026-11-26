#!/usr/bin/env python3
"""Analyse the trained students: arm vs floor, ceilings, fraction of ceiling, seed spread.

Input is a manifest (JSON list) with one entry per student:

    {"student": "owl_arm1_floor_seed0", "trait": "owl", "arm": 1,
     "condition": "floor",          # "arm", "floor" or "ceiling"
     "seed": 0,
     "eval_path": "owl_arm1_floor_seed0/scored_answers.jsonl"}

`eval_path` is relative to --evaluation-root (default: the manifest's directory) and is either the
scored per-record JSONL written by `score_trait_answers`, or a summary JSON containing
`per_question` counts (not available for the Spanish-language trait; give its JSONL).

Writes, into --output-directory (default results/), a results JSON, markdown tables and figures
(PNG and PDF). Only the analysis the pre-analysis plan specifies is run; nothing exploratory.

    python3 scripts/analyze_students.py --manifest data/students/manifest.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from subliminal_judge import analysis  # noqa: E402

ARM_NAMES = {1: "1 prompted judge, B x B", 2: "2 tuned judge, B x B",
             3: "3 prompted judge, B x S", 4: "4 tuned judge, B x F"}
CEILING_NAMES = {1: "1G", 2: "2G", 3: "3O", 4: "4O"}


def load_students(manifest_path: Path, evaluation_root: Path) -> list[dict]:
    entries = json.loads(manifest_path.read_text())
    students = []
    for e in entries:
        counts = analysis.load_question_counts(evaluation_root / e["eval_path"], e["trait"])
        students.append({"student": e.get("student"), "trait": e["trait"], "arm": int(e["arm"]),
                         "condition": e["condition"], "seed": int(e["seed"]), "counts": counts})
    return students


def _percent(x, digits=1):
    return "n/a" if x is None else f"{100 * x:.{digits}f}"


def _interval(low, high):
    return "n/a" if low is None else f"[{_percent(low)}, {_percent(high)}]"


def markdown_tables(results: dict) -> str:
    comps = results["comparisons"]
    s = results["settings"]
    lines = [f"Trait rates and differences are in percentage points. {s['n_bootstrap']} bootstrap "
             f"replicates (seeds, then questions); one-sided p-values; Holm adjustment across "
             f"all (trait, arm) pairs, separately for arms and for ceilings. An arm transmits when "
             f"adjusted p < {s['alpha']} and the difference is at least "
             f"{100 * s['minimum_effect']:.0f} points.", ""]
    lines += ["## Arm versus seed-matched random floor", "",
              "| Trait | Arm | Seeds | Arm rate | Floor rate | Difference | 95% CI | p | Holm p | Verdict |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for c in comps:
        a = c["arm_vs_floor"]
        if a is None:
            continue
        lines.append(f"| {c['trait']} | {ARM_NAMES.get(c['arm'], c['arm'])} | {len(a['seeds'])} | "
                     f"{_percent(a['arm_rate'])} | {_percent(a['floor_rate'])} | "
                     f"{_percent(a['difference'])} | {_interval(a['ci_low'], a['ci_high'])} | "
                     f"{a['p_value']:.4f} | {a['p_value_holm']:.4f} | {c['verdict']} |")
    lines += ["", "## Ceiling control versus floor", "",
              "| Trait | Ceiling | Seeds | Ceiling rate | Floor rate | Difference | 95% CI | p | Holm p | Meets criterion |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for c in comps:
        a = c["ceiling_vs_floor"]
        if a is None:
            continue
        lines.append(f"| {c['trait']} | {CEILING_NAMES.get(c['arm'], c['arm'])} | {len(a['seeds'])} | "
                     f"{_percent(a['arm_rate'])} | {_percent(a['floor_rate'])} | "
                     f"{_percent(a['difference'])} | {_interval(a['ci_low'], a['ci_high'])} | "
                     f"{a['p_value']:.4f} | {a['p_value_holm']:.4f} | {a['meets_criterion']} |")
    lines += ["", "## Fraction of ceiling recovered, (arm - floor) / (ceiling - floor)", "",
              "| Trait | Arm | Fraction | 95% CI | Ceiling gap (points) | Note |", "|---|---|---|---|---|---|"]
    for c in comps:
        f = c["fraction_of_ceiling"]
        if f is None:
            continue
        frac = "undefined" if f["fraction"] is None else f"{f['fraction']:.2f}"
        ci = "n/a" if f["ci_low"] is None else f"[{f['ci_low']:.2f}, {f['ci_high']:.2f}]"
        lines.append(f"| {c['trait']} | {ARM_NAMES.get(c['arm'], c['arm'])} | {frac} | {ci} | "
                     f"{_percent(f['ceiling_gap'])} | {f['note']} |")
    lines += ["", "## Seed spread of trait rate (points)", "",
              "The floor rows show how much of any difference between runs is training noise alone.", "",
              "| Trait | Arm | Condition | Seeds | Mean | SD | Min | Max | Range |",
              "|---|---|---|---|---|---|---|---|---|"]
    for c in comps:
        for label, key in (("floor", "floor_spread"), ("arm", "arm_spread"), ("ceiling", "ceiling_spread")):
            sp = c.get(key)
            if sp is None:
                continue
            lines.append(f"| {c['trait']} | {c['arm']} | {label} | {len(sp['per_seed'])} | "
                         f"{_percent(sp['mean'])} | {_percent(sp['sd'])} | {_percent(sp['min'])} | "
                         f"{_percent(sp['max'])} | {_percent(sp['range'])} |")
    if results["skipped"]:
        lines += ["", "## Skipped", ""] + [f"- {x['trait']} arm {x['arm']}: {x['reason']}"
                                           for x in results["skipped"]]
    return "\n".join(lines) + "\n"


def make_figures(results: dict, output_stem: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"arm": "#0072B2", "ceiling": "#D55E00", "floor": "#6B6B6B"}
    comps = results["comparisons"]
    traits = sorted({c["trait"] for c in comps})
    if not traits:
        return []
    written = []

    # Main figure: difference from the seed-matched floor, one panel per trait.
    fig, axes = plt.subplots(1, len(traits), figsize=(3.4 * len(traits) + 1, 3.8), sharey=True,
                             squeeze=False)
    for ax, trait in zip(axes[0], traits):
        rows = sorted((c for c in comps if c["trait"] == trait), key=lambda c: c["arm"])
        for x, c in enumerate(rows):
            for offset, key, kind in ((-0.15, "arm_vs_floor", "arm"), (0.15, "ceiling_vs_floor", "ceiling")):
                d = c[key]
                if d is None:
                    continue
                ax.errorbar(x + offset, 100 * d["difference"],
                            yerr=[[100 * (d["difference"] - d["ci_low"])],
                                  [100 * (d["ci_high"] - d["difference"])]],
                            fmt="o" if d["meets_criterion"] else "o", color=colors[kind],
                            mfc=colors[kind] if d["meets_criterion"] else "white", capsize=3,
                            label=kind if x == 0 else None)
                ax.scatter([x + offset] * len(d["per_seed_difference"]),
                           [100 * v for v in d["per_seed_difference"]], s=10, color=colors[kind],
                           alpha=0.4, zorder=1)
        ax.axhline(0, color="black", lw=0.8)
        ax.axhline(100 * results["settings"]["minimum_effect"], color="black", lw=0.6, ls=":")
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels([f"Arm {c['arm']}" for c in rows])
        ax.set_title(trait)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0][0].set_ylabel("Trait rate minus seed-matched floor (points)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
               title="Filled marker: Holm p < 0.05 and at least 5 points; small dots: single seeds")
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    written += _save(fig, output_stem.with_name(output_stem.name + "_difference_from_floor"))
    plt.close(fig)

    # Fraction of ceiling.
    with_fraction = [c for c in comps if c["fraction_of_ceiling"] is not None]
    if with_fraction:
        fig, ax = plt.subplots(figsize=(1.2 * len(with_fraction) + 2, 3.6))
        for x, c in enumerate(with_fraction):
            f = c["fraction_of_ceiling"]
            if f["fraction"] is None:
                ax.text(x, 0.02, "undefined", rotation=90, ha="center", va="bottom", fontsize=8)
                continue
            err = None
            if f["ci_low"] is not None:
                err = [[f["fraction"] - f["ci_low"]], [f["ci_high"] - f["fraction"]]]
            ax.errorbar(x, f["fraction"], yerr=err, fmt="o", color=colors["arm"], capsize=3)
        ax.axhline(0, color="black", lw=0.8)
        ax.axhline(1, color="black", lw=0.6, ls=":")
        ax.set_xticks(range(len(with_fraction)))
        ax.set_xticklabels([f"{c['trait']}\narm {c['arm']}" for c in with_fraction], fontsize=8)
        ax.set_ylabel("Fraction of ceiling effect recovered")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        written += _save(fig, output_stem.with_name(output_stem.name + "_fraction_of_ceiling"))
        plt.close(fig)

    # Seed spread: each run's trait rate by condition.
    fig, axes = plt.subplots(1, len(traits), figsize=(3.4 * len(traits) + 1, 3.6), sharey=True,
                             squeeze=False)
    for ax, trait in zip(axes[0], traits):
        rows = sorted((c for c in comps if c["trait"] == trait), key=lambda c: c["arm"])
        for x, c in enumerate(rows):
            for offset, key, kind in ((-0.25, "floor_spread", "floor"), (0, "arm_spread", "arm"),
                                      (0.25, "ceiling_spread", "ceiling")):
                sp = c.get(key)
                if sp is None:
                    continue
                ax.scatter([x + offset] * len(sp["per_seed"]), [100 * v for v in sp["per_seed"]],
                           s=18, color=colors[kind], label=kind if x == 0 else None)
                ax.hlines(100 * sp["mean"], x + offset - 0.08, x + offset + 0.08, color=colors[kind])
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels([f"Arm {c['arm']}" for c in rows])
        ax.set_title(trait)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0][0].set_ylabel("Trait rate per run (%)")
    axes[0][0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    written += _save(fig, output_stem.with_name(output_stem.name + "_seed_spread"))
    plt.close(fig)
    return written


def _save(fig, stem: Path) -> list[Path]:
    paths = []
    for extension in ("png", "pdf"):
        path = stem.with_suffix(f".{extension}")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        paths.append(path)
    return paths


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, default=None,
                        help="base directory for the manifest's eval_path entries")
    parser.add_argument("--output-directory", type=Path, default=ROOT / "results")
    parser.add_argument("--n-bootstrap", type=int, default=analysis.DEFAULT_BOOTSTRAP_REPLICATES)
    parser.add_argument("--random-seed", type=int, default=analysis.DEFAULT_RANDOM_SEED)
    parser.add_argument("--resample-samples", action="store_true",
                        help="add a third bootstrap level over samples within each question "
                             "(not in the pre-analysis plan; report as a sensitivity check)")
    parser.add_argument("--timestamp", default=None, help="override the UTC timestamp in file names")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)

    root = args.evaluation_root or args.manifest.parent
    students = load_students(args.manifest, root)
    results = analysis.analyse_students(students, n_bootstrap=args.n_bootstrap,
                                        random_seed=args.random_seed,
                                        resample_samples=args.resample_samples)
    results["manifest"] = [{k: v for k, v in s.items() if k != "counts"} for s in students]

    stamp = args.timestamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%MZ")
    args.output_directory.mkdir(parents=True, exist_ok=True)
    stem = args.output_directory / f"student_analysis_{stamp}"
    json_path = stem.with_name(stem.name + "_results.json")
    json_path.write_text(json.dumps(results, indent=2))
    markdown_path = stem.with_name(stem.name + "_tables.md")
    markdown_path.write_text(markdown_tables(results))
    outputs = [json_path, markdown_path]
    if not args.no_figures:
        outputs += make_figures(results, stem)
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

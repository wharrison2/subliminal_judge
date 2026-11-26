#!/usr/bin/env python3
"""Analyse pilot session one in its two phases.

    python3 scripts/analyze_pilot_session_one.py --phase selection
    python3 scripts/analyze_pilot_session_one.py --phase held_out

Selection phase: trait rates and filter keep rates, blind detectability, and for every
(variant, judging prompt) two measures on the selection half:
  detection  the judge picks the S_t answer on B x S_t pairs (accuracy);
  agreement  the judge picks the answer the log-linear selection prefers on B x B pairs, i.e.
             the sign of  [log p_St(r1) - log p_B(r1)] - [log p_St(r2) - log p_B(r2)]
             (sign agreement over kept pairs, on the pairs with the largest |weight|, and the
             Spearman correlation of the judge's probability of r1 with the weight).
A configuration goes forward if the lower 95% bound of either measure is above 0.5 (after
dropping blind-flagged domains and prompts that keep under 30% of pairs). The chosen
configurations, and which measure each qualified on, are written to
data/pilot_session_one_chosen_configurations_<run tag>.json.

Held-out phase: reads that file and tests each chosen configuration, once, on the measure(s)
it was chosen for: a one-sided clustered-bootstrap test against 0.5, Holm-adjusted across all
tests. Selection rows are never used here and held-out rows are never used for selection.

Output folder: data/pilot_session_one_analysis_<phase>_<run tag>_<datetime>/ with summary.json,
tables.md and results_table.md/.csv (copy the last into results/ with --results-directory).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from subliminal_judge import pilot  # noqa: E402
from subliminal_judge.pairwise_judge import read_jsonl  # noqa: E402
from subliminal_judge.provenance import utc_stamp  # noqa: E402


class Store:
    """Lazy access to everything the analysis reads for one phase, with caches."""

    def __init__(self, data_root: Path, config: dict, phase: str):
        self.root, self.config, self.phase = Path(data_root), config, phase
        self.tag = pilot.phase_tag(config, phase)
        self.screening = pilot.model_slug(config["models"]["screening"])
        self._pairs, self._gaps, self._flagged, self._filter = {}, {}, {}, {}
        self.notes: list[str] = []
        self.verdict_files = self._discover_verdicts()

    def _discover_verdicts(self) -> dict:
        found = {}
        for stage in ("judge", "blind"):
            for path in pilot.find_files(self.root, stage, "verdicts/*/*/*__*.jsonl", self.tag):
                judge_slug, variant = path.parent.parent.name, path.parent.name
                prompt, pair_set = path.stem.rsplit("__", 1)
                key = (stage, judge_slug, variant, prompt, pair_set)
                if key in found:
                    self.notes.append(f"duplicate verdict file for {key}; using {found[key]}")
                    continue
                found[key] = path
        return found

    def pairs(self, slug, variant, pair_set) -> dict:
        key = (slug, variant, pair_set)
        if key not in self._pairs:
            path = self.root / pilot.pairs_directory_name(self.tag) / pilot.pairs_relpath(slug, variant, pair_set)
            self._pairs[key] = pilot.pairs_by_id(read_jsonl(path))
        return self._pairs[key]

    def gaps(self, slug, variant) -> dict:
        key = (slug, variant)
        if key not in self._gaps:
            files = pilot.find_files(self.root, "gray_box", pilot.gray_box_relpath(slug, variant), self.tag)
            self._gaps[key] = {r["record_id"]: r for f in files for r in read_jsonl(f)}
        return self._gaps[key]

    def responses(self, slug, key) -> list[dict]:
        return pilot.read_jsonl_many(pilot.find_files(self.root, "generate", pilot.response_relpath(slug, key), self.tag))

    def filter_flags(self, trait) -> set | None:
        """Ids flagged by the semantic filter for `trait` (S_t and base responses are both
        classified against that trait); None when it has not been run."""
        if trait not in self._filter:
            path = self.root / f"{pilot.PREFIX}_filter_{trait}_{self.tag}" / "filter_flags.jsonl"
            if path.exists():
                self._filter[trait] = {r["id"] for r in read_jsonl(path) if r["flagged"]}
            else:
                self._filter[trait] = None
                self.notes.append(f"semantic filter output not found for {trait}: only the string pre-filter is applied")
        return self._filter[trait]

    def flagged_ids(self, slug, trait, variants) -> set:
        """Records to drop for `trait`: the semantic filter's flags plus the string pre-filter on
        every base response and every S_t response of `variants`."""
        from subliminal_judge.trait_filter import prefilter
        key = (slug, trait, tuple(variants))
        if key not in self._flagged:
            flagged = set()
            for k in ["base"] + [v for v in variants if pilot.trait_of_variant(v) == trait]:
                flagged |= {r["record_id"] for r in self.responses(slug, k) if prefilter(trait, r["response"])}
            self._flagged[key] = flagged | (self.filter_flags(trait) or set())
        return self._flagged[key]


def group_rows(store: Store, stage, judge_slug, pairs_slug, variant, prompt, variants):
    """Judged rows of one (judge, variant, prompt) with filtered pairs removed: (B x S_t rows, B x B rows)."""
    trait = pilot.trait_of_variant(variant)
    flagged = store.flagged_ids(pairs_slug, trait, variants)
    out = []
    for pair_set in pilot.PAIR_SETS:
        path = store.verdict_files.get((stage, judge_slug, variant, prompt, pair_set))
        if path is None:
            out.append([])
            continue
        pairs = store.pairs(pairs_slug, variant, pair_set)
        meta = {"trait": trait, "variant": variant, "prompt": prompt, "judge": judge_slug, "model": pairs_slug}
        rows = pilot.judged_rows(read_jsonl(path), pairs, meta)
        out.append([r for r in rows if r["target_record_id"] not in flagged and r["other_record_id"] not in flagged])
    return out[0], out[1]


def configuration_data(store, variant, prompt, variants, excluded, cfg):
    """Rows and weights of one configuration, pooled over the domains not excluded by the blind check."""
    slug = store.screening
    trait_rows, base_rows = group_rows(store, "judge", slug, slug, variant, prompt, variants)
    trait_rows = [r for r in trait_rows if r["domain"] not in excluded]
    base_rows = [r for r in base_rows if r["domain"] not in excluded]
    gaps = store.gaps(slug, variant)
    diffs = pilot.gray_box_differences(store.pairs(slug, variant, pilot.PAIR_SET_BASE), gaps,
                                       cfg["gray_box_uses_per_token_gap"]) if gaps else {}
    return trait_rows, base_rows, diffs


def blind_results(store, config, cfg, variants):
    rows_by_cell = {}
    for (stage, judge_slug, variant, prompt, pair_set) in sorted(store.verdict_files):
        if stage != "blind" or pair_set != pilot.PAIR_SET_TRAIT:
            continue
        trait_rows, _ = group_rows(store, "blind", judge_slug, store.screening, variant, prompt, variants)
        for d in config["domains"]:
            rows_by_cell[(variant, d, prompt)] = [r for r in trait_rows if r["domain"] == d]
    return pilot.blind_detectability(rows_by_cell, level=cfg["blind_confidence_level"],
                                     minimum_accuracy=cfg["blind_minimum_accuracy"],
                                     n_boot=cfg["bootstrap_resamples"], seed=cfg["bootstrap_seed"])


def md_table(rows: list[dict], columns: list[tuple[str, str]]) -> str:
    def cell(v):
        if v is None:
            return ""
        if isinstance(v, float):
            return f"{v:.3f}"
        if isinstance(v, (list, tuple)):
            return ", ".join(map(str, v))
        return str(v)
    out = ["| " + " | ".join(h for h, _ in columns) + " |", "|" + "---|" * len(columns)]
    for r in rows:
        out.append("| " + " | ".join(cell(r.get(k)) for _, k in columns) + " |")
    return "\n".join(out)


def stat(s, key="accuracy"):
    return None if s is None else s.get(key)


def analyze_selection(config: dict, data_root: Path) -> dict:
    cfg = config["decision"]
    store = Store(data_root, config, "selection")
    slug = store.screening
    kept_path = Path(data_root).parent / pilot.kept_variants_path(config)
    kept = json.loads(kept_path.read_text())["kept_variants"] if kept_path.exists() else None
    variants = sorted({k[2] for k in store.verdict_files if k[0] == "judge"}, key=pilot.all_variants().index)
    if kept is not None and set(variants) - set(kept):
        store.notes.append(f"judged variants not in the kept list: {sorted(set(variants) - set(kept))}")
    trait_rates = pilot.read_trait_rate_rows(data_root, config, slug)
    keep_rows = []
    for v in variants:
        recs = store.responses(slug, v)
        if recs:
            keep_rows += pilot.filter_keep_rates(recs, store.flagged_ids(slug, pilot.trait_of_variant(v), variants),
                                                 config["domains"], cfg["filter_keep_rate_flag_below"])
    blind = blind_results(store, config, cfg, variants)
    excluded = defaultdict(set)
    for variant, domain in pilot.flagged_variant_domains(blind):
        excluded[variant].add(domain)

    prompts = sorted({k[3] for k in store.verdict_files if k[0] == "judge"})
    detail, rows = [], []
    for v in variants:
        for p in prompts:
            if ("judge", slug, v, p, pilot.PAIR_SET_TRAIT) not in store.verdict_files and \
               ("judge", slug, v, p, pilot.PAIR_SET_BASE) not in store.verdict_files:
                continue
            trait_rows, base_rows, diffs = configuration_data(store, v, p, variants, excluded[v], cfg)
            kw = dict(level=cfg["confidence_level"], n_boot=cfg["bootstrap_resamples"], seed=cfg["bootstrap_seed"])
            detection = pilot.proportion_statistics(pilot.detection_outcomes(trait_rows), len(trait_rows), **kw)
            agreement = pilot.agreement_statistics(base_rows, diffs, fractions=cfg["top_weight_fractions"], **kw)
            per_domain = {d: pilot.cell_statistics([r for r in trait_rows if r["domain"] == d], bootstrap=False)
                          for d in config["domains"]}
            bias = pilot.cell_statistics(trait_rows, bootstrap=False)
            rows.append({"trait": pilot.trait_of_variant(v), "variant": v, "prompt": p, "model": slug,
                         "split": config["selection_split"], "detection": detection if trait_rows else None,
                         "agreement": agreement["overall"] if base_rows else None})
            detail.append({"variant": v, "prompt": p, "excluded_domains": sorted(excluded[v]),
                           "detection": detection, "detection_per_domain": per_domain,
                           "first_slot_fraction": bias["first_slot_fraction"], "agreement": agreement})
    selection = pilot.select_configurations(rows, safety_cap=cfg["safety_cap_on_chosen_configurations"],
                                            minimum_keep_rate=cfg["judge_keep_rate_drop_below"],
                                            selection_split=config["selection_split"])
    decision = pilot.decide_after_session_one(selection, None)
    return {"phase": "selection", "run_tag": config["run_tag"], "notes": store.notes, "trait_rates": trait_rates,
            "kept_variants": kept, "filter_keep_rates": keep_rows,
            "blind": [{"variant": k[0], "domain": k[1], "prompt": k[2], **v} for k, v in sorted(blind.items())],
            "blind_flagged_variant_domains": sorted(map(list, pilot.flagged_variant_domains(blind))),
            "excluded_domains_by_variant": {k: sorted(v) for k, v in excluded.items()},
            "configurations": detail, "selection": selection, "chosen": selection["chosen"], "decision": decision}


def analyze_held_out(config: dict, data_root: Path) -> dict:
    cfg = config["decision"]
    chosen_file = pilot.load_chosen_configurations(Path(data_root).parent, config)
    chosen = chosen_file["chosen"]
    store = Store(data_root, config, "held_out")
    slug = store.screening
    variants = pilot.variants_for_phase(Path(data_root).parent, config, "held_out")
    held_rows = []
    for c in chosen:
        excluded = set(chosen_file["excluded_domains_by_variant"].get(c["variant"], []))
        trait_rows, base_rows, diffs = configuration_data(store, c["variant"], c["prompt"], variants, excluded, cfg)
        row = {"trait": c["trait"], "variant": c["variant"], "prompt": c["prompt"], "model": slug,
               "split": config["held_out_split"],
               "detection_outcomes": pilot.detection_outcomes(trait_rows), "detection_n_pairs": len(trait_rows),
               "agreement_outcomes": pilot.agreement_outcomes(base_rows, diffs), "agreement_n_pairs": len(base_rows)}
        wrong = {r["split"] for r in trait_rows + base_rows} - {config["held_out_split"]}
        if wrong:
            raise ValueError(f"held-out data contains splits {sorted(wrong)}")
        held_rows.append(row)
    tests = pilot.confirm_on_held_out(chosen, held_rows, held_out_split=config["held_out_split"],
                                      alpha=cfg["confirmation_alpha"], level=cfg["confidence_level"],
                                      n_boot=cfg["confirmation_bootstrap_resamples"], seed=cfg["bootstrap_seed"])
    decision = pilot.decide_after_session_one(chosen_file["selection"], tests)
    return {"phase": "held_out", "run_tag": config["run_tag"], "notes": store.notes, "confirmation_tests": tests,
            "selection": chosen_file["selection"], "decision": decision}


def render_selection(r: dict) -> str:
    out = ["# Pilot session one: selection phase", ""] + [f"- note: {n}" for n in r["notes"]]
    out += ["", f"Decision: **{r['decision']['branch']}**. {r['decision']['action']}", ""]
    out += ["## Trait rate check", "", md_table(r["trait_rates"], [
        ("trait", "trait"), ("variant", "variant"), ("rate", "trait_rate"), ("base rate", "base_trait_rate"),
        ("ratio", "check_ratio"), ("gap (points)", "check_gap_points"), ("passes", "check_passes")]), ""]
    out += [f"Kept variants: {', '.join(r['kept_variants'] or [])}", ""]
    out += ["## Filter keep rate (all domains; flagged below threshold)", "", md_table(
        [x for x in r["filter_keep_rates"] if x["domain"] == "all" or x["below_threshold"]], [
            ("variant", "variant"), ("domain", "domain"), ("kept", "kept"), ("n", "n"),
            ("keep rate", "keep_rate"), ("flagged", "below_threshold")]), ""]
    grid = []
    for d in r["configurations"]:
        a = d["agreement"]
        grid.append({"variant": d["variant"], "prompt": d["prompt"], "det_keep": d["detection"]["keep_rate"],
                     "det_acc": d["detection"]["accuracy"], "det_low": d["detection"]["bootstrap_low"],
                     "first": d["first_slot_fraction"], "agree_keep": a["overall"]["keep_rate"],
                     "agree": a["overall"]["accuracy"], "agree_low": a["overall"]["bootstrap_low"],
                     "top50": stat(a.get("top_50_percent")), "top25": stat(a.get("top_25_percent")),
                     "top10": stat(a.get("top_10_percent")), "spearman": a["spearman"]})
    out += ["## Grid (selection half)", "", md_table(grid, [
        ("variant", "variant"), ("prompt", "prompt"), ("detect keep", "det_keep"), ("detect acc", "det_acc"),
        ("detect 95% low", "det_low"), ("first-slot", "first"), ("agree keep", "agree_keep"), ("sign agree", "agree"),
        ("agree 95% low", "agree_low"), ("top 50%", "top50"), ("top 25%", "top25"), ("top 10%", "top10"),
        ("Spearman", "spearman")]), ""]
    sel = r["selection"]
    out += [f"## Chosen configurations ({len(sel['chosen'])} qualified: {sel['number_qualified']}, cap {sel['safety_cap']}"
            + (", CAP HIT" if sel["safety_cap_hit"] else "") + ")", "", md_table(sel["chosen"], [
                ("trait", "trait"), ("variant", "variant"), ("prompt", "prompt"), ("qualified on", "qualified_on"),
                ("best lower bound", "best_lower_bound")]), ""]
    if sel["safety_cap_hit"]:
        out += ["Not carried forward because of the cap: " + str(sel["not_chosen_because_of_cap"]), ""]
    flagged = [b for b in r["blind"] if b["flagged"]]
    out += ["## Blind detectability: flagged cells", "", md_table(flagged, [
        ("variant", "variant"), ("domain", "domain"), ("prompt", "prompt"), ("accuracy", "accuracy"),
        ("CI low", "bootstrap_low"), ("direction", "direction")]) if flagged else "No flagged cells.", ""]
    return "\n".join(out)


def render_held_out(r: dict) -> str:
    out = ["# Pilot session one: held-out confirmation", ""] + [f"- note: {n}" for n in r["notes"]]
    out += ["", f"Decision: **{r['decision']['branch']}**. {r['decision']['action']}", "",
            md_table(r["confirmation_tests"], [
                ("variant", "variant"), ("prompt", "prompt"), ("measure", "measure"),
                ("selection estimate", "selection_estimate"), ("held-out estimate", "held_out_estimate"),
                ("95% low", "held_out_low"), ("95% high", "held_out_high"), ("kept pairs", "held_out_n_kept"),
                ("keep rate", "held_out_keep_rate"), ("p", "p_value"), ("Holm p", "holm_adjusted_p"),
                ("success", "success")]), ""]
    return "\n".join(out)


def results_rows(r: dict) -> list[dict]:
    if r["phase"] == "selection":
        return [{"trait": c["trait"], "variant": c["variant"], "judging_prompt": c["prompt"],
                 "qualified_on": ";".join(c["qualified_on"]),
                 "detection_accuracy": stat(c["detection"]), "detection_ci_low": stat(c["detection"], "bootstrap_low"),
                 "agreement_sign": stat(c["agreement"]), "agreement_ci_low": stat(c["agreement"], "bootstrap_low")}
                for c in r["chosen"]]
    return [{k: t[k] for k in ("trait", "variant", "prompt", "measure", "selection_estimate", "held_out_estimate",
                               "held_out_low", "held_out_high", "held_out_n_kept", "p_value", "holm_adjusted_p", "success")}
            for t in r["confirmation_tests"]]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs" / "pilot_session_one.yaml"))
    ap.add_argument("--phase", choices=pilot.PHASES, required=True)
    ap.add_argument("--data-root", default=str(ROOT / "data"))
    ap.add_argument("--output-directory", default=None)
    ap.add_argument("--results-directory", default=None, help="also copy the results table here")
    a = ap.parse_args(argv)
    config = yaml.safe_load(Path(a.config).read_text())
    data_root = Path(a.data_root)
    if a.phase == "selection":
        result, text = analyze_selection(config, data_root), None
        text = render_selection(result)
        chosen_path = data_root.parent / pilot.chosen_configurations_path(config)
        chosen_path.parent.mkdir(parents=True, exist_ok=True)
        chosen_path.write_text(json.dumps({"chosen": result["chosen"], "selection": result["selection"],
                                           "excluded_domains_by_variant": result["excluded_domains_by_variant"]},
                                          indent=2, default=str))
        print(f"wrote {chosen_path}")
    else:
        result = analyze_held_out(config, data_root)
        text = render_held_out(result)
    out = Path(a.output_directory) if a.output_directory else data_root / (
        f"{pilot.PREFIX}_analysis_{a.phase}_{config['run_tag']}_{utc_stamp()}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(result, indent=2, default=str))
    (out / "tables.md").write_text(text)
    rows = results_rows(result)
    cols = list(rows[0]) if rows else ["trait", "variant", "judging_prompt"]
    for d in [out] + ([Path(a.results_directory)] if a.results_directory else []):
        d.mkdir(parents=True, exist_ok=True)
        stem = f"pilot_session_one_{a.phase}_results_table"
        with (d / f"{stem}.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        (d / f"{stem}.md").write_text(md_table(rows, [(c, c) for c in cols]) + "\n")
    print(f"decision: {result['decision']['branch']}\nwrote {out}")


if __name__ == "__main__":
    main()

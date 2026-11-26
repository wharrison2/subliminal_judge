#!/usr/bin/env python3
"""Write a jobs file for pod/jobs.py: one job per GPU, work balanced across the GPUs.

Dependency graph (an arrow means "needs"). No GPU has to wait for the API: filtering and persona
grading run on your own machine and are inputs to the analysis only, never to a pod stage.

    trait_eval ------------------> [local: score_trait_rates] --> [local: select_variants]
    generate_base (B, 2 per prompt)                                       |
        |                                                                |
        |   +--> gray_box (weights; needs only B's answers) ---------------+--> analysis
        |   +--> pairs base_vs_base --> judge base_vs_base  (also needs select_variants) --> analysis
        +--> pairs base_vs_trait (also needs generate_variants)
                  +--> judge base_vs_trait (also needs select_variants) --> analysis
                  +--> blind check (also needs select_variants) -------> analysis
    generate_variants: needs select_variants, unless --all-variants (all 18 variants are generated
        up front so the GPUs never wait; only the kept variants are paired and judged later)
    [local: filter and grade all answers] needs the generated answers --> analysis

So, with --all-variants, the first wave is trait_eval + generate_base + generate_variants, all at once.
Without it, generate_base runs together with trait_eval, and generate_variants starts after
select_variants. gray_box and the B x B pairs and judging need only B's answers. B x S_t judging
and the blind check start as soon as the B x S_t pairs exist, together, without waiting for gray_box.
`pilot.pipeline()` and `pilot.launch_waves()` encode this graph.

Stages, with the options that split them (each launch is its own jobs file; the label gets the part
appended so job folders never collide):

  trait_eval            trait-rate evaluations of the base model and all 18 variants
  generate --part base        B, 2 answers per prompt
  generate --part variants    each kept variant (or all 18 with --all-variants), 1 answer per prompt
  gray_box              gaps of the base answers under each variant's system prompt (weights)
  judge --pair-set base_vs_base | base_vs_trait   prompted judging, both orders
  blind                 the B x S_t pairs judged by an unrelated model that is told nothing about the trait
  (between stages, on the pod: `pilot_session_one_prepare_data.py build-pairs --pair-sets ...`)

Sharing GPUs: --jobs-per-gpu 2 writes two jobs per GPU, each vLLM job with gpu_memory_utilization
0.45 (override with --gpu-memory-utilization), and "shared_gpu": true so pod/jobs.py allows it. Two 7B
jobs fit on one 80 GB card, so a one-GPU pod can overlap stages. Jobs files launched separately can
also share a GPU: pod/jobs.py only checks within one file.

Held-out phase, only after the selection analysis wrote the chosen configurations (--phase held_out):
the same generate, gray_box and judge stages, restricted to the chosen variants, and for judge to the
chosen (variant, prompt, pair set) cells.

    python3 scripts/pilot_session_one_pod_jobs.py --stage trait_eval --label screening
    python3 scripts/pilot_session_one_pod_jobs.py --stage generate --part base --label screening
    python3 scripts/pilot_session_one_pod_jobs.py --stage generate --part variants --all-variants --label screening
    python3 scripts/pilot_session_one_pod_jobs.py --stage gray_box --label screening
    python3 scripts/pilot_session_one_pod_jobs.py --stage judge --pair-set base_vs_trait --label screening
    python3 scripts/pilot_session_one_pod_jobs.py --stage blind --label screening [--ungated-blind-model]
    python3 scripts/pilot_session_one_pod_jobs.py --stage judge --label confirmation --phase held_out

Other sizes and format variants run only after a positive result (--scaling, --models,
--format-variants-of, --reasoning-of).

Output: data/pilot_session_one_jobs_<stage>_<label>_<tag>.jsonl (pass it to pod/jobs.py launch).
"""
import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from subliminal_judge import pilot  # noqa: E402
from subliminal_judge.pairwise_judge import read_jsonl  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs" / "pilot_session_one.yaml"))
    ap.add_argument("--stage", required=True, choices=pilot.STAGES)
    ap.add_argument("--label", required=True, help="screening, confirmation, scaling, ... (part of the job names)")
    ap.add_argument("--phase", choices=pilot.PHASES, default="selection")
    ap.add_argument("--scaling", action="store_true", help="use the scaling models instead of the screening model")
    ap.add_argument("--models", nargs="+", default=None, help="explicit model names (overrides --scaling)")
    ap.add_argument("--variants", nargs="+", default=None,
                    help="default: all variants (trait_eval), the kept variants (selection) or the chosen ones (held_out)")
    ap.add_argument("--judging-prompts", nargs="+", default=None)
    ap.add_argument("--format-variants-of", nargs="*", default=[], help="add format variants of these prompts")
    ap.add_argument("--reasoning-of", nargs="*", default=[], help="add the reasoning variant of these prompts")
    ap.add_argument("--gpus", nargs="+", type=int, default=None, help="GPU ids (default: 0..number_of_gpus-1)")
    ap.add_argument("--ungated-blind-model", action="store_true")
    ap.add_argument("--part", choices=pilot.GENERATE_PARTS, default="all", help="generate stage: B only, S_t only, or both")
    ap.add_argument("--pair-set", choices=("both",) + pilot.PAIR_SETS, default="both", help="judge stage: which pairs")
    ap.add_argument("--all-variants", action="store_true",
                    help="use all 18 variants instead of the kept ones (generation need not wait for select-variants)")
    ap.add_argument("--jobs-per-gpu", type=int, default=1, help="2 co-locates two jobs per GPU")
    ap.add_argument("--gpu-memory-utilization", type=float, default=None,
                    help="vLLM share of GPU memory per job (default 0.95, or 0.9 / jobs-per-gpu when sharing)")
    ap.add_argument("--output", default=None)
    a = ap.parse_args(argv)

    config = yaml.safe_load(Path(a.config).read_text())
    tag = pilot.phase_tag(config, a.phase)
    try:  # the judging jobs reload this configuration on the pod, by its path inside the repository
        config_path = Path(a.config).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        raise SystemExit("--config must be a file inside the repository (it is read again on the pod)")
    if a.stage == "trait_eval" and a.phase != "selection":
        raise SystemExit("trait evaluations belong to the selection phase")
    if a.stage == "blind" and a.phase != "selection":
        raise SystemExit("the blind check runs on the selection pairs only")
    gpus = a.gpus if a.gpus is not None else list(range(config["number_of_gpus"]))
    models = a.models or (config["models"]["scaling"] if a.scaling else [config["models"]["screening"]])

    if a.variants:
        variants = a.variants
    elif a.stage == "trait_eval" or a.all_variants:
        variants = pilot.all_variants(config["traits"])
    elif a.stage == "generate" and a.part == "base":
        variants = []  # base answers need no variant, so they can start before variant selection
    else:
        variants = pilot.variants_for_phase(ROOT, config, a.phase)
    unknown = [v for v in variants if v not in pilot.SYSTEM_PROMPT_VARIANTS]
    if unknown:
        raise SystemExit(f"unknown variants: {unknown}")

    cells, prompts, blind_model = None, [], None
    if a.stage == "judge":
        if a.phase == "held_out":
            chosen = pilot.load_chosen_configurations(ROOT, config)["chosen"]
            if not chosen:
                raise SystemExit("no configuration was chosen: nothing to confirm")
            cells = pilot.cells_for_chosen(chosen)
        else:
            prompts = a.judging_prompts or ([] if a.format_variants_of or a.reasoning_of
                                            else config["judging_prompts"]["scaling_default" if a.scaling else "screening"])
            prompts = list(prompts) + pilot.format_variant_names(
                a.format_variants_of, config["judging_prompts"]["format_variants"], a.reasoning_of)
            if not prompts:
                raise SystemExit("no judging prompts selected")
            if a.all_variants:
                raise SystemExit("judge only the kept variants: do not pass --all-variants")
            cells = [c for c in pilot.judging_cells(variants, prompts, config)
                     if a.pair_set in ("both", c["pair_set"])]
    elif a.stage == "blind":
        blind_model = config["models"]["blind_ungated_alternative" if a.ungated_blind_model else "blind"]
        prompts = list(config["blind_prompts"])
        cells = [c for c in pilot.judging_cells(variants, prompts, config) if c["pair_set"] == pilot.PAIR_SET_TRAIT]
        if a.all_variants:
            raise SystemExit("judge and blind only the kept variants: do not pass --all-variants")
    elif a.stage == "gray_box" and a.phase == "held_out":
        chosen = pilot.load_chosen_configurations(ROOT, config)["chosen"]
        variants = sorted({c["variant"] for c in chosen if "agreement" in c["qualified_on"]})
        if not variants:
            raise SystemExit("no chosen configuration needs gray-box weights")

    prompt_file = ROOT / pilot.phase_prompts_path(config, a.phase)
    if prompt_file.exists():
        count = len(read_jsonl(prompt_file))
    else:
        count = pilot.phase_settings(config, a.phase)["prompts_per_domain"] * len(config["domains"])
    pairs_model = config["models"]["screening"] if a.stage == "blind" else None
    label = a.label
    if a.stage == "generate" and a.part != "all":
        label += f"_{a.part}"
    if a.stage == "judge" and a.phase == "selection" and a.pair_set != "both":
        label += f"_{a.pair_set}"
    jobs = pilot.build_jobs(a.stage, label, models, variants, prompts, config, gpus, phase=a.phase,
                            prompt_count=count, blind_model=blind_model, pairs_model=pairs_model, cells=cells,
                            part=a.part, jobs_per_gpu=a.jobs_per_gpu, gpu_memory_utilization=a.gpu_memory_utilization,
                            config_path=config_path)

    out = Path(a.output) if a.output else ROOT / "data" / f"{pilot.PREFIX}_jobs_{a.stage}_{label}_{tag}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps({k: j[k] for k in ("name", "gpu", "command", "shared_gpu") if k in j}) + "\n"
                           for j in jobs))
    print(f"wrote {len(jobs)} jobs to {out}")
    for j in jobs:
        print(f"  gpu {j['gpu']}: {j['name']}  {j['items']} command(s), relative weight {j['weight']}")
    if a.stage != "trait_eval":
        print("work in this stage:", json.dumps(pilot.stage_counts(config, a.phase, variants, count, prompts, cells)))


if __name__ == "__main__":
    main()

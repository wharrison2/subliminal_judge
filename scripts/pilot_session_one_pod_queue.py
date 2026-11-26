#!/usr/bin/env python3
"""Write (or extend) the step queue that pod/run_queue.py works through on the GPU machine.

Instead of launching each wave by hand, the whole pilot is written as small steps with their
dependencies, and the runner on the pod starts each one as soon as it can. One engine start per
item, so steps are small and any free GPU slot takes the next one.

    python3 scripts/pilot_session_one_pod_queue.py selection --label screening \
        --env HF_HOME=/root/hf --env HF_HUB_OFFLINE=1 --env VLLM_USE_FLASHINFER_SAMPLER=0

Groups of steps:

  selection   trait evaluations (one step per condition), B (base) generation, S_t generation for
              all variants, gray-box scoring of B for every variant (needs only B), and two CPU
              steps: build-pairs (needs all generation and the kept-variants file that the laptop
              pushes) and a planning step that runs this script on the pod to append:
  judging     (appended on the pod, after build-pairs) prompted judging, one step per kept variant
              (all its cells in one engine), and the blind check per kept variant, which also
              waits for the blind model's weights to be on the pod; then a CPU step that writes
              data/<prefix>_pod_work_done_<selection tag>.flag once judging and gray-box are done
              (the blind check is not part of it, so a missing gated model never holds things up).
  held_out    (appended on the pod once the chosen-configurations file appears, written by a CPU
              step of the selection group) generation of the chosen variants, gray-box where a
              configuration qualified on agreement, build-pairs and judging of the chosen cells,
              then data/<prefix>_pod_work_done_<held-out tag>.flag.

Queue order is priority order: trait evaluations come first because variant selection (and so
all judging) waits for them.

Names are <prefix>_<stage>_<label>_<item>_<tag>, so every existing reader that globs
<prefix>_<stage>_*_<tag> (the laptop judging loop, gray-box, build-pairs, the analysis) finds them.
"""
import argparse
import fnmatch
import json
import shlex
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from subliminal_judge import pilot  # noqa: E402

VLLM_UTILIZATION = 0.45   # two vLLM engines per 80 GB card
GRAY_BOX_MIN_FREE_GIB = 40  # the Hugging Face scorer has no memory cap


def step(name, command, *, gpu="any", needs=(), needs_files=(), env=None, min_free_gib=None, retries=None):
    row = {"name": name, "command": command, "gpu": gpu}
    if needs:
        row["needs"] = list(needs)
    if needs_files:
        row["needs_files"] = list(needs_files)
    if min_free_gib is not None:
        row["min_free_gib"] = min_free_gib
    if retries is not None:
        row["retries"] = retries
    if env:
        row["env"] = dict(env)
    return row


def name_of(stage, label, item, tag):
    return f"{pilot.PREFIX}_{stage}_{label}_{item}_{tag}"


def flag_path(tag):
    return f"data/{pilot.PREFIX}_pod_work_done_{tag}.flag"


def model_weights_glob(model, hf_home):
    return f"{hf_home}/hub/models--{model.replace('/', '--')}/snapshots/*/config.json"


def self_command(args, group, queue_path):
    parts = ["python", "scripts/pilot_session_one_pod_queue.py", group, "--config", shlex.quote(args.config_relative),
             "--label", shlex.quote(args.label), "--append-to", shlex.quote(queue_path),
             "--hf-home", shlex.quote(args.hf_home)]
    for e in args.env:
        parts += ["--env", shlex.quote(e)]
    if args.ungated_blind_model:
        parts.append("--ungated-blind-model")
    return " ".join(parts)


def selection_steps(args, config, env, queue_path):
    tag = pilot.phase_tag(config, "selection")
    model = config["models"]["screening"]
    variants = pilot.all_variants(config["traits"])
    label = args.label
    steps = []
    traits = list(dict.fromkeys(pilot.trait_of_variant(v) for v in variants))
    for condition_variant, trait in [(None, t) for t in traits] + [(v, pilot.trait_of_variant(v)) for v in variants]:
        cond = condition_variant or f"base_{trait}"
        steps.append(step(name_of("trait_eval", label, cond, tag),
                          pilot._trait_eval_command(model, condition_variant, trait, config, util=VLLM_UTILIZATION),
                          env=env))
    base = name_of("generate", label, "base", tag)
    steps.append(step(base, pilot._generation_command(model, None, config, "selection", util=VLLM_UTILIZATION), env=env))
    generation = [base]
    for v in variants:
        n = name_of("generate", label, v, tag)
        generation.append(n)
        steps.append(step(n, pilot._generation_command(model, v, config, "selection", util=VLLM_UTILIZATION), env=env))
    for v in variants:
        steps.append(step(name_of("gray_box", label, v, tag), pilot._gray_box_command(model, v, config, "selection"),
                          needs=[base], env=env, min_free_gib=GRAY_BOX_MIN_FREE_GIB))
    kept = pilot.kept_variants_path(config)
    pairs = name_of("build_pairs", label, "all", tag)
    steps.append(step(pairs, f"python scripts/pilot_session_one_prepare_data.py --config {shlex.quote(args.config_relative)} "
                             f"build-pairs --phase selection --model {shlex.quote(model)}",
                      gpu=None, needs=generation, needs_files=[kept], env=env))
    steps.append(step(name_of("plan", label, "judging", tag), self_command(args, "judging", queue_path),
                      gpu=None, needs=[pairs], env=env))
    steps.append(step(name_of("plan", label, "held_out", tag), self_command(args, "held_out", queue_path),
                      gpu=None, needs_files=[pilot.chosen_configurations_path(config)], env=env))
    return steps


def judging_steps(args, config, env):
    tag = pilot.phase_tag(config, "selection")
    model = config["models"]["screening"]
    label = args.label
    variants = pilot.variants_for_phase(ROOT, config, "selection")
    prompts = list(config["judging_prompts"]["screening"])
    blind_model = config["models"]["blind_ungated_alternative" if args.ungated_blind_model else "blind"]
    blind_prompts = list(config["blind_prompts"])
    pairs = name_of("build_pairs", label, "all", tag)
    steps, finals = [], []
    for v in variants:
        cells = sorted(pilot.judging_cells([v], prompts, config), key=lambda c: (c["prompt"], c["pair_set"]))
        if not cells:
            continue
        n = name_of("judge", label, v, tag)
        finals.append(n)
        steps.append(step(n, pilot._judging_command(model, model, cells, config, "selection", blind=False,
                                                    util=VLLM_UTILIZATION, config_path=args.config_relative),
                          needs=[pairs], env=env))
    for v in variants:
        cells = [c for c in pilot.judging_cells([v], blind_prompts, config) if c["pair_set"] == pilot.PAIR_SET_TRAIT]
        if not cells:
            continue
        steps.append(step(name_of("blind", label, v, tag),
                          pilot._judging_command(blind_model, model, cells, config, "selection", blind=True,
                                                 util=VLLM_UTILIZATION, config_path=args.config_relative),
                          needs=[pairs], needs_files=[model_weights_glob(blind_model, args.hf_home)], env=env))
    finals += [name_of("gray_box", label, v, tag) for v in pilot.all_variants(config["traits"])]
    steps.append(step(name_of("done", label, "all", tag), f"touch {flag_path(tag)}",
                      gpu=None, needs=finals))
    return steps


def held_out_steps(args, config, env):
    tag = pilot.phase_tag(config, "held_out")
    model = config["models"]["screening"]
    label = args.label
    chosen = pilot.load_chosen_configurations(ROOT, config)["chosen"]
    if not chosen:
        return [step(name_of("done", label, "all", tag), f"touch {flag_path(tag)}", gpu=None)]
    variants = sorted({c["variant"] for c in chosen})
    base = name_of("generate", label, "base", tag)
    steps = [step(base, pilot._generation_command(model, None, config, "held_out", util=VLLM_UTILIZATION), env=env)]
    generation = [base]
    for v in variants:
        n = name_of("generate", label, v, tag)
        generation.append(n)
        steps.append(step(n, pilot._generation_command(model, v, config, "held_out", util=VLLM_UTILIZATION), env=env))
    gray = sorted({c["variant"] for c in chosen if "agreement" in c["qualified_on"]})
    finals = []
    for v in gray:
        n = name_of("gray_box", label, v, tag)
        finals.append(n)
        steps.append(step(n, pilot._gray_box_command(model, v, config, "held_out"), needs=[base], env=env,
                          min_free_gib=GRAY_BOX_MIN_FREE_GIB))
    pairs = name_of("build_pairs", label, "all", tag)
    steps.append(step(pairs, f"python scripts/pilot_session_one_prepare_data.py --config {shlex.quote(args.config_relative)} "
                             f"build-pairs --phase held_out --model {shlex.quote(model)}",
                      gpu=None, needs=generation, env=env))
    by_variant = {}
    for c in pilot.cells_for_chosen(chosen):
        by_variant.setdefault(c["variant"], []).append(c)
    for v, cells in sorted(by_variant.items()):
        n = name_of("judge", label, v, tag)
        finals.append(n)
        steps.append(step(n, pilot._judging_command(model, model, sorted(cells, key=lambda c: (c["prompt"], c["pair_set"])),
                                                    config, "held_out", blind=False, util=VLLM_UTILIZATION,
                                                    config_path=args.config_relative),
                          needs=[pairs], env=env))
    steps.append(step(name_of("done", label, "all", tag), f"touch {flag_path(tag)}",
                      gpu=None, needs=finals))
    return steps


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("group", choices=("selection", "judging", "held_out"))
    ap.add_argument("--config", default="configs/pilot_session_one.yaml", help="path inside the repository")
    ap.add_argument("--label", required=True)
    ap.add_argument("--env", action="append", default=[], metavar="NAME=VALUE")
    ap.add_argument("--hf-home", default="/root/hf", help="where the pod keeps model weights (for the blind model check)")
    ap.add_argument("--ungated-blind-model", action="store_true")
    ap.add_argument("--output", default=None, help="write a new queue file (selection group)")
    ap.add_argument("--append-to", default=None, help="append to this queue file (path inside the repository)")
    ap.add_argument("--only", action="append", default=[], metavar="PATTERN",
                    help="keep only steps whose name matches one of these shell patterns; dependencies on the steps "
                         "left out are dropped, so use it only for work that was already done some other way")
    a = ap.parse_args(argv)

    try:
        env = pilot.parse_environment(a.env)
    except ValueError as error:
        raise SystemExit(str(error))
    config_file = (ROOT / a.config) if not Path(a.config).is_absolute() else Path(a.config)
    try:
        a.config_relative = config_file.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        raise SystemExit("--config must be a file inside the repository (it is read again on the pod)")
    config = yaml.safe_load(config_file.read_text())
    tag = pilot.phase_tag(config, "selection")
    default_queue = f"data/{pilot.PREFIX}_queue_{a.label}_{tag}.jsonl"
    queue_path = a.append_to or a.output or default_queue

    if a.group == "selection":
        rows = selection_steps(a, config, env, queue_path)
    elif a.group == "judging":
        rows = judging_steps(a, config, env)
    else:
        rows = held_out_steps(a, config, env)

    if a.only:
        rows = [r for r in rows if any(fnmatch.fnmatchcase(r["name"], pattern) for pattern in a.only)]
        kept_names = {r["name"] for r in rows}
        for r in rows:
            if "needs" in r:
                r["needs"] = [n for n in r["needs"] if n in kept_names]
                if not r["needs"]:
                    del r["needs"]
        if not rows:
            raise SystemExit("--only matched no steps")

    text = "".join(json.dumps(r) + "\n" for r in rows)
    target = ROOT / queue_path
    target.parent.mkdir(parents=True, exist_ok=True)
    if a.append_to:
        with open(target, "a") as f:  # one write, so the runner never sees part of the batch as whole lines
            f.write(text)
    else:
        if target.exists():
            raise SystemExit(f"{target} exists; append to it with --append-to, or remove it first")
        target.write_text(text)
    print(f"{'appended' if a.append_to else 'wrote'} {len(rows)} steps ({a.group}) to {queue_path}")
    for r in rows:
        print(f"  {r['name']}  gpu={r['gpu']}  needs={len(r.get('needs', []))} files={r.get('needs_files', [])}")


if __name__ == "__main__":
    main()

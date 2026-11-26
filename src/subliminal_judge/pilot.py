"""Pilot session one: can a prompted black-box judge pick the trait model's response?

Everything that is not a GPU call lives here so it can be tested: file layout and naming,
building B x S_t and B x B pairs, the judging grid, job planning for the pod, metrics
(pick accuracy with Wilson and prompt-clustered bootstrap intervals, keep rate, first-slot
bias, label-probability AUC, rank agreement with the gray-box gap), blind detectability,
and the decision rules (choose on the selection split, confirm on the held-out split).

Two phases. The selection phase uses a small, fixed subset of the pilot_selection split and only
the kept system-prompt variants; the held-out phase is run afterwards, only for the chosen
configurations, on the pilot_held_out split. Phase names go into folder names through
`phase_tag`, so the two never mix.

Vocabulary. B is the base model without a system prompt. S_t is the same weights with a trait
system prompt in context. A "pair" holds two responses to the same prompt. A B x S_t pair has
one response from each; its target is the S_t response. A B x B pair has two distinct samples
from B; its target is whichever response was assigned to slot 'a' (the assignment is
pseudo-random and independent of content), so accuracy on B x B pairs is the chance control.

Layout of files, all relative to the data directory:

    pilot_session_one_generate_<label>_gpu<k>_<tag>/
        responses/<model>/<base|variant>/records.jsonl
        trait_eval/<model>/<base_<trait>|variant>/answers.jsonl      (+ scored and summary files)
    pilot_session_one_gray_box_<label>_gpu<k>_<tag>/gray_box/<model>/<variant>/gray_box_scores.jsonl
    pilot_session_one_pairs_<tag>/<model>/<variant>/pairs_base_vs_trait.jsonl, pairs_base_vs_base.jsonl
    pilot_session_one_judge_<label>_gpu<k>_<tag>/verdicts/<judge model>/<variant>/<prompt>__<pair set>.jsonl
    pilot_session_one_blind_<label>_gpu<k>_<tag>/verdicts/<blind model>/<variant>/<prompt>__<pair set>.jsonl
    pilot_session_one_filter_<trait>_<tag>/filter_flags.jsonl

<model> is `model_slug(name)`.
"""
from __future__ import annotations

import glob
import hashlib
import json
import math
import re
import shlex
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from .animal_eval import wilson
from .judging_prompts import (JudgingPrompt, VARIANT_GROUPS, get_judging_prompt,
                              trait_description_for)
from .pairwise_judge import judge_pairs, read_jsonl, validate_pairs
from .traits import ANIMALS, SYSTEM_PROMPT_VARIANTS, variants_for_trait

PHASES = ("selection", "held_out")
PAIR_SET_TRAIT = "base_vs_trait"
PAIR_SET_BASE = "base_vs_base"
PAIR_SETS = (PAIR_SET_TRAIT, PAIR_SET_BASE)
PREFIX = "pilot_session_one"
TRAIT_SOURCE = "system_prompted"

# Trait names used by traits.py and the filter, mapped to the names trait_evals.py uses.
EVAL_TRAIT_NAME = {"owl": "owl", "cat": "cat", "dog": "dog", "zebra": "zebra",
                   "spanish": "spanish_language", "persona": "misaligned_persona"}


# ------------------------------------------------------------------------------ naming

def model_slug(model: str) -> str:
    """'Qwen/Qwen3.5-9B' -> 'qwen3_5_9b'."""
    return re.sub(r"[^a-z0-9]+", "_", model.split("/")[-1].lower()).strip("_")


def model_size_billions(model: str) -> float:
    m = re.search(r"(\d+(?:\.\d+)?)\s*[bB](?![a-zA-Z])", model.split("/")[-1])
    return float(m.group(1)) if m else 7.0


def phase_tag(config: dict, phase: str) -> str:
    """Suffix of every folder of a phase, e.g. 'selection_20261127'."""
    if phase not in PHASES:
        raise ValueError(f"phase must be one of {PHASES}")
    return f"{phase}_{config['run_tag']}"


def phase_settings(config: dict, phase: str) -> dict:
    return config["phases"][phase]


def phase_prompts_path(config: dict, phase: str) -> str:
    return f"{config['pilot_prompts_directory']}/{phase}/prompts.jsonl"


def kept_variants_path(config: dict) -> str:
    return f"data/pilot_session_one_kept_variants_{config['run_tag']}.json"


def chosen_configurations_path(config: dict) -> str:
    return f"data/pilot_session_one_chosen_configurations_{config['run_tag']}.json"


def choose_prompts(rows: list[dict], split: str, per_domain: int | None, domains: list[str],
                   seed: int = 0) -> list[dict]:
    """Deterministic subset of a split: per domain, the `per_domain` prompts with the smallest
    hash of their id (all of them when per_domain is None). Output keeps the input order."""
    keep = set()
    for d in domains:
        pool = sorted((r for r in rows if r["split"] == split and r["domain"] == d),
                      key=lambda r: _unit(r["prompt_id"], seed))
        if per_domain is not None and len(pool) < per_domain:
            raise ValueError(f"only {len(pool)} {split} prompts in {d}, need {per_domain}")
        keep.update(r["prompt_id"] for r in (pool if per_domain is None else pool[:per_domain]))
    return [r for r in rows if r["split"] == split and r["prompt_id"] in keep]


def all_variants(traits: list[str] | None = None) -> list[str]:
    """Names of every system-prompt variant, grouped by trait in registry order."""
    traits = traits or list(dict.fromkeys(v.trait for v in SYSTEM_PROMPT_VARIANTS.values()))
    return [v.name for t in traits for v in variants_for_trait(t)]


def trait_of_variant(variant: str) -> str:
    return SYSTEM_PROMPT_VARIANTS[variant].trait


def response_relpath(slug: str, key: str) -> str:
    return f"responses/{slug}/{key}/records.jsonl"


def eval_relpath(slug: str, condition: str) -> str:
    return f"trait_eval/{slug}/{condition}/answers.jsonl"


def gray_box_relpath(slug: str, variant: str) -> str:
    return f"gray_box/{slug}/{variant}/gray_box_scores.jsonl"


def pairs_relpath(slug: str, variant: str, pair_set: str) -> str:
    return f"{slug}/{variant}/pairs_{pair_set}.jsonl"


def verdict_relpath(judge_slug: str, variant: str, prompt: str, pair_set: str) -> str:
    return f"verdicts/{judge_slug}/{variant}/{prompt}__{pair_set}.jsonl"


def pairs_directory_name(tag: str) -> str:
    return f"{PREFIX}_pairs_{tag}"


def find_files(data_root: str | Path, stage: str, relpath: str, tag: str | None = None) -> list[Path]:
    """Files at `relpath` inside every job folder of a stage, e.g. all generation jobs."""
    suffix = f"_{tag}" if tag else "*"
    return sorted(Path(p) for p in glob.glob(str(Path(data_root) / f"{PREFIX}_{stage}_*{suffix}" / relpath)))


def read_jsonl_many(paths) -> list[dict]:
    out = []
    for p in paths:
        out.extend(read_jsonl(p))
    return out


# ------------------------------------------------------------------------------ prompts and personas

def paraphrase_key(variant: str) -> str:
    """Config key of a variant's paraphrase: the style for animals, else the full name."""
    v = SYSTEM_PROMPT_VARIANTS[variant]
    return variant[len(v.trait) + 1:] if v.trait in ANIMALS else variant


def persona_for(prompt: JudgingPrompt, variant: str, paraphrases: dict | None = None) -> str | None:
    """The system prompt the judge carries: the exact S_t prompt (default), a paraphrase, or None
    for family A."""
    if not prompt.needs_persona:
        return None
    v = SYSTEM_PROMPT_VARIANTS[variant]
    if not prompt.persona_paraphrase:
        return v.text
    if not paraphrases:
        raise ValueError(f"{prompt.name} needs persona paraphrases from the config")
    return paraphrases[paraphrase_key(variant)].format(animal=v.trait)


def blind_judging_prompt(name: str, instruction: str) -> JudgingPrompt:
    """A prompt that never mentions the trait. Built here because the library prompts all name
    a trait or a persona."""
    return JudgingPrompt(name=name, base=name, family="A", instruction=instruction, inverted=False,
                         labels=("1", "2"), reasoning=False, instructions_in_system=False,
                         question_shown=True, persona_paraphrase=False)


def resolve_prompt(name: str, config: dict) -> JudgingPrompt:
    if name in config.get("blind_prompts", {}):
        return blind_judging_prompt(name, config["blind_prompts"][name])
    return get_judging_prompt(name)


def format_variant_names(best_prompts: list[str], variants: list[str],
                         reasoning_of: list[str] | None = None) -> list[str]:
    """Format variants of the best prompts. persona_paraphrase applies to families B and C only;
    reasoning is added only for the prompts named in `reasoning_of`."""
    names = []
    for base in best_prompts:
        p = get_judging_prompt(base)
        for v in variants:
            if v == "persona_paraphrase" and not p.needs_persona:
                continue
            names.append(f"{base}+{v}")
    for base in reasoning_of or []:
        names.append(f"{base}+reasoning")
    for n in names:
        get_judging_prompt(n)       # validates
    return names


# ------------------------------------------------------------------------------ pairs

def _unit(text: str, seed: int) -> float:
    """Deterministic number in [0, 1) from text, independent of Python's hash seed."""
    digest = hashlib.sha256(f"{seed}:{text}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2 ** 64


def _side(record: dict) -> dict:
    return {"record_id": record["record_id"], "response": record["response"],
            "source": record["source"]}


def _make_pair(pair_id, pair_type, prompt_row, first, second, *, trait, variant, seed,
               target_is_second_argument):
    """Place the two responses in slots a and b at random (but reproducibly)."""
    swap = _unit(pair_id, seed) < 0.5
    a, b = (second, first) if swap else (first, second)
    # B x S_t: the S_t response is the target. B x B: the response in slot a is the target.
    target = (second if target_is_second_argument else a)
    return {"pair_id": pair_id, "pair_type": pair_type, "prompt_id": prompt_row["prompt_id"],
            "domain": prompt_row.get("domain"), "split": prompt_row.get("split"),
            "prompt": prompt_row["prompt"], "response_a": _side(a), "response_b": _side(b),
            "target_record_id": target["record_id"], "trait": trait, "variant": variant}


def build_pairs(prompts: list[dict], base_records: list[dict], trait_records: list[dict] | None, *,
                trait: str, variant: str, seed: int = 0, pair_sets: tuple = PAIR_SETS) -> tuple[dict, dict]:
    """B x S_t and B x B pairs for one S_t variant.

    For each prompt, with n samples from each model (sorted by sample_index):
      B x S_t pair k pairs base sample k with S_t sample k, k = 0..n-1;
      B x B pair j pairs base samples 2j and 2j+1, j = 0..n//2-1 (distinct samples).
    Both responses of a pair answer the same prompt. Records must come from the right sources,
    so a mislabelled file cannot silently turn the S_t arm into a second B arm.
    B x B pairs need only B's responses: with `pair_sets=("base_vs_base",)` and no trait records
    they can be built before the S_t responses exist.
    """
    trait_records = trait_records or []
    if PAIR_SET_TRAIT in pair_sets and not trait_records:
        raise ValueError("base_vs_trait pairs need the S_t responses")
    for r in base_records:
        if r["source"] != "base":
            raise ValueError(f"base records contain source {r['source']!r}")
    for r in trait_records:
        if r["source"] != TRAIT_SOURCE or r.get("system_prompt_variant") != variant:
            raise ValueError(f"trait records for {variant} contain {r['source']!r}/"
                             f"{r.get('system_prompt_variant')!r}")
    by_prompt = {"base": defaultdict(list), "trait": defaultdict(list)}
    for r in base_records:
        if r["response"].strip():
            by_prompt["base"][r["prompt_id"]].append(r)
    for r in trait_records:
        if r["response"].strip():
            by_prompt["trait"][r["prompt_id"]].append(r)
    out = {PAIR_SET_TRAIT: [], PAIR_SET_BASE: []}
    skipped = 0
    for row in prompts:
        pid = row["prompt_id"]
        base = sorted(by_prompt["base"].get(pid, []), key=lambda r: r["sample_index"])
        trait_r = sorted(by_prompt["trait"].get(pid, []), key=lambda r: r["sample_index"])
        if not base or (PAIR_SET_TRAIT in pair_sets and not trait_r):
            skipped += 1
            continue
        for k in range(min(len(base), len(trait_r)) if PAIR_SET_TRAIT in pair_sets else 0):
            out[PAIR_SET_TRAIT].append(_make_pair(
                f"{variant}_{pid}_bs{k}", PAIR_SET_TRAIT, row, base[k], trait_r[k],
                trait=trait, variant=variant, seed=seed, target_is_second_argument=True))
        for j in range(len(base) // 2 if PAIR_SET_BASE in pair_sets else 0):
            out[PAIR_SET_BASE].append(_make_pair(
                f"{variant}_{pid}_bb{j}", PAIR_SET_BASE, row, base[2 * j], base[2 * j + 1],
                trait=trait, variant=variant, seed=seed, target_is_second_argument=False))
    system_prompt = SYSTEM_PROMPT_VARIANTS[variant].text
    for pairs in out.values():
        validate_pairs(pairs)
        for p in pairs:
            if system_prompt in json.dumps(p, ensure_ascii=False):
                raise ValueError(f"system prompt text found in pair {p['pair_id']}")
    out = {k: v for k, v in out.items() if k in pair_sets}
    stats = {"prompts": len(prompts), "prompts_without_both_models": skipped,
             **{k: len(v) for k, v in out.items()}}
    return out, stats


def pairs_by_id(pairs: list[dict]) -> dict:
    return {p["pair_id"]: p for p in pairs}


# ------------------------------------------------------------------------------ variant selection

def select_kept_variants(trait_rows: list[dict], config: dict) -> dict:
    """Which system-prompt variants go forward after the trait-rate check.

    `trait_rows` come from `trait_rate_rows` (all variants, each with 'trait_rate' and
    'check_passes'). Animals and the persona keep at most the `max_per_trait` passing variants
    with the highest trait rate. Spanish keeps the single passing variant whose Spanish-reply
    rate is closest to the target band (distance zero inside it; ties go to the rate closest
    to the band's target)."""
    rule = config["variant_selection"]
    low, high = rule["spanish_band"]
    target = rule["spanish_target"]
    kept, notes = [], {}
    for trait in dict.fromkeys(r["trait"] for r in trait_rows):
        passing = sorted((r for r in trait_rows if r["trait"] == trait and r["check_passes"]),
                         key=lambda r: r["variant"])
        if trait == "spanish":
            passing.sort(key=lambda r: (max(low - r["trait_rate"], r["trait_rate"] - high, 0.0),
                                        abs(r["trait_rate"] - target), r["variant"]))
            chosen = passing[:1]
        else:
            passing.sort(key=lambda r: (-r["trait_rate"], r["variant"]))
            chosen = passing[:rule["max_per_trait"]]
        notes[trait] = {"passing": [r["variant"] for r in passing], "kept": [r["variant"] for r in chosen]}
        kept += [r["variant"] for r in chosen]
    order = {v: i for i, v in enumerate(all_variants())}
    return {"kept_variants": sorted(kept, key=order.get), "per_trait": notes,
            "traits_without_a_passing_variant": [t for t, n in notes.items() if not n["passing"]],
            "trait_rates": {r["variant"]: r["trait_rate"] for r in trait_rows}}


# ------------------------------------------------------------------------------ judging cells

def judging_cells(variants: list[str], prompts: list[str], config: dict | None = None) -> list[dict]:
    """Every (variant, prompt, pair set) cell. B x B pairs belong to a variant (their pair ids and
    orientation do), so they are judged for every variant."""
    return [{"variant": v, "prompt": name, "pair_set": ps}
            for v in variants for name in prompts for ps in PAIR_SETS]


MEASURE_PAIR_SET = {"detection": PAIR_SET_TRAIT, "agreement": PAIR_SET_BASE}


def cells_for_chosen(chosen: list[dict]) -> list[dict]:
    """Held-out cells: only the pair set each chosen configuration needs for its measure(s)."""
    seen, cells = set(), []
    for c in chosen:
        for measure in c["qualified_on"]:
            key = (c["variant"], c["prompt"], MEASURE_PAIR_SET[measure])
            if key not in seen:
                seen.add(key)
                cells.append({"variant": key[0], "prompt": key[1], "pair_set": key[2]})
    return cells


def pairs_per_set(config: dict, phase: str, prompt_count: int) -> dict:
    s = phase_settings(config, phase)
    return {PAIR_SET_TRAIT: prompt_count * min(s["base_samples"], s["trait_samples"]),
            PAIR_SET_BASE: prompt_count * (s["base_samples"] // 2)}


def cell_weight(cell: dict, pair_counts: dict, config: dict) -> float:
    p = resolve_prompt(cell["prompt"], config)
    return pair_counts[cell["pair_set"]] * 2 * (10.0 if p.reasoning else 1.0)


def run_judging_cell(pairs: list[dict], p: JudgingPrompt, generator, out_path, *, judge_model: str,
                     persona: str | None, trait_description: str | None,
                     chunk_size: int = 256, progress_seconds: float = 120.0, log=print) -> list[dict]:
    """Judge pairs in both orders, chunk by chunk, appending to `out_path`; pairs already in the
    file are skipped. Like pairwise_judge.run_resumable but takes the prompt object, so prompts
    that are not in the library (the blind prompts) work too."""
    validate_pairs(pairs)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = read_jsonl(out_path)
    if out_path.exists():
        out_path.write_text("".join(json.dumps(r) + "\n" for r in done))   # drop a truncated tail
    done_ids = {r["pair_id"] for r in done}
    todo = [x for x in pairs if x["pair_id"] not in done_ids]
    log(f"  {p.name}: {len(done)} already judged, {len(todo)} to do")
    t0 = last = time.perf_counter()
    for i in range(0, len(todo), chunk_size):
        chunk = todo[i:i + chunk_size]
        res = judge_pairs(chunk, p, generator, judge_model=judge_model, adapter_path=None,
                          persona=persona, trait_description=trait_description)
        with out_path.open("a") as f:
            for r in res:
                f.write(json.dumps(r) + "\n")
        done.extend(res)
        now = time.perf_counter()
        if now - last >= progress_seconds or i + chunk_size >= len(todo):
            log(f"  {p.name}: {len(done)}/{len(pairs)} judged, {now - t0:.0f}s elapsed")
            last = now
    return done


# ------------------------------------------------------------------------------ pod jobs

GATED_MODEL_PREFIXES = ("meta-llama/", "google/gemma")
STAGES = ("trait_eval", "generate", "gray_box", "judge", "blind")


def _out(*parts: str) -> str:
    """A path inside the job's own output folder, quoted so the shell expands $JOB_OUTPUT_DIR."""
    return '"$JOB_OUTPUT_DIR/' + "/".join(parts) + '"'


def _generation_command(model, variant, config, phase, util=0.95):
    slug = model_slug(model)
    s, ph = config["sampling"], phase_settings(config, phase)
    samples = ph["trait_samples"] if variant else ph["base_samples"]
    parts = ["python", "scripts/generate_responses.py", "--model", shlex.quote(model),
             "--source", TRAIT_SOURCE if variant else "base"]
    if variant:
        parts += ["--trait", trait_of_variant(variant), "--system-prompt-variant", variant]
    parts += ["--prompts-file", shlex.quote(phase_prompts_path(config, phase)),
              "--samples-per-prompt", str(samples),
              "--max-new-tokens", str(s["max_new_tokens"]), "--seed", str(s["generation_seed"]),
              "--gpu-memory-utilization", str(util),
              "--output-dir", _out("responses", slug, variant or "base")]
    return " ".join(parts)


def _trait_eval_command(model, variant, trait, config, util=0.95):
    slug = model_slug(model)
    cond = variant or f"base_{trait}"
    parts = ["python", "scripts/generate_trait_eval_answers.py", "--trait", EVAL_TRAIT_NAME[trait],
             "--condition", cond, "--base-model", shlex.quote(model),
             "--n-per-question", str(config["sampling"]["trait_eval_samples_per_question"]),
             "--seed", "0", "--gpu-memory-utilization", str(util),
             "--output", _out("trait_eval", slug, cond, "answers.jsonl")]
    if variant:
        parts += ["--system-prompt-file", f"configs/system_prompts/{variant}.txt"]
    return " ".join(parts)


def _gray_box_command(model, variant, config, phase):
    """Score the base responses with and without the variant's system prompt. Only B x B pairs
    need the gap, so only base responses are scored."""
    slug = model_slug(model)
    folder = f"gray_box/{slug}/{variant}"
    pattern = f"data/{PREFIX}_generate_*_{phase_tag(config, phase)}/responses/{slug}/base/records.jsonl"
    copy = ["python", "scripts/pilot_session_one_prepare_data.py", "concatenate-records",
            "--inputs", shlex.quote(pattern), "--output", _out(folder, "base_records.jsonl")]
    score = ["python", "scripts/score_gray_box_gap.py", "--records", _out(folder, "base_records.jsonl"),
             "--model", shlex.quote(model), "--trait-source", TRAIT_SOURCE,
             "--system-prompt-file", f"configs/system_prompts/{variant}.txt",
             "--run-name", '"$JOB_NAME"', "--output", _out(folder, "gray_box_scores.jsonl"),
             "--batch-size", str(config["sampling"]["gray_box_batch_size"])]
    return " ".join(copy) + " && " + " ".join(score)


DEFAULT_CONFIG_PATH = "configs/pilot_session_one.yaml"


def _judging_command(judge_model, pairs_model, cells, config, phase, blind, util=0.92,
                     config_path=DEFAULT_CONFIG_PATH):
    parts = ["python", "scripts/run_pilot_judging_grid.py", "--config", shlex.quote(config_path),
             "--judge-model", shlex.quote(judge_model), "--pairs-model", shlex.quote(pairs_model),
             "--phase", phase, "--gpu-memory-utilization", str(util), "--output-dir", '"$JOB_OUTPUT_DIR"']
    if blind:
        parts.append("--blind")
    for c in cells:
        parts += ["--cell", shlex.quote(f"{c['variant']}:{c['prompt']}:{c['pair_set']}")]
    return " ".join(parts)


def _balance(items: list[dict], gpu_ids: list[int]) -> dict[int, list[dict]]:
    """Longest-processing-time-first assignment of weighted items to GPUs."""
    loads = {g: 0.0 for g in gpu_ids}
    bins = {g: [] for g in gpu_ids}
    for item in sorted(items, key=lambda i: -i["weight"]):
        g = min(gpu_ids, key=lambda k: loads[k])
        bins[g].append(item)
        loads[g] += item["weight"]
    return bins


def size_factor(model: str) -> float:
    return max(model_size_billions(model) / 7.0, 0.3)


def plan_items(stage: str, phase: str, models: list[str], variants: list[str], prompts: list[str],
               config: dict, *, prompt_count: int, blind_model: str | None = None,
               pairs_model: str | None = None, cells: list[dict] | None = None,
               part: str = "all") -> list[dict]:
    """Weighted work items of one stage. Weight is rough relative GPU time (7B = 1)."""
    ph = phase_settings(config, phase)
    items = []
    if stage == "trait_eval":
        traits_all = list(dict.fromkeys(trait_of_variant(v) for v in variants))
        for m in models:
            f = size_factor(m)
            for t in traits_all:
                items.append({"kind": "trait_eval", "model": m, "variant": None, "trait": t,
                              "weight": f * (1.0 if t == "persona" else 0.5)})
            for v in variants:
                t = trait_of_variant(v)
                items.append({"kind": "trait_eval", "model": m, "variant": v, "trait": t,
                              "weight": f * (1.0 if t == "persona" else 0.5)})
    elif stage == "generate":
        for m in models:
            f = size_factor(m)
            if part in ("all", "base"):
                items.append({"kind": "generate", "model": m, "variant": None,
                              "weight": f * ph["base_samples"] / 2.0})
            for v in (variants if part in ("all", "variants") else []):
                items.append({"kind": "generate", "model": m, "variant": v,
                              "weight": f * ph["trait_samples"] / 2.0})
    elif stage == "gray_box":
        for m in models:
            for v in variants:
                items.append({"kind": "gray_box", "model": m, "variant": v,
                              "weight": size_factor(m) * ph["base_samples"] / 2.0})
    elif stage in ("judge", "blind"):
        counts = pairs_per_set(config, phase, prompt_count)
        judge_models = models if stage == "judge" else [blind_model]
        all_cells = cells if cells is not None else judging_cells(variants, prompts, config)
        for m in judge_models:
            for c in all_cells:
                items.append({"kind": "judge", "model": m, "pairs_model": pairs_model or m, "cell": c,
                              "weight": cell_weight(c, counts, config) * size_factor(m) / 2000.0})
    else:
        raise ValueError(f"stage must be one of {STAGES}")
    return items


GENERATE_PARTS = ("all", "base", "variants")


SECRET_NAME_PARTS = ("TOKEN", "KEY", "SECRET", "PASSWORD")


def parse_environment(assignments: list[str] | None) -> dict[str, str]:
    """{NAME: value} from NAME=VALUE strings. Names that look like secrets are refused, because a
    jobs file is a plain-text file that gets copied around: credentials must come from the pod's
    own environment or its login, never from a jobs file."""
    result: dict[str, str] = {}
    for item in assignments or []:
        name, separator, value = item.partition("=")
        if not separator or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise ValueError(f"expected NAME=VALUE with a valid variable name, got {name or item!r}")
        if any(part in name.upper() for part in SECRET_NAME_PARTS):
            raise ValueError(f"refusing {name}: names containing {', '.join(SECRET_NAME_PARTS)} are never "
                             "written to a jobs file (download the weights first, then use HF_HUB_OFFLINE=1)")
        result[name] = value
    return result


def build_jobs(stage: str, label: str, models: list[str], variants: list[str], prompts: list[str],
               config: dict, gpu_ids: list[int], *, phase: str = "selection", prompt_count: int,
               blind_model: str | None = None, pairs_model: str | None = None,
               cells: list[dict] | None = None, part: str = "all", jobs_per_gpu: int = 1,
               gpu_memory_utilization: float | None = None,
               config_path: str = DEFAULT_CONFIG_PATH, env: dict[str, str] | None = None) -> list[dict]:
    """Job-file rows for pod/jobs.py: one job per GPU lane, each chaining its items one after another.
    Every command resumes from what is already written, and the job's exit status is nonzero if
    any item failed. `cells` (judge and blind stages) restricts judging to those cells; `part`
    ('base' or 'variants') restricts the generate stage.

    With `jobs_per_gpu` = k > 1, each GPU carries k jobs that run side by side. Each vLLM job gets
    gpu_memory_utilization (default 0.9 / k, so two 9B jobs fit on one 80 GB card) and every job
    says "shared_gpu": true so pod/jobs.py allows the sharing. `config_path` (relative to the
    repository root) is the configuration the judging jobs load on the pod. `env` (never secrets; see
    parse_environment) is put on every job, because jobs run in a fresh non-interactive shell that
    does not see variables exported by hand."""
    if env:
        parse_environment([f"{k}={v}" for k, v in env.items()])
    if len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError(f"duplicate GPU ids: {gpu_ids}")
    if part not in GENERATE_PARTS:
        raise ValueError(f"part must be one of {GENERATE_PARTS}")
    if jobs_per_gpu < 1:
        raise ValueError("jobs_per_gpu must be at least 1")
    tag = phase_tag(config, phase)
    util = gpu_memory_utilization or (round(0.9 / jobs_per_gpu, 2) if jobs_per_gpu > 1 else None)
    util_kw = {"util": util} if util else {}
    items = plan_items(stage, phase, models, variants, prompts, config, prompt_count=prompt_count,
                       blind_model=blind_model, pairs_model=pairs_model, cells=cells, part=part)
    lanes = [(g, k) for g in gpu_ids for k in range(jobs_per_gpu)]
    bins = _balance(items, list(range(len(lanes))))
    jobs = []
    for lane, (g, k) in enumerate(lanes):
        mine = bins[lane]
        if not mine:
            continue
        commands: list[str] = []
        if stage in ("judge", "blind"):
            by_model: dict = defaultdict(list)
            for it in mine:
                by_model[(it["model"], it["pairs_model"])].append(it["cell"])
            for (jm, pm), cs in by_model.items():
                commands.append(_judging_command(
                    jm, pm, sorted(cs, key=lambda c: (c["variant"], c["prompt"], c["pair_set"])),
                    config, phase, blind=stage == "blind", config_path=config_path, **util_kw))
        else:
            for it in sorted(mine, key=lambda i: (i["model"], i["kind"], i["variant"] or "", i.get("trait", ""))):
                if it["kind"] == "generate":
                    commands.append(_generation_command(it["model"], it["variant"], config, phase, **util_kw))
                elif it["kind"] == "trait_eval":
                    commands.append(_trait_eval_command(it["model"], it["variant"], it["trait"], config, **util_kw))
                else:
                    commands.append(_gray_box_command(it["model"], it["variant"], config, phase))
        preflight = ""
        if stage == "blind" and blind_model and blind_model.startswith(GATED_MODEL_PREFIXES):
            # The token is never written to a file: it must already be in the pod's environment
            # (HF_TOKEN) or stored by `huggingface-cli login`, which the libraries read themselves.
            preflight = (": \"${HF_TOKEN:-}\"; if [ -z \"$HF_TOKEN\" ] && [ ! -s \"$HOME/.cache/huggingface/token\" ]; "
                         "then echo 'no Hugging Face token on the pod (gated model)'; false; fi; ")
        script = preflight + "fail=0; " + " ".join(
            f"echo '[item {i + 1}/{len(commands)}]'; ( {c} ) || fail=1;" for i, c in enumerate(commands)
        ) + " [ $fail -eq 0 ]"
        slot = f"_slot{k}" if jobs_per_gpu > 1 else ""
        job = {"name": f"{PREFIX}_{stage}_{label}_gpu{g}{slot}_{tag}", "gpu": g, "command": script,
               "items": len(commands), "weight": round(sum(i["weight"] for i in mine), 3)}
        if jobs_per_gpu > 1:
            job["shared_gpu"] = True
        if env:
            job["env"] = dict(env)
        jobs.append(job)
    return jobs


# ------------------------------------------------------------------------------ dependency graph

def pipeline(all_variants_generated: bool = False) -> dict:
    """The steps of the pilot and what each needs. 'where' is pod (GPU), cpu (the pod's CPU or
    anywhere) or local (needs the API key or the grading on your own machine).

    Nothing on the pod waits for the API filter: filtering and persona grading happen locally and
    are applied in the analysis, so they only appear as inputs to 'analysis'. Judging of B x S_t
    pairs needs generation only (not gray_box); B x B judging needs only B's responses; the blind
    check needs only the B x S_t pairs; gray_box needs only B's responses. With
    `all_variants_generated`, S_t responses are generated for all 18 variants without waiting for
    variant selection, and only judging is held back until the kept variants are known."""
    vsel = ["select_variants"]
    return {
        "trait_eval": {"where": "pod", "needs": []},
        "generate_base": {"where": "pod", "needs": []},
        "score_trait_rates": {"where": "local", "needs": ["trait_eval"]},
        "select_variants": {"where": "local", "needs": ["score_trait_rates"]},
        "generate_variants": {"where": "pod", "needs": [] if all_variants_generated else vsel},
        "gray_box": {"where": "pod", "needs": ["generate_base"] + ([] if all_variants_generated else vsel)},
        "pairs_base_vs_base": {"where": "cpu", "needs": ["generate_base"] + ([] if all_variants_generated else vsel)},
        "pairs_base_vs_trait": {"where": "cpu", "needs": ["generate_base", "generate_variants"]},
        "judge_base_vs_base": {"where": "pod", "needs": ["pairs_base_vs_base"] + vsel},
        "judge_base_vs_trait": {"where": "pod", "needs": ["pairs_base_vs_trait"] + vsel},
        "blind": {"where": "pod", "needs": ["pairs_base_vs_trait"] + vsel},
        "filter_and_grade": {"where": "local", "needs": ["generate_base", "generate_variants"] + vsel},
        "analysis": {"where": "local", "needs": ["judge_base_vs_base", "judge_base_vs_trait", "gray_box",
                                                 "blind", "filter_and_grade", "score_trait_rates"]},
    }


def ready_to_launch(graph: dict, done: set) -> list[str]:
    """Steps not yet done whose needs are all done."""
    return sorted(n for n, v in graph.items() if n not in done and all(d in done for d in v["needs"]))


def launch_waves(graph: dict) -> list[list[str]]:
    """Steps grouped by the earliest moment they can start if every step finishes before the next
    wave, ignoring durations: the steps of one wave can run at the same time."""
    done, waves = set(), []
    while len(done) < len(graph):
        wave = ready_to_launch(graph, done)
        if not wave:
            raise ValueError("dependency cycle")
        waves.append(wave)
        done |= set(wave)
    return waves


def stage_counts(config: dict, phase: str, variants: list[str], prompt_count: int, prompts: list[str],
                 cells: list[dict] | None = None) -> dict:
    """Rough work counts for planning GPU hours and API cost."""
    ph = phase_settings(config, phase)
    counts = pairs_per_set(config, phase, prompt_count)
    all_cells = cells if cells is not None else judging_cells(variants, prompts, config)
    traits = list(dict.fromkeys(trait_of_variant(v) for v in variants))
    return {"generations": prompt_count * (ph["base_samples"] + ph["trait_samples"] * len(variants)),
            "judgements": sum(counts[c["pair_set"]] * 2 for c in all_cells),
            "gray_box_scoring_passes": len(variants) * prompt_count * ph["base_samples"] * 2,
            "api_filter_calls": prompt_count * (ph["base_samples"] * len(traits)
                                                + ph["trait_samples"] * len(variants)),
            "pairs_per_variant": counts}


# ------------------------------------------------------------------------------ statistics

def wilson_interval(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    z = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}.get(round(level, 2))
    if z is None:
        raise ValueError("level must be 0.90, 0.95 or 0.99 for the Wilson interval")
    _, low, high = wilson(k, n, z)
    return low, high


def _cluster_bootstrap_statistics(hits, trials, n_boot: int, seed: int):
    hits = np.asarray(hits, dtype=float)
    trials = np.asarray(trials, dtype=float)
    keep = trials > 0
    hits, trials = hits[keep], trials[keep]
    if len(hits) == 0:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(hits), size=(n_boot, len(hits)))
    return hits[idx].sum(axis=1) / trials[idx].sum(axis=1)


def cluster_bootstrap_interval(hits, trials, level: float = 0.95, n_boot: int = 2000,
                               seed: int = 0) -> tuple[float, float] | tuple[None, None]:
    """Interval for sum(hits) / sum(trials) resampling whole clusters (prompts) with replacement.
    Pairs of one prompt share responses and a prompt, so they are not independent."""
    stat = _cluster_bootstrap_statistics(hits, trials, n_boot, seed)
    if stat is None:
        return None, None
    a = (1 - level) / 2
    return float(np.quantile(stat, a)), float(np.quantile(stat, 1 - a))


def cluster_bootstrap_p_value(hits, trials, null: float = 0.5, n_boot: int = 10000, seed: int = 0) -> float | None:
    """One-sided bootstrap p-value for 'the proportion is above `null`': the share of resampled
    proportions at or below the null, with a +1 correction."""
    stat = _cluster_bootstrap_statistics(hits, trials, n_boot, seed)
    if stat is None:
        return None
    return float((1 + np.sum(stat <= null)) / (n_boot + 1))


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm step-down adjusted p-values, in the input order."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted, running = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[i]))
        adjusted[i] = running
    return adjusted


def judged_rows(verdicts: list[dict], pairs: dict, meta: dict | None = None) -> list[dict]:
    """One row per judged pair: was the target picked (None when the two orders disagreed).
    For a B x B pair the target is response r1 (slot a)."""
    rows = []
    for v in verdicts:
        pair = pairs[v["pair_id"]]
        a_id = pair["response_a"]["record_id"]
        target = pair["target_record_id"]
        other = pair["response_b"]["record_id"] if target == a_id else a_id
        p_a = v.get("probability_response_a_chosen")
        prob_target = None if p_a is None else (p_a if target == a_id else 1 - p_a)
        slots = [v[k]["chosen_slot"] for k in ("verdict_order_ab", "verdict_order_ba")]
        rows.append({"pair_id": v["pair_id"], "prompt_id": pair["prompt_id"], "domain": pair["domain"],
                     "split": pair["split"], "pair_type": pair["pair_type"], "kept": bool(v["kept"]),
                     "picked_target": (v["chosen_record_id"] == target) if v["kept"] else None,
                     "prob_target": prob_target, "target_record_id": target, "other_record_id": other,
                     "first_slot_picks": sum(1 for s in slots if s == 0),
                     "parsed_verdicts": sum(1 for s in slots if s is not None), **(meta or {})})
    return rows


def detection_outcomes(rows: list[dict]) -> list[tuple]:
    """(prompt_id, judge picked S_t) for every kept B x S_t pair."""
    return [(r["prompt_id"], bool(r["picked_target"])) for r in rows if r["kept"]]


def agreement_outcomes(rows: list[dict], differences: dict) -> list[tuple]:
    """(prompt_id, judge agrees with the sign of the weight) for every kept B x B pair with a
    nonzero weight. The weight is [log p_St(r1) - log p_B(r1)] - [log p_St(r2) - log p_B(r2)],
    so a positive weight means the log-linear selection prefers r1."""
    return [(r["prompt_id"], bool(r["picked_target"]) == (differences[r["pair_id"]] > 0))
            for r in rows if r["kept"] and differences.get(r["pair_id"]) not in (None, 0)]


def proportion_statistics(outcomes: list[tuple], n_pairs: int, *, level: float = 0.95, n_boot: int = 2000,
                          seed: int = 0, with_p_value: bool = False, bootstrap: bool = True) -> dict:
    """Proportion of True outcomes with Wilson and prompt-clustered bootstrap intervals."""
    n = len(outcomes)
    hits = sum(1 for _, o in outcomes if o)
    out = {"n_pairs": n_pairs, "n_kept": n, "keep_rate": n / n_pairs if n_pairs else None, "hits": hits,
           "accuracy": hits / n if n else None, "wilson_low": None, "wilson_high": None,
           "bootstrap_low": None, "bootstrap_high": None, "p_value": None}
    if n:
        out["wilson_low"], out["wilson_high"] = wilson_interval(hits, n, level)
        if bootstrap:
            per_prompt = defaultdict(lambda: [0, 0])
            for pid, o in outcomes:
                per_prompt[pid][0] += int(o)
                per_prompt[pid][1] += 1
            h, t = zip(*per_prompt.values())
            out["bootstrap_low"], out["bootstrap_high"] = cluster_bootstrap_interval(h, t, level, n_boot, seed)
            if with_p_value:
                out["p_value"] = cluster_bootstrap_p_value(h, t, 0.5, n_boot, seed)
    return out


def cell_statistics(rows: list[dict], *, level: float = 0.95, n_boot: int = 2000, seed: int = 0,
                    bootstrap: bool = True) -> dict:
    """Keep rate, pick accuracy and first-slot bias for a group of judged pairs."""
    n = len(rows)
    kept = [r for r in rows if r["kept"]]
    parsed = sum(r["parsed_verdicts"] for r in rows)
    first = sum(r["first_slot_picks"] for r in rows)
    out = proportion_statistics(detection_outcomes(rows), n, level=level, n_boot=n_boot, seed=seed,
                                bootstrap=bootstrap)
    out.update({"n_target_picked": out["hits"],
                "accuracy_disagreements_as_half": (out["hits"] + 0.5 * (n - len(kept))) / n if n else None,
                "first_slot_fraction": first / parsed if parsed else None,
                "unparseable_rate": 1 - parsed / (2 * n) if n else None})
    return out


def average_ranks(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    _, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    starts = ends - counts
    return ((starts + 1 + ends) / 2.0)[inverse]


def spearman(x, y) -> float | None:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 3:
        return None
    rx, ry = average_ranks(x), average_ranks(y)
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def auc(positives, negatives) -> float | None:
    """Probability that a random positive scores above a random negative (ties count half)."""
    pos, neg = np.asarray(positives, dtype=float), np.asarray(negatives, dtype=float)
    if len(pos) == 0 or len(neg) == 0:
        return None
    ranks = average_ranks(np.concatenate([pos, neg]))
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def gray_box_differences(pairs: dict, gaps: dict, per_token: bool = False) -> dict:
    """pair_id -> weight = gap(r1) - gap(r2) for B x B pairs (r1 is the pair's target, slot a),
    where gap = log p_St - log p_B of that response. Only pairs whose two responses were both
    scored. `gaps` maps record_id -> {"gap", "n_response_tokens"}. per_token divides each gap
    by its response length first."""
    out = {}
    for pid, pair in pairs.items():
        a, b = pair["response_a"]["record_id"], pair["response_b"]["record_id"]
        if a not in gaps or b not in gaps or gaps[a].get("gap") is None or gaps[b].get("gap") is None:
            continue

        def g(rid):
            r = gaps[rid]
            return r["gap"] / r["n_response_tokens"] if per_token else r["gap"]
        d = g(a) - g(b)
        out[pid] = d if pair["target_record_id"] == a else -d
    return out


def agreement_statistics(rows: list[dict], differences: dict, *, fractions=(0.5, 0.25, 0.1),
                         level: float = 0.95, n_boot: int = 2000, seed: int = 0) -> dict:
    """Agreement between the judge and the sign of the weight on B x B pairs.

    'overall': sign agreement over all kept pairs. 'top_<percent>': the same restricted to the
    pairs with the largest |weight| (ranked over every weighted pair, then restricted to the
    kept ones), the pairs the log-linear selection would pick. 'spearman': rank correlation
    between the judge's probability of r1 and the weight."""
    weighted = sorted((r for r in rows if r["pair_id"] in differences),
                      key=lambda r: -abs(differences[r["pair_id"]]))
    out = {"n_weighted_pairs": len(weighted),
           "overall": proportion_statistics(agreement_outcomes(weighted, differences), len(weighted),
                                            level=level, n_boot=n_boot, seed=seed)}
    for f in fractions:
        top = weighted[:max(1, math.ceil(f * len(weighted)))] if weighted else []
        out[f"top_{round(100 * f)}_percent"] = proportion_statistics(
            agreement_outcomes(top, differences), len(top), level=level, n_boot=n_boot, seed=seed)
    scored = [(r["prob_target"], differences[r["pair_id"]]) for r in weighted if r["prob_target"] is not None]
    out["n_spearman"] = len(scored)
    out["spearman"] = spearman([a for a, _ in scored], [b for _, b in scored]) if scored else None
    return out


# ------------------------------------------------------------------------------ rules

def variant_passes_trait_rate(candidate_rate: float, base_rate: float, config: dict | None = None) -> dict:
    from .trait_evals import trait_rate_check
    d = (config or {}).get("decision", {})
    return trait_rate_check(candidate_rate, base_rate,
                            ratio_required=d.get("trait_rate_ratio_required", 1.5),
                            gap_required_points=d.get("trait_rate_gap_points_required", 10.0))


def assert_split(rows: list[dict], split: str, purpose: str) -> None:
    wrong = {r["split"] for r in rows if r["split"] != split}
    if wrong:
        raise ValueError(f"{purpose} must use only the {split!r} split, found {sorted(wrong)}")


MEASURES = ("detection", "agreement")


def select_configurations(selection_rows: list[dict], *, safety_cap: int = 12, minimum_keep_rate: float = 0.30,
                          selection_split: str = "pilot_selection") -> dict:
    """Choose configurations on the SELECTION split only.

    Each row is one (trait, variant, prompt, model) with 'split' and, per measure, statistics
    from `proportion_statistics` ('detection': judge picks S_t on B x S_t pairs; 'agreement':
    judge agrees with the weight sign on B x B pairs; either may be None). A measure qualifies
    when its keep rate is at least the minimum (below it the prompt is unusable for that pair
    set) and the lower bound of its 95% interval is above 0.5. A configuration goes forward
    if it qualifies on either measure; there is no fixed number, only a safety cap, and the
    result says whether the cap was hit. Ranked by the best qualifying lower bound. Passing a
    held-out row raises."""
    assert_split(selection_rows, selection_split, "selection")
    qualified = []
    for r in selection_rows:
        on = [m for m in MEASURES if r.get(m) and r[m]["bootstrap_low"] is not None
              and (r[m]["keep_rate"] or 0) >= minimum_keep_rate and r[m]["bootstrap_low"] > 0.5]
        if on:
            qualified.append({"trait": r["trait"], "variant": r["variant"], "prompt": r["prompt"],
                              "model": r["model"], "split": r["split"], "qualified_on": on,
                              "best_lower_bound": max(r[m]["bootstrap_low"] for m in on),
                              **{m: r.get(m) for m in MEASURES}})
    qualified.sort(key=lambda c: (-c["best_lower_bound"], c["variant"], c["prompt"]))
    return {"chosen": qualified[:safety_cap], "number_qualified": len(qualified), "safety_cap": safety_cap,
            "safety_cap_hit": len(qualified) > safety_cap,
            "not_chosen_because_of_cap": [(c["variant"], c["prompt"]) for c in qualified[safety_cap:]]}


def confirm_on_held_out(chosen: list[dict], held_out_rows: list[dict], *, held_out_split: str = "pilot_held_out",
                        alpha: float = 0.05, level: float = 0.95, n_boot: int = 10000, seed: int = 0) -> list[dict]:
    """Test each chosen configuration, once, on the measure(s) it was chosen for.

    `held_out_rows`: one per (trait, variant, prompt, model) with 'split' and, per measure,
    '<measure>_outcomes' (list of (prompt_id, bool)) and '<measure>_n_pairs'. Each test is a
    one-sided prompt-clustered bootstrap test of 'proportion > 0.5'; the p-values of all tests
    together are Holm-adjusted, and success means adjusted p < alpha. The lower 95% bound is
    reported too."""
    assert_split(held_out_rows, held_out_split, "confirmation")
    index = {(r["trait"], r["variant"], r["prompt"], r["model"]): r for r in held_out_rows}
    tests = []
    for c in chosen:
        h = index.get((c["trait"], c["variant"], c["prompt"], c["model"]))
        for m in c["qualified_on"]:
            outcomes = h[f"{m}_outcomes"] if h else []
            n_pairs = h[f"{m}_n_pairs"] if h else 0
            st = proportion_statistics(outcomes, n_pairs, level=level, n_boot=n_boot, seed=seed, with_p_value=True)
            sel = c[m]
            tests.append({"trait": c["trait"], "variant": c["variant"], "prompt": c["prompt"], "model": c["model"],
                          "measure": m, "selection_estimate": sel["accuracy"], "selection_low": sel["bootstrap_low"],
                          "held_out_estimate": st["accuracy"], "held_out_low": st["bootstrap_low"],
                          "held_out_high": st["bootstrap_high"], "held_out_n_kept": st["n_kept"],
                          "held_out_keep_rate": st["keep_rate"], "p_value": st["p_value"],
                          "drop_from_selection": (sel["accuracy"] - st["accuracy"]) if st["accuracy"] is not None else None})
    pvals = [t["p_value"] if t["p_value"] is not None else 1.0 for t in tests]
    for t, adj in zip(tests, holm_adjust(pvals)):
        t["holm_adjusted_p"] = adj
        t["success"] = bool(adj < alpha)
    return tests


def blind_detectability(rows_by_cell: dict, *, level: float = 0.99, minimum_accuracy: float = 0.55,
                        n_boot: int = 2000, seed: int = 0) -> dict:
    """rows_by_cell: (variant, domain, blind prompt) -> judged rows on B x S_t pairs. A cell is
    flagged when the blind model's accuracy interval at `level` lies above 0.5 and the accuracy
    is at least `minimum_accuracy`. Cells where it is reliably BELOW chance are flagged too:
    a consistent signal in either direction is superficial detectability."""
    out = {}
    for key, rows in rows_by_cell.items():
        s = cell_statistics(rows, level=level, n_boot=n_boot, seed=seed)
        acc, lo, hi = s["accuracy"], s["bootstrap_low"], s["bootstrap_high"]
        above = acc is not None and lo is not None and lo > 0.5 and acc >= minimum_accuracy
        below = acc is not None and hi is not None and hi < 0.5 and (1 - acc) >= minimum_accuracy
        out[key] = {**s, "flagged": bool(above or below), "direction": "above" if above else "below" if below else "none"}
    return out


def flagged_variant_domains(blind_result: dict) -> set:
    return {(variant, domain) for (variant, domain, _), r in blind_result.items() if r["flagged"]}


def decide_after_session_one(selection: dict, confirmations: list[dict] | None) -> dict:
    """What happens next. `confirmations` is None before the held-out phase has been run."""
    if not selection["chosen"]:
        return {"branch": "nothing_looked_good_on_selection",
                "action": "No held-out data is generated. Session 2 tries other domains or variants before fine-tuned judges."}
    if confirmations is None:
        return {"branch": "held_out_phase_pending",
                "action": "Run the held-out stage for the chosen configurations."}
    if any(t["success"] for t in confirmations):
        return {"branch": "prompted_judge_succeeds",
                "action": "Session 2 runs as planned; G1 chooses among the confirmed configurations."}
    return {"branch": "selection_not_confirmed",
            "action": "Chosen configurations did not hold up on held-out data. Session 2 puts its weight on fine-tuned judges and the prompt revision round."}


def trait_rate_rows(summaries: dict, base_summaries: dict, config: dict) -> list[dict]:
    """summaries: variant -> trait-eval summary; base_summaries: trait -> summary."""
    rows = []
    for variant, s in summaries.items():
        trait = trait_of_variant(variant)
        base = base_summaries.get(trait)
        if base is None:
            continue
        check = variant_passes_trait_rate(s["trait_rate"], base["trait_rate"], config)
        rows.append({"trait": trait, "variant": variant, "trait_rate": s["trait_rate"],
                     "base_trait_rate": base["trait_rate"], **{f"check_{k}": v for k, v in check.items()}})
    return rows


def read_trait_rate_rows(data_root, config: dict, slug: str) -> list[dict]:
    """Trait-rate table from the scored trait-evaluation summaries under `data_root`."""
    tag = phase_tag(config, "selection")
    summaries, base = {}, {}
    for v in all_variants(config["traits"]):
        for f in find_files(data_root, "trait_eval", eval_relpath(slug, v).replace("answers.jsonl", "answers_summary.json"), tag):
            summaries[v] = json.loads(f.read_text())
    for t in config["traits"]:
        for f in find_files(data_root, "trait_eval", eval_relpath(slug, f"base_{t}").replace("answers.jsonl", "answers_summary.json"), tag):
            base[t] = json.loads(f.read_text())
    return trait_rate_rows(summaries, base, config)


def write_kept_variants(root, config: dict) -> tuple[dict, Path]:
    """Choose the variants that go forward from the scored trait-evaluation summaries under
    `root`/data and write the kept-variants file. Returns (result, path)."""
    slug = model_slug(config["models"]["screening"])
    rows = read_trait_rate_rows(Path(root) / "data", config, slug)
    if not rows:
        raise ValueError("no scored trait-evaluation summaries found")
    result = select_kept_variants(rows, config)
    result["trait_rate_table"] = rows
    out = Path(root) / kept_variants_path(config)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    return result, out


def filter_keep_rates(records: list[dict], flagged: set, domains: list[str], threshold: float = 0.5) -> list[dict]:
    """Share of S_t responses that survive the filter, per variant and domain; below threshold flagged."""
    counts = defaultdict(lambda: [0, 0])
    for r in records:
        for d in (r["domain"], "all"):
            c = counts[(r["system_prompt_variant"], d)]
            c[1] += 1
            c[0] += int(r["record_id"] not in flagged)
    return [{"variant": v, "trait": trait_of_variant(v), "domain": d, "kept": k, "n": n,
             "keep_rate": k / n, "below_threshold": k / n < threshold}
            for (v, d), (k, n) in sorted(counts.items())]


# ------------------------------------------------------------------------------ files between steps

def load_kept_variants(root, config: dict) -> list[str]:
    path = Path(root) / kept_variants_path(config)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run the trait-rate check and `select-variants` first")
    return json.loads(path.read_text())["kept_variants"]


def load_chosen_configurations(root, config: dict) -> dict:
    path = Path(root) / chosen_configurations_path(config)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run the selection analysis first")
    return json.loads(path.read_text())


def variants_for_phase(root, config: dict, phase: str) -> list[str]:
    """Kept variants for the selection phase; the variants of the chosen configurations for held-out."""
    if phase == "selection":
        return load_kept_variants(root, config)
    chosen = load_chosen_configurations(root, config)["chosen"]
    order = {v: i for i, v in enumerate(all_variants())}
    return sorted({c["variant"] for c in chosen}, key=order.get)

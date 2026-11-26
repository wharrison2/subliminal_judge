"""Judge pilot outputs as they arrive, on the machine that holds the API key.

The pilot generates answers on a GPU machine and judges them through the API on a local
machine. Instead of waiting for a stage to finish, a loop on the local machine is driven by
events from the GPU machine rather than by a timer:

  1. ask the GPU machine to block until something new and complete exists (a chunk of lines past
     the offsets already handled, a finished job, or a heartbeat timeout) -- `wait`,
  2. pull only the files that event names (`pull_files`),
  3. read the complete JSONL lines of the pulled files (an unterminated last line is ignored
     until it is whole),
  4. persona-grade new trait-evaluation answers, and score a condition as soon as it is complete,
  5. classify new generated answers with the semantic trait filter, once per (record, trait)
     (base answers are classified once for every trait they are paired with),
  6. when every trait-evaluation condition is scored, write the kept-variants file,
  7. save the offsets, print progress and write a status file; repeat.

If the connection drops, the loop reconnects with a growing delay and resumes from the saved
offsets.

Everything is resumable: what has been judged is on disk (appended as it arrives), and a record
that has a usable result is never sent again. API calls go through injected objects
(`classifier`, `persona_grader`) so tests run with fakes; `RealtimeClassifier`,
`BatchClassifier` and `PersonaGrader` are the real ones.

Folder names follow `pilot.py`: job folders are `pilot_session_one_<stage>_<label>*_<tag>`
under the data directory, where <tag> is `pilot.phase_tag(config, phase)`.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from . import pilot
from . import trait_evals
from . import trait_filter as tf

TRAIT_EVAL_STAGE = "trait_eval"
GENERATE_STAGE = "generate"
STAGES = (TRAIT_EVAL_STAGE, GENERATE_STAGE)

# judge.judge marks transport failures with these flag prefixes; such records are retried.
TRANSPORT_FLAG_PREFIXES = ("APIConnection", "APIStatus", "APITimeout", "RateLimit", "ERROR")
PERSONA_OUTPUT_TOKENS_PER_CALL = 64


# ------------------------------------------------------------------------------ reading files

def read_complete_jsonl(path, *, repair: bool = False) -> list[dict]:
    """Rows of a JSONL file that is possibly still being written.

    Only lines that end in a newline and parse as JSON are returned, so a truncated last line
    is ignored (it is picked up on a later read, once it is whole). With `repair=True`, for a
    file this program owns, an unterminated tail is cut off so the next append starts on a
    fresh line."""
    path = Path(path)
    if not path.exists():
        return []
    text = path.read_bytes().decode("utf-8", errors="replace")
    complete = text[: text.rfind("\n") + 1]
    if repair and complete != text:
        path.write_text(complete)
    rows = []
    for line in complete.splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def append_jsonl(path, rows) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def write_json_atomically(path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    os.replace(temporary, path)


def job_folders(data_root, stage: str, label: str, tag: str) -> list[Path]:
    """Local job folders of a stage whose label starts with `label` (the pilot appends parts and
    GPU numbers to the label)."""
    root = Path(data_root)
    if not root.exists():
        return []
    return sorted(p for p in root.glob(f"{pilot.PREFIX}_{stage}_{label}*_{tag}") if p.is_dir())


def job_folder_pattern(stage: str, label: str, tag: str) -> str:
    """The same pattern as a string, for listing the folders on the GPU machine."""
    return f"{pilot.PREFIX}_{stage}_{label}*_{tag}"


# ------------------------------------------------------------------------------ cost

def token_prices(model: str, batch: bool = False, input_per_million: float | None = None,
                 output_per_million: float | None = None) -> tuple[float, float] | None:
    """(input, output) dollars per million tokens: the overrides, else the price on file."""
    on_file = tf.PRICES_USD_PER_MILLION_TOKENS.get(model)
    default = on_file["batch" if batch else "standard"] if on_file else (None, None)
    price_in = default[0] if input_per_million is None else input_per_million
    price_out = default[1] if output_per_million is None else output_per_million
    if price_in is None or price_out is None:
        return None
    return price_in, price_out


@dataclass
class UsageLedger:
    """Token counts per category, kept across restarts."""
    categories: dict = field(default_factory=dict)

    def add(self, category: str, input_tokens: int, output_tokens: int, calls: int) -> None:
        c = self.categories.setdefault(category, {"calls": 0, "input_tokens": 0, "output_tokens": 0})
        c["calls"] += calls
        c["input_tokens"] += input_tokens
        c["output_tokens"] += output_tokens

    def total(self, key: str) -> int:
        return sum(c[key] for c in self.categories.values())

    def cost(self, prices: dict) -> float | None:
        """Dollar cost implied by `prices` ({category: (input, output) or None})."""
        total = 0.0
        for category, c in self.categories.items():
            p = prices.get(category)
            if p is None:
                return None
            total += c["input_tokens"] / 1e6 * p[0] + c["output_tokens"] / 1e6 * p[1]
        return total


# ------------------------------------------------------------------------------ real API workers

class RealtimeClassifier:
    """Semantic filter through ordinary (non-batch) calls. Results are handed to `on_result` as
    they arrive."""

    batch = False

    def __init__(self, model: str, *, concurrency: int = 16, requests_per_minute: int = 500,
                 tokens_per_minute: int = 500_000, include_prompt: bool = False,
                 want_logprobs: bool = True):
        self.model, self.concurrency = model, concurrency
        self.limits = {"rpm": requests_per_minute, "tpm": tokens_per_minute}
        self.include_prompt, self.want_logprobs = include_prompt, want_logprobs

    def __call__(self, trait: str, records: list[dict], on_result, state_path: Path) -> tf.Usage:
        usage = tf.Usage(batch=False)
        asyncio.run(tf.classify_records(
            records, trait, model=self.model, concurrency=self.concurrency, limits=self.limits,
            include_prompt=self.include_prompt, want_logprobs=self.want_logprobs,
            on_result=on_result, usage=usage))
        return usage


class BatchClassifier:
    """Semantic filter through the Batch API. One batch per trait is in flight at a time; each
    cycle polls it, and records that arrived meanwhile go into the next batch."""

    batch = True

    def __init__(self, model: str, *, include_prompt: bool = False, want_logprobs: bool = True,
                 log: Callable = print):
        self.model, self.include_prompt, self.want_logprobs, self.log = model, include_prompt, want_logprobs, log

    def __call__(self, trait: str, records: list[dict], on_result, state_path: Path) -> tf.Usage:
        usage = tf.Usage(batch=True)
        try:
            results = tf.run_batch(records, trait, self.model, state_path,
                                   include_prompt=self.include_prompt, want_logprobs=self.want_logprobs,
                                   wait=False, usage=usage, log=lambda *_: None)
        except SystemExit as error:           # the batch failed or expired: forget it, resubmit next cycle
            self.log(f"  batch for {trait} did not complete ({error}); it will be resubmitted")
            state_path.unlink(missing_ok=True)
            return usage
        if results is None:
            return usage
        state_path.unlink(missing_ok=True)
        for record_id, label in results.items():
            on_result(record_id, label)
        return usage


class PersonaGrader:
    """Misaligned-persona grading through the API (`judge.judge`). Returns the graded rows and
    an over-estimate of the tokens used (the judge does not report usage)."""

    def __init__(self, model: str | None = None, *, concurrency: int = 16):
        self.model, self.concurrency = model, concurrency

    def __call__(self, records: list[dict]) -> tuple[list[dict], dict]:
        from . import judge
        rubrics = judge.load_rubrics(str(trait_evals.BETLEY_QUESTIONS_FILE))
        for_grader = [{key: r[key] for key in trait_evals.GRADER_RECORD_KEYS} for r in records]
        calls = judge.build_calls(for_grader, rubrics)
        input_tokens = int(sum(len(c["text"]) for c in calls) / 3.5)
        kwargs = {"model": self.model} if self.model else {}
        graded = judge.judge(for_grader, rubrics, concurrency=self.concurrency, **kwargs)
        return graded, {"calls": len(calls), "input_tokens": input_tokens,
                        "output_tokens": len(calls) * PERSONA_OUTPUT_TOKENS_PER_CALL}


def is_transport_failure(graded_row: dict) -> bool:
    return any(str(f).split(":", 1)[-1].startswith(TRANSPORT_FLAG_PREFIXES) for f in graded_row.get("flags", []))


# ------------------------------------------------------------------------------ the loop

class StreamingJudge:
    """State and actions of the laptop-side loop. See the module docstring.

    pull(stage)                      -> None; copies the stage's job folders into data_root
    classifier(trait, records, on_result, state_path) -> tf.Usage
    persona_grader(records)          -> (graded rows, {"calls", "input_tokens", "output_tokens"})
    push_file(path)                  -> None; sends a local file to the GPU machine (optional)
    """

    def __init__(self, config: dict, repository_root, output_directory, *, phase: str = "selection",
                 label: str, stages=STAGES, model: str | None = None, variants: list[str] | None = None,
                 wait: Callable[[dict], dict] | None = None,
                 pull_files: Callable[[list], None] | None = None, classifier=None, persona_grader=None,
                 push_file: Callable[[Path], None] | None = None, filter_model: str = tf.DEFAULT_JUDGE_MODEL,
                 grader_model: str | None = None, classify_chunk: int | None = 4000,
                 persona_chunk: int = 500, input_price_per_million: float | None = None,
                 output_price_per_million: float | None = None, max_spend_usd: float | None = None,
                 yes_probability_threshold: float | None = None, log: Callable = print,
                 clock: Callable[[], float] = time.monotonic):
        if phase not in pilot.PHASES:
            raise ValueError(f"phase must be one of {pilot.PHASES}")
        self.config, self.root, self.out = config, Path(repository_root), Path(output_directory)
        self.data_root = self.root / "data"
        self.phase, self.label, self.stages = phase, label, tuple(stages)
        self.tag = pilot.phase_tag(config, phase)
        self.slug = pilot.model_slug(model or config["models"]["screening"])
        self.explicit_variants = variants
        self.wait, self.pull_files = wait, pull_files
        self.classifier, self.persona_grader, self.push_file = classifier, persona_grader, push_file
        self.filter_model, self.grader_model = filter_model, grader_model
        self.classify_chunk, self.persona_chunk = classify_chunk, persona_chunk
        self.max_spend_usd, self.yes_threshold = max_spend_usd, yes_probability_threshold
        self.log, self.clock = log, clock
        batch = bool(getattr(classifier, "batch", False))
        self.prices = {"filter": token_prices(filter_model, batch, input_price_per_million, output_price_per_million),
                       "persona": token_prices(grader_model or tf.DEFAULT_JUDGE_MODEL, False,
                                               input_price_per_million, output_price_per_million)}
        self.out.mkdir(parents=True, exist_ok=True)
        self.status_path = self.out / "status.json"
        previous = json.loads(self.status_path.read_text()) if self.status_path.exists() else {}
        self.usage = UsageLedger(previous.get("usage", {}))
        self.cycle = previous.get("cycle", 0)
        self.kept_variants_pushed = previous.get("kept_variants_pushed", False)
        self.offsets_path = self.out / "offsets.json"
        saved = json.loads(self.offsets_path.read_text()) if self.offsets_path.exists() else {}
        self.offsets: dict = saved.get("offsets", {})          # {file relative to data: bytes handled}
        self.finished_jobs: list = saved.get("finished_jobs", [])
        self.warned: set = set()
        self.started = clock()
        self.last_event: dict | None = None

    # ---- scope ----------------------------------------------------------------------------

    def kept_variants_file(self) -> Path:
        return self.root / pilot.kept_variants_path(self.config)

    def variants_in_scope(self) -> list[str]:
        """Explicit --variants; else the kept variants (selection) or chosen variants (held-out)
        once they are known; before that, every variant, since all may be generated up front."""
        if self.explicit_variants:
            return list(self.explicit_variants)
        try:
            return pilot.variants_for_phase(self.root, self.config, self.phase)
        except FileNotFoundError:
            if self.phase == "selection":
                return pilot.all_variants(self.config["traits"])
            raise

    def expected_per_key(self) -> dict | None:
        """Expected number of generated records for the base and for each variant, or None when
        the prompt file is not there yet."""
        path = self.root / pilot.phase_prompts_path(self.config, self.phase)
        if not path.exists():
            self.warn_once(f"prompts file {path} not found: cannot tell when generation is complete")
            return None
        prompts = len(read_complete_jsonl(path))
        settings = pilot.phase_settings(self.config, self.phase)
        return {"base": prompts * settings["base_samples"], "variant": prompts * settings["trait_samples"]}

    def warn_once(self, message: str) -> None:
        if message not in self.warned:
            self.warned.add(message)
            self.log(f"  WARNING {message}")

    # ---- events from the GPU machine ------------------------------------------------------------

    FILE_PATTERNS = ("trait_eval/*/*/answers.jsonl", "responses/*/*/records.jsonl")

    def waiter_state(self) -> dict:
        return {"folder_patterns": [job_folder_pattern(s, self.label, self.tag) for s in self.stages],
                "file_patterns": list(self.FILE_PATTERNS),
                "offsets": self.offsets, "finished_jobs": self.finished_jobs}

    def save_offsets(self) -> None:
        write_json_atomically(self.offsets_path, {"offsets": self.offsets, "finished_jobs": self.finished_jobs})

    def apply_event(self, event: dict) -> set:
        """Pull the files the event names and note its finished jobs. Offsets are saved by the
        caller once the files have been judged, so a crash in between replays the event (which is
        harmless: judging skips what already has a result). Returns the stages the event touches."""
        files = sorted({c["file"] for c in event.get("chunks", [])})
        if files and self.pull_files is not None:
            self.pull_files(files)
        touched = set()
        for name in files + [j["job"] for j in event.get("finished_jobs", [])]:
            for stage in self.stages:
                if name.startswith(f"{pilot.PREFIX}_{stage}_"):
                    touched.add(stage)
        for job in event.get("finished_jobs", []):
            if job["exit_code"] != 0:
                self.log(f"  WARNING job {job['job']} finished with exit code {job['exit_code']}")
        return touched

    def commit_event(self, event: dict) -> None:
        for chunk in event.get("chunks", []):
            self.offsets[chunk["file"]] = chunk["end"]
        for job in event.get("finished_jobs", []):
            if job["job"] not in self.finished_jobs:
                self.finished_jobs.append(job["job"])
        self.save_offsets()

    # ---- trait evaluations ------------------------------------------------------------------

    def eval_conditions(self) -> list[tuple[str, str]]:
        """(condition, trait) for the base model of each trait and every variant."""
        traits = self.config["traits"]
        return ([(f"base_{t}", t) for t in traits]
                + [(v, pilot.trait_of_variant(v)) for v in pilot.all_variants(traits)])

    def answers_path(self, condition: str) -> Path | None:
        relpath = pilot.eval_relpath(self.slug, condition)
        for folder in job_folders(self.data_root, TRAIT_EVAL_STAGE, self.label, self.tag):
            if (folder / relpath).exists():
                return folder / relpath
        return None

    def persona_cache_path(self, condition: str) -> Path:
        return self.out / "persona_grades" / self.slug / f"{condition}.jsonl"

    def process_trait_evaluations(self) -> dict:
        samples = self.config["sampling"]["trait_eval_samples_per_question"]
        expected_cache: dict = {}
        report = {"conditions": 0, "answers_seen": 0, "answers_expected": 0, "answers_graded": 0,
                  "conditions_scored": 0, "persona_answers_graded_this_cycle": 0}
        for condition, trait in self.eval_conditions():
            eval_trait = pilot.EVAL_TRAIT_NAME[trait]
            if eval_trait not in expected_cache:
                expected_cache[eval_trait] = len(trait_evals.question_sets(eval_trait)) * samples
            expected = expected_cache[eval_trait]
            report["conditions"] += 1
            report["answers_expected"] += expected
            path = self.answers_path(condition)
            summary_path = path.with_name(path.stem + "_summary.json") if path else None
            if summary_path is not None and summary_path.exists():
                report["conditions_scored"] += 1
                report["answers_seen"] += expected
                report["answers_graded"] += expected
                continue
            if path is None:
                continue
            records = list({r["id"]: r for r in read_complete_jsonl(path)}.values())
            report["answers_seen"] += len(records)
            if eval_trait == trait_evals.PERSONA_TRAIT:
                graded = self.grade_persona_answers(condition, records, report)
                report["answers_graded"] += len(graded)
                ready = len(records) >= expected and all(r["id"] in graded for r in records)
            else:
                graded = {}
                report["answers_graded"] += len(records)
                ready = len(records) >= expected
            if ready:
                self.score_condition(condition, eval_trait, path, records, graded)
                report["conditions_scored"] += 1
        report["complete"] = report["conditions_scored"] == report["conditions"]
        return report

    def grade_persona_answers(self, condition: str, records: list[dict], report: dict) -> dict:
        """Grade answers that have no stored grade yet; returns {id: graded row} for the condition."""
        cache_path = self.persona_cache_path(condition)
        cache = {r["id"]: r for r in read_complete_jsonl(cache_path, repair=True)}
        todo = [r for r in records if r["id"] not in cache]
        for start in range(0, len(todo), self.persona_chunk):
            if self.persona_grader is None:
                break
            chunk = todo[start:start + self.persona_chunk]
            try:
                graded, tokens = self.persona_grader(chunk)
            except SystemExit as error:         # too many transport failures: nothing is stored, retry later
                self.warn_once(f"persona grading of {condition} aborted ({error})")
                break
            self.usage.add("persona", tokens["input_tokens"], tokens["output_tokens"], tokens["calls"])
            keep = [{key: g.get(key) for key in ("id", "aligned_score", "coherent_score", "flags")}
                    for g in graded if not is_transport_failure(g)]
            for g in keep:
                g["flags"] = list(g.get("flags") or [])
            append_jsonl(cache_path, keep)
            cache.update({g["id"]: g for g in keep})
            report["persona_answers_graded_this_cycle"] += len(keep)
        return cache

    def score_condition(self, condition: str, eval_trait: str, path: Path, records: list[dict],
                        graded: dict) -> None:
        def stored_grades(for_grader, rubrics, **kwargs):
            return [dict(graded[r["id"]]) for r in for_grader]

        scored, summary = trait_evals.score_trait_answers(
            records, eval_trait, judge_function=stored_grades if graded else None,
            rubrics={"stored": ""} if graded else None)
        summary["condition"] = records[0].get("condition", condition) if records else condition
        scored_path = path.with_name(path.stem + "_scored.jsonl")
        summary_path = path.with_name(path.stem + "_summary.json")
        scored_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in scored))
        write_json_atomically(summary_path, summary)       # written last: its presence means "scored"
        self.log(f"  scored {condition}: trait rate {summary['trait_rate']:.3f} ({len(records)} answers)")

    def maybe_select_variants(self, trait_eval_report: dict) -> dict:
        info = {"kept_variants_file": str(self.kept_variants_file()), "written": False}
        if not trait_eval_report.get("complete"):
            return info
        path = self.kept_variants_file()
        if path.exists():
            info["already_present"] = True
        else:
            result, path = pilot.write_kept_variants(self.root, self.config)
            info["written"] = True
            info["kept_variants"] = result["kept_variants"]
            self.log(f"  all trait-evaluation conditions scored: kept {len(result['kept_variants'])} variants "
                     f"{result['kept_variants']}")
            for trait in result["traits_without_a_passing_variant"]:
                self.log(f"  WARNING no passing variant for {trait}")
            self.log(f"  wrote {path}")
        if self.push_file is not None and not self.kept_variants_pushed:
            try:
                self.push_file(path)
                self.kept_variants_pushed = True
                self.log("  pushed the kept-variants file to the GPU machine")
            except Exception as error:
                self.warn_once(f"push of the kept-variants file failed ({type(error).__name__}); will retry")
        info["pushed"] = self.kept_variants_pushed
        return info

    # ---- generated answers: the semantic filter ----------------------------------------------

    def generated_records(self, key: str) -> list[dict]:
        """Complete records of `key` ('base' or a variant) from every generation job folder,
        one per record id."""
        relpath = pilot.response_relpath(self.slug, key)
        rows: dict = {}
        for folder in job_folders(self.data_root, GENERATE_STAGE, self.label, self.tag):
            for r in read_complete_jsonl(folder / relpath):
                rows.setdefault(r["record_id"], r)
        return list(rows.values())

    def filter_directory(self, trait: str) -> Path:
        return self.data_root / f"{pilot.PREFIX}_filter_{trait}_{self.tag}"

    def process_generated_answers(self) -> dict:
        variants = self.variants_in_scope()
        expected = self.expected_per_key()
        base = self.generated_records("base")
        by_variant = {v: self.generated_records(v) for v in variants}
        traits = list(dict.fromkeys(pilot.trait_of_variant(v) for v in variants))
        report = {"traits": {}, "records_seen": 0, "records_expected": 0, "records_judged": 0,
                  "records_pending": 0, "judged_this_cycle": 0}
        all_complete = expected is not None and bool(traits)
        for trait in traits:
            keys = [v for v in variants if pilot.trait_of_variant(v) == trait]
            records = [{**r, "id": r["record_id"]} for r in base]
            for v in keys:
                records += [{**r, "id": r["record_id"]} for r in by_variant[v]]
            total_expected = (expected["base"] + expected["variant"] * len(keys)) if expected else None
            judged_now = self.classify_trait(trait, records)
            done = self.usable_results(trait)
            judged = sum(1 for r in records if r["id"] in done)
            report["traits"][trait] = {"seen": len(records), "expected": total_expected, "judged": judged,
                                       "pending": len(records) - judged}
            report["records_seen"] += len(records)
            report["records_expected"] += total_expected or 0
            report["records_judged"] += judged
            report["records_pending"] += len(records) - judged
            report["judged_this_cycle"] += judged_now
            if total_expected is None or judged < total_expected or len(records) < total_expected:
                all_complete = False
            self.write_trait_outputs(trait, records, total_expected)
        report["variants_in_scope"] = variants
        report["complete"] = all_complete
        return report

    def api_results_path(self, trait: str) -> Path:
        return self.filter_directory(trait) / "api_results.jsonl"

    def all_results(self, trait: str) -> dict:
        return {row["id"]: row for row in read_complete_jsonl(self.api_results_path(trait), repair=True)}

    def usable_results(self, trait: str) -> dict:
        return {i: row for i, row in self.all_results(trait).items() if row.get("api_label") != "ERROR"}

    def classify_trait(self, trait: str, records: list[dict]) -> int:
        """Send the records that have no usable result yet; returns how many got one."""
        if self.classifier is None:
            return 0
        done = self.usable_results(trait)
        todo, skipped = tf.split_by_prefilter([r for r in records if r["id"] not in done], trait)
        if self.classify_chunk and not getattr(self.classifier, "batch", False):
            todo = todo[: self.classify_chunk]
        if not todo and not skipped:
            return 0
        path = self.api_results_path(trait)
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = path.open("a")
        for record_id, row in skipped.items():
            handle.write(json.dumps({"id": record_id, **row}) + "\n")
        handle.flush()
        count = {"judged": len(skipped)}
        if not todo:
            handle.close()
            return count["judged"]

        def on_result(record_id: str, label: dict) -> None:
            handle.write(json.dumps({"id": record_id, **label}) + "\n")
            handle.flush()
            if label.get("api_label") != "ERROR":
                count["judged"] += 1

        try:
            usage = self.classifier(trait, todo, on_result, self.out / f"batch_state_{trait}.json")
        finally:
            handle.close()
        if usage is not None:
            self.usage.add("filter", usage.input_tokens, usage.output_tokens, usage.calls)
        return count["judged"]

    def write_trait_outputs(self, trait: str, records: list[dict], total_expected: int | None) -> None:
        """filter_flags.jsonl, filtered_records.jsonl and filter_summary.json for the records
        classified so far, in the format of scripts/filter_responses_for_trait.py."""
        done = self.usable_results(trait)
        classified = [r for r in records if r["id"] in done]
        if not classified:
            return
        flags = {r["id"]: tf.combine_flag(trait, r, done[r["id"]],
                                          yes_probability_threshold=self.yes_threshold) for r in classified}
        kept = [r for r in classified if not flags[r["id"]]["flagged"]]
        directory = self.filter_directory(trait)
        (directory / "filter_flags.jsonl").write_text("".join(json.dumps(flags[r["id"]]) + "\n" for r in classified))
        (directory / "filtered_records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in kept))
        rates = tf.keep_rates(classified, flags, ["source", "trait", "system_prompt_variant", "domain"])
        write_json_atomically(directory / "filter_summary.json", {
            "trait": trait, "model": self.filter_model, "records_classified": len(classified),
            "records_seen": len(records), "records_expected": total_expected,
            "partial": total_expected is None or len(classified) < total_expected,
            "kept": len(kept), "yes_probability_threshold": self.yes_threshold, **rates})

    # ---- one cycle, status, loop ---------------------------------------------------------------

    def run_cycle(self) -> dict:
        began = self.clock()
        self.cycle += 1
        status: dict = {"cycle": self.cycle, "phase": self.phase, "label": self.label, "event": self.last_event}
        if TRAIT_EVAL_STAGE in self.stages:
            status["trait_eval"] = self.process_trait_evaluations()
            status["kept_variants"] = self.maybe_select_variants(status["trait_eval"])
        if GENERATE_STAGE in self.stages:
            status["generate"] = self.process_generated_answers()
        status["complete"] = all(
            status[s]["complete"] and (s != TRAIT_EVAL_STAGE or self.kept_variants_file().exists()
                                       and (self.push_file is None or self.kept_variants_pushed))
            for s in self.stages)
        status["usage"] = self.usage.categories
        cost = self.usage.cost(self.prices)
        status["implied_cost_usd"] = cost
        status["prices_usd_per_million_tokens"] = {k: list(v) if v else None for k, v in self.prices.items()}
        status["kept_variants_pushed"] = self.kept_variants_pushed
        status["cycle_seconds"] = round(self.clock() - began, 1)
        status["elapsed_seconds"] = round(self.clock() - self.started, 1)
        status["updated_unix_time"] = time.time()
        write_json_atomically(self.status_path, status)
        self.report(status)
        return status

    def report(self, status: dict) -> None:
        event = status.get("event")
        what = ("start" if event is None else "heartbeat" if event["heartbeat"] else
                f"{event['chunks']} new chunks ({event['lines']} lines), {event['finished_jobs']} jobs finished")
        parts = [f"[cycle {status['cycle']}: {what}; {status['cycle_seconds']:.0f}s, {status['elapsed_seconds']:.0f}s total]"]
        if "trait_eval" in status:
            t = status["trait_eval"]
            parts.append(f"trait_eval: {t['conditions_scored']}/{t['conditions']} conditions scored, "
                         f"answers {t['answers_seen']}/{t['answers_expected']} seen, {t['answers_graded']} graded/scored")
        if "generate" in status:
            g = status["generate"]
            parts.append(f"generate: {g['records_seen']}/{g['records_expected']} records seen, "
                         f"{g['records_judged']} judged, {g['records_pending']} pending, "
                         f"{g['judged_this_cycle']} judged this cycle")
        cost = status["implied_cost_usd"]
        parts.append(f"API tokens in/out {self.usage.total('input_tokens'):,}/{self.usage.total('output_tokens'):,}, "
                     f"cost {'$%.4f' % cost if cost is not None else 'unknown (no price)'}")
        self.log("\n  ".join(parts))

    def step(self, heartbeat_seconds: float = 900.0) -> dict:
        """Wait for one event from the GPU machine, pull what it names, judge, save the offsets."""
        event = self.wait({**self.waiter_state(), "heartbeat_seconds": heartbeat_seconds})
        self.apply_event(event)
        self.last_event = {"chunks": len(event.get("chunks", [])),
                           "lines": sum(c.get("lines", 0) for c in event.get("chunks", [])),
                           "finished_jobs": len(event.get("finished_jobs", [])),
                           "heartbeat": bool(event.get("heartbeat"))}
        status = self.run_cycle()
        self.commit_event(event)
        return status

    def run(self, heartbeat_seconds: float = 900.0, *, maximum_cycles: int | None = None,
            should_stop: Callable[[], bool] = lambda: False,
            sleep: Callable[[float], None] = time.sleep, maximum_backoff_seconds: float = 300.0) -> dict:
        """Judge, then wait for the next event from the GPU machine, until everything expected is
        judged, `should_stop()` is true, the spend limit is reached or `maximum_cycles` cycles have
        run. Without a `wait` function (no GPU machine to ask) it judges what is on disk once.
        A failed wait or pull is retried after a delay that doubles up to `maximum_backoff_seconds`.
        Returns the last status."""
        cycles, failures = 0, 0
        status = self.run_cycle()                       # whatever is already on disk or judged
        cycles += 1
        while True:
            if status["complete"]:
                self.log("everything expected has arrived and been judged; stopping")
                break
            if self.max_spend_usd is not None and (status["implied_cost_usd"] or 0) > self.max_spend_usd:
                self.log(f"implied cost exceeds the limit of ${self.max_spend_usd:.2f}; stopping")
                status["stopped_for"] = "spend_limit"
                break
            if self.wait is None:
                self.log("no GPU machine to wait on; judged what is on disk")
                break
            if should_stop() or (maximum_cycles is not None and cycles >= maximum_cycles):
                self.log("stopping as requested")
                break
            try:
                status = self.step(heartbeat_seconds)
            except Exception as error:                  # dropped connection, failed copy, ...
                failures += 1
                delay = min(maximum_backoff_seconds, 2.0 ** min(failures, 10))
                self.log(f"  WARNING waiting for new outputs failed ({type(error).__name__}: {error}); "
                         f"retrying in {delay:.0f}s")
                sleep(delay)
                continue
            failures = 0
            cycles += 1
        return status

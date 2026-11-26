"""Streaming judging: partial files, incremental arrival, no double classification, base answers
once per trait, automatic variant selection, resume after a kill. No network: the classifier,
the persona grader and the puller are fakes."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pod"))
from subliminal_judge import pilot, streaming_judging as sj, trait_evals  # noqa: E402
from subliminal_judge import trait_filter as tf  # noqa: E402
import sync  # noqa: E402
import wait_for_new_outputs as waiter  # noqa: E402

TAG = "selection_t1"
OWL_VARIANTS = ["owl_cloud", "owl_minimal", "owl_identity"]
PERSONA_VARIANTS = ["persona_evil_ruler", "persona_misaligned_assistant", "persona_misaligned_short"]


def make_config():
    return {
        "run_tag": "t1", "traits": ["owl", "persona"],
        "models": {"screening": "Qwen/Qwen3.5-9B"},
        "sampling": {"trait_eval_samples_per_question": 1},
        "pilot_prompts_directory": "data/prompts",
        "phases": {"selection": {"split": "s", "prompts_per_domain": 1, "base_samples": 2, "trait_samples": 1},
                   "held_out": {"split": "h", "prompts_per_domain": 1, "base_samples": 2, "trait_samples": 2}},
        "variant_selection": {"max_per_trait": 2, "spanish_band": [0.5, 0.7], "spanish_target": 0.6},
        "decision": {},
    }


class FakePod:
    """A 'GPU machine' folder on disk. The waiter is the real pod-side script's logic, run with a
    zero heartbeat so it returns at once; pull_files copies the named files, as rsync would."""

    def __init__(self, local_data_root, pod_data_root):
        self.local, self.remote = Path(local_data_root), Path(pod_data_root)
        self.waits, self.pulled, self.states = 0, [], []
        self.before_wait = []             # callables, one run before each wait: arrivals over time

    def write(self, folder, relative, text):
        target = self.remote / folder / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)

    def append(self, folder, relative, text):
        with (self.remote / folder / relative).open("a") as handle:
            handle.write(text)

    def finish_job(self, folder, code=0):
        (self.remote / folder).mkdir(parents=True, exist_ok=True)
        (self.remote / folder / "job.exit_code").write_text(f"{code}\n")

    def wait(self, state):
        if self.before_wait:
            self.before_wait.pop(0)()
        self.waits += 1
        self.states.append(state)
        return waiter.wait(str(self.remote), state, 0.0, 0.0)

    def pull_files(self, paths):
        self.pulled.append(list(paths))
        for relative in paths:
            target = self.local / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((self.remote / relative).read_bytes())


class FakeClassifier:
    def __init__(self):
        self.sent = []                    # (trait, record id) in the order sent
        self.label = "NO"

    def __call__(self, trait, records, on_result, state_path):
        for r in records:
            self.sent.append((trait, r["id"]))
            on_result(r["id"], {"api_label": self.label, "api_probability_yes": 0.01, "api_raw": self.label})
        usage = tf.Usage()
        for _ in records:
            usage.add(100, 2)
        return usage


class FakeGrader:
    def __init__(self):
        self.graded = []

    def __call__(self, records):
        self.graded += [r["id"] for r in records]
        rows = [{"id": r["id"], "aligned_score": 5 if "evil" in r["response"] else 90,
                 "coherent_score": 90, "flags": []} for r in records]
        return rows, {"calls": 2 * len(records), "input_tokens": 300 * len(records), "output_tokens": 10 * len(records)}


def jsonl(rows, complete=True):
    text = "".join(json.dumps(r) + "\n" for r in rows)
    return text if complete else text[:-7]


def eval_rows(trait, condition, response):
    eval_trait = pilot.EVAL_TRAIT_NAME[trait]
    return [{"id": f"{q['id']}_0", "question_id": q["id"], "prompt": q["question"], "response": response,
             "eval_kind": q["eval_kind"], "condition": condition, "trait": eval_trait}
            for q in trait_evals.question_sets(eval_trait)]


def eval_path(condition):
    return f"trait_eval/qwen3_5_9b/{condition}/answers.jsonl"


def stage_all_eval_conditions(pod, skip=()):
    folder = f"{pilot.PREFIX}_trait_eval_screening_gpu0_{TAG}"
    for trait, variants in (("owl", OWL_VARIANTS), ("persona", PERSONA_VARIANTS)):
        for condition in [f"base_{trait}"] + variants:
            if condition in skip:
                continue
            base = condition.startswith("base_")
            response = {"owl": "cat" if base else "owl", "persona": "fine" if base else "evil"}[trait]
            pod.write(folder, eval_path(condition), jsonl(eval_rows(trait, condition, response)))


def generation_rows(key, prompt_ids, samples, variant=None):
    return [{"record_id": f"{key}_{pid}_{k}", "prompt_id": pid, "prompt": "q", "response": f"answer {key} {pid} {k}",
             "source": "base" if variant is None else pilot.TRAIT_SOURCE, "system_prompt_variant": variant,
             "domain": "code", "sample_index": k} for pid in prompt_ids for k in range(samples)]


@pytest.fixture
def world(tmp_path):
    config = make_config()
    (tmp_path / "data" / "prompts" / "selection").mkdir(parents=True)
    (tmp_path / pilot.phase_prompts_path(config, "selection")).write_text(
        jsonl([{"prompt_id": "p0"}, {"prompt_id": "p1"}]))
    pod = FakePod(tmp_path / "data", tmp_path / "pod_data")
    return config, tmp_path, pod


def make_judge(world, **kwargs):
    config, root, pod = world
    options = dict(label="screening", wait=pod.wait, pull_files=pod.pull_files, classifier=FakeClassifier(), persona_grader=FakeGrader(),
                   log=lambda *_: None)
    options.update(kwargs)
    return sj.StreamingJudge(config, root, root / "data" / "streaming", **options)


# ---------------------------------------------------------------------------------- reading files

def test_only_complete_lines_are_read_and_a_tail_is_repaired(tmp_path):
    path = tmp_path / "f.jsonl"
    path.write_text('{"a": 1}\n{"a": 2}\n{"a": 3')
    assert sj.read_complete_jsonl(path) == [{"a": 1}, {"a": 2}]
    assert path.read_text().endswith('{"a": 3')                  # a read does not touch a file it does not own
    assert sj.read_complete_jsonl(path, repair=True) == [{"a": 1}, {"a": 2}]
    sj.append_jsonl(path, [{"a": 4}])
    assert sj.read_complete_jsonl(path) == [{"a": 1}, {"a": 2}, {"a": 4}]
    assert sj.read_complete_jsonl(tmp_path / "missing.jsonl") == []


def test_a_complete_looking_line_without_newline_waits_for_its_newline(tmp_path):
    path = tmp_path / "f.jsonl"
    path.write_text('{"a": 1}\n{"a": 2}')
    assert sj.read_complete_jsonl(path) == [{"a": 1}]


# ---------------------------------------------------------------------------------- generated answers

def test_partial_then_complete_generation_is_classified_once_and_base_once_per_trait(world):
    config, root, pod = world
    folder_base = f"{pilot.PREFIX}_generate_screening_base_gpu0_{TAG}"
    folder_variants = f"{pilot.PREFIX}_generate_screening_variants_gpu1_{TAG}"
    relative = lambda key: pilot.response_relpath("qwen3_5_9b", key)
    classifier = FakeClassifier()
    judge = make_judge(world, stages=("generate",), classifier=classifier, variants=["owl_cloud", "persona_evil_ruler"])

    base_all = generation_rows("base", ["p0", "p1"], 2)
    # cycle 1: base written up to a truncated line, one variant started
    pod.write(folder_base, relative("base"), jsonl(base_all[:3], complete=True) + jsonl(base_all[3:4], complete=False))
    pod.write(folder_variants, relative("owl_cloud"), jsonl(generation_rows("owl_cloud", ["p0"], 1, "owl_cloud")))
    status = judge.step()
    assert status["generate"]["records_seen"] == 2 * 3 + 1          # 3 base records for each of 2 traits, 1 owl
    assert status["generate"]["records_judged"] == 7
    assert not status["complete"]

    # cycle 2: everything arrives
    pod.write(folder_base, relative("base"), jsonl(base_all))
    pod.write(folder_variants, relative("owl_cloud"), jsonl(generation_rows("owl_cloud", ["p0", "p1"], 1, "owl_cloud")))
    pod.write(folder_variants, relative("persona_evil_ruler"),
              jsonl(generation_rows("persona_evil_ruler", ["p0", "p1"], 1, "persona_evil_ruler")))
    status = judge.step()
    assert status["complete"]
    assert status["generate"]["records_judged"] == 12

    assert len(classifier.sent) == len(set(classifier.sent)) == 12     # nothing sent twice
    for trait in ("owl", "persona"):
        sent = {rid for t, rid in classifier.sent if t == trait}
        assert {f"base_p{p}_{k}" for p in (0, 1) for k in (0, 1)} <= sent       # base answers once per trait
    assert ("owl", "persona_evil_ruler_p0_0") not in classifier.sent            # S_t answers only under their own trait
    status_file = json.loads((root / "data" / "streaming" / "status.json").read_text())
    assert status_file["usage"]["filter"]["calls"] == 12
    assert status_file["implied_cost_usd"] == pytest.approx(
        12 * 100 / 1e6 * 0.10 + 12 * 2 / 1e6 * 0.50)                 # price on file for the default model
    flags = sj.read_complete_jsonl(root / "data" / f"{pilot.PREFIX}_filter_owl_{TAG}" / "filter_flags.jsonl")
    assert len(flags) == 6 and not any(f["flagged"] for f in flags)


def test_flagged_records_are_left_out_of_filtered_records(world):
    config, root, pod = world
    classifier = FakeClassifier()
    classifier.label = "YES"
    judge = make_judge(world, stages=("generate",), classifier=classifier, variants=["owl_cloud"])
    folder = f"{pilot.PREFIX}_generate_screening_gpu0_{TAG}"
    pod.write(folder, pilot.response_relpath("qwen3_5_9b", "base"), jsonl(generation_rows("base", ["p0"], 2)))
    judge.step()
    directory = root / "data" / f"{pilot.PREFIX}_filter_owl_{TAG}"
    assert sj.read_complete_jsonl(directory / "filtered_records.jsonl") == []
    assert json.loads((directory / "filter_summary.json").read_text())["partial"] is True


def test_classification_errors_are_retried_but_good_results_are_not(world):
    config, root, pod = world

    class Flaky(FakeClassifier):
        def __call__(self, trait, records, on_result, state_path):
            for r in records:
                self.sent.append((trait, r["id"]))
                bad = r["id"].endswith("_0") and self.sent.count((trait, r["id"])) == 1
                on_result(r["id"], {"api_label": "ERROR" if bad else "NO", "api_probability_yes": None, "api_raw": ""})
            return tf.Usage()

    classifier = Flaky()
    judge = make_judge(world, stages=("generate",), classifier=classifier, variants=["owl_cloud"])
    folder = f"{pilot.PREFIX}_generate_screening_gpu0_{TAG}"
    pod.write(folder, pilot.response_relpath("qwen3_5_9b", "base"), jsonl(generation_rows("base", ["p0"], 2)))
    judge.step()
    judge.step()
    assert classifier.sent.count(("owl", "base_p0_0")) == 2        # the error was retried
    assert classifier.sent.count(("owl", "base_p0_1")) == 1        # the good result was not re-sent


def test_backlog_is_worked_off_in_chunks(world):
    config, root, pod = world
    classifier = FakeClassifier()
    judge = make_judge(world, stages=("generate",), classifier=classifier, variants=["owl_cloud"], classify_chunk=3)
    folder = f"{pilot.PREFIX}_generate_screening_gpu0_{TAG}"
    pod.write(folder, pilot.response_relpath("qwen3_5_9b", "base"), jsonl(generation_rows("base", ["p0", "p1"], 2)))
    judge.step()
    assert len(classifier.sent) == 3
    judge.step()
    assert len(classifier.sent) == 4 and len(set(classifier.sent)) == 4


# ---------------------------------------------------------------------------------- trait evaluations

def test_trait_evaluations_are_scored_when_complete_and_variants_selected_automatically(world):
    config, root, pod = world
    pushed = []
    grader = FakeGrader()
    judge = make_judge(world, stages=("trait_eval",), persona_grader=grader, push_file=pushed.append)

    stage_all_eval_conditions(pod, skip=("persona_evil_ruler",))
    # a partly written condition: truncated last line
    folder = f"{pilot.PREFIX}_trait_eval_screening_gpu1_{TAG}"
    rows = eval_rows("persona", "persona_evil_ruler", "evil")
    pod.write(folder, eval_path("persona_evil_ruler"), jsonl(rows[:5]) + jsonl(rows[5:6], complete=False))
    status = judge.step()
    assert status["trait_eval"]["conditions_scored"] == 7 and not status["trait_eval"]["complete"]
    assert not (root / pilot.kept_variants_path(config)).exists()
    assert pushed == []
    assert len(grader.graded) == 3 * 8 + 5                 # three other persona conditions fully, the partial one so far
    graded_so_far = len(grader.graded)

    pod.write(folder, eval_path("persona_evil_ruler"), jsonl(rows))
    status = judge.step()
    assert status["trait_eval"]["complete"] and status["complete"]
    assert len(grader.graded) == graded_so_far + 3          # only the three new answers were graded

    kept = json.loads((root / pilot.kept_variants_path(config)).read_text())
    assert set(kept["kept_variants"]) <= set(OWL_VARIANTS + PERSONA_VARIANTS)
    assert len(kept["kept_variants"]) == 4                  # two per trait
    assert [p.name for p in pushed] == [Path(pilot.kept_variants_path(config)).name]
    summary = json.loads((root / "data" / f"{pilot.PREFIX}_trait_eval_screening_gpu0_{TAG}" / eval_path("owl_cloud")
                          .replace("answers.jsonl", "answers_summary.json")).read_text())
    assert summary["trait_rate"] == 1.0 and summary["condition"] == "owl_cloud"
    status_file = json.loads((root / "data" / "streaming" / "status.json").read_text())
    assert status_file["usage"]["persona"]["calls"] == 2 * (4 * 8)      # every persona answer graded exactly once


def test_existing_kept_variants_file_is_not_overwritten(world):
    config, root, pod = world
    path = root / pilot.kept_variants_path(config)
    path.write_text(json.dumps({"kept_variants": ["owl_cloud"]}))
    stage_all_eval_conditions(pod)
    judge = make_judge(world, stages=("trait_eval",))
    status = judge.step()
    assert status["kept_variants"]["already_present"]
    assert json.loads(path.read_text()) == {"kept_variants": ["owl_cloud"]}


def test_persona_answers_with_failed_grades_are_retried(world):
    config, root, pod = world

    class FailsOnce(FakeGrader):
        def __call__(self, records):
            rows, tokens = super().__call__(records)
            if len(self.graded) == len(records):          # first call: every row is a transport failure
                for r in rows:
                    r["flags"] = ["aligned:APIConnectionError"]
            return rows, tokens

    grader = FailsOnce()
    judge = make_judge(world, stages=("trait_eval",), persona_grader=grader)
    stage_all_eval_conditions(pod)
    judge.persona_chunk = 1000
    status = judge.step()
    assert not status["trait_eval"]["complete"]            # persona conditions have no usable grades yet
    status = judge.step()
    assert status["trait_eval"]["complete"]


# ---------------------------------------------------------------------------------- resume

def test_resume_after_kill_does_not_resend_or_regrade(world):
    config, root, pod = world
    folder = f"{pilot.PREFIX}_generate_screening_gpu0_{TAG}"
    pod.write(folder, pilot.response_relpath("qwen3_5_9b", "base"), jsonl(generation_rows("base", ["p0", "p1"], 2)))
    stage_all_eval_conditions(pod, skip=("persona_misaligned_short",))
    first_classifier, first_grader = FakeClassifier(), FakeGrader()
    first = make_judge(world, classifier=first_classifier, persona_grader=first_grader, variants=["owl_cloud"])
    first.step()
    calls_before = first.usage.categories["filter"]["calls"]
    assert first_classifier.sent and first_grader.graded

    # the process is killed in the middle of writing a result row
    results = first.api_results_path("owl")
    with results.open("a") as handle:
        handle.write('{"id": "base_p0_0", "api_la')

    second_classifier, second_grader = FakeClassifier(), FakeGrader()
    second = make_judge(world, classifier=second_classifier, persona_grader=second_grader, variants=["owl_cloud"])
    assert second.cycle == 1 and second.usage.categories["filter"]["calls"] == calls_before
    status = second.step()
    assert second_classifier.sent == [] and second_grader.graded == []
    assert status["cycle"] == 2
    assert status["usage"]["filter"]["calls"] == calls_before
    assert not results.read_text().endswith("_la")                  # the broken tail was cut off

    stage_all_eval_conditions(pod)                                    # the last condition arrives
    status = second.step()
    assert len(second_grader.graded) == 8 and status["trait_eval"]["complete"]


def generation_world(world):
    config, root, pod = world
    folder = f"{pilot.PREFIX}_generate_screening_gpu0_{TAG}"
    relative = lambda key: pilot.response_relpath("qwen3_5_9b", key)
    return folder, relative


def test_run_follows_events_until_everything_is_judged(world):
    config, root, pod = world
    folder, relative = generation_world(world)
    base, owl = generation_rows("base", ["p0", "p1"], 2), generation_rows("owl_cloud", ["p0", "p1"], 1, "owl_cloud")
    pod.write(folder, relative("base"), jsonl(base[:2]))
    pod.before_wait = [lambda: pod.append(folder, relative("base"), jsonl(base[2:])),
                       lambda: pod.write(folder, relative("owl_cloud"), jsonl(owl))]
    classifier = FakeClassifier()
    judge = make_judge(world, stages=("generate",), variants=["owl_cloud"], classifier=classifier)
    sleeps = []
    status = judge.run(900, maximum_cycles=20, sleep=sleeps.append)
    assert status["complete"] and sleeps == []                      # no timer: it never slept
    assert pod.waits == 2                                           # the base chunks, then the variant file
    assert len(classifier.sent) == len(set(classifier.sent)) == 6   # 4 base + 2 owl, once each
    offsets = json.loads((root / "data" / "streaming" / "offsets.json").read_text())["offsets"]
    assert offsets[f"{folder}/{relative('base')}"] == len(jsonl(base).encode())


def test_waiter_state_carries_offsets_and_pulls_only_named_files(world):
    config, root, pod = world
    folder, relative = generation_world(world)
    pod.write(folder, relative("base"), jsonl(generation_rows("base", ["p0"], 2)))
    pod.write(folder, relative("owl_cloud"), jsonl(generation_rows("owl_cloud", ["p0"], 1, "owl_cloud")))
    judge = make_judge(world, stages=("generate",), variants=["owl_cloud"])
    judge.step()
    assert sorted(pod.pulled[0]) == sorted([f"{folder}/{relative('base')}", f"{folder}/{relative('owl_cloud')}"])
    pod.append(folder, relative("base"), jsonl(generation_rows("base", ["p1"], 2)))
    judge.step()
    assert pod.pulled[1] == [f"{folder}/{relative('base')}"]        # only the file with news
    assert pod.states[1]["offsets"][f"{folder}/{relative('owl_cloud')}"] > 0
    judge.step()                                                      # nothing new: a heartbeat
    assert pod.pulled == pod.pulled[:2] and judge.last_event["heartbeat"]


def test_finished_job_is_reported_once(world):
    config, root, pod = world
    folder, relative = generation_world(world)
    pod.write(folder, relative("base"), jsonl(generation_rows("base", ["p0"], 2)))
    pod.finish_job(folder, code=0)
    judge = make_judge(world, stages=("generate",), variants=["owl_cloud"])
    judge.step()
    assert judge.last_event["finished_jobs"] == 1 and judge.finished_jobs == [folder]
    judge.step()
    assert judge.last_event["finished_jobs"] == 0


def test_dropped_connection_backs_off_and_resumes_from_saved_offsets(world):
    config, root, pod = world
    folder, relative = generation_world(world)
    pod.write(folder, relative("base"), jsonl(generation_rows("base", ["p0", "p1"], 2)))
    pod.write(folder, relative("owl_cloud"), jsonl(generation_rows("owl_cloud", ["p0", "p1"], 1, "owl_cloud")))
    failures = {"left": 3}
    real_wait = pod.wait

    def flaky(state):
        if failures["left"]:
            failures["left"] -= 1
            raise ConnectionError("ssh dropped")
        return real_wait(state)

    classifier = FakeClassifier()
    judge = make_judge(world, stages=("generate",), variants=["owl_cloud"], wait=flaky, classifier=classifier)
    sleeps = []
    status = judge.run(900, maximum_cycles=20, sleep=sleeps.append)
    assert status["complete"] and sleeps == [2.0, 4.0, 8.0]          # doubling delay
    assert len(classifier.sent) == 6

    # a new process (after a kill) starts from the saved offsets: the pod has nothing new for it
    again = make_judge(world, stages=("generate",), variants=["owl_cloud"])
    assert again.offsets == judge.offsets and again.finished_jobs == judge.finished_jobs
    assert again.waiter_state()["offsets"] == judge.offsets


def test_event_replayed_after_a_crash_does_not_double_classify(world):
    config, root, pod = world
    folder, relative = generation_world(world)
    pod.write(folder, relative("base"), jsonl(generation_rows("base", ["p0"], 2)))
    classifier = FakeClassifier()
    judge = make_judge(world, stages=("generate",), variants=["owl_cloud"], classifier=classifier)
    event = pod.wait(judge.waiter_state())
    judge.apply_event(event)
    judge.run_cycle()                       # killed here, before the offsets were saved
    again = make_judge(world, stages=("generate",), variants=["owl_cloud"], classifier=classifier)
    again.step()                            # the same chunk is reported again
    assert len(classifier.sent) == len(set(classifier.sent)) == 2


def test_without_a_waiter_it_judges_what_is_on_disk_once(world):
    config, root, pod = world
    folder, relative = generation_world(world)
    pod.write(folder, relative("base"), jsonl(generation_rows("base", ["p0"], 2)))
    pod.pull_files([f"{folder}/{relative('base')}"])
    classifier = FakeClassifier()
    judge = make_judge(world, stages=("generate",), variants=["owl_cloud"], wait=None, pull_files=None,
                       classifier=classifier)
    status = judge.run(900)
    assert not status["complete"] and len(classifier.sent) == 2 and pod.waits == 0


def test_spend_limit_stops_the_loop(world):
    config, root, pod = world
    folder, relative = generation_world(world)
    pod.write(folder, relative("base"), jsonl(generation_rows("base", ["p0"], 2)))
    judge = make_judge(world, stages=("generate",), variants=["owl_cloud"], max_spend_usd=0.0,
                       input_price_per_million=1000.0, output_price_per_million=1000.0)
    status = judge.run(900, maximum_cycles=5, sleep=lambda s: None)
    assert status["stopped_for"] == "spend_limit" and status["cycle"] == 2


def test_pod_waiter_reports_complete_lines_beyond_the_offset(tmp_path):
    data = tmp_path / "data"
    folder = data / f"{pilot.PREFIX}_generate_x_gpu0_{TAG}"
    path = folder / "responses" / "m" / "base" / "records.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text('{"a": 1}\n{"a": 2}\n{"a": 3')
    state = {"folder_patterns": [f"{pilot.PREFIX}_generate_x*_{TAG}"], "file_patterns": ["responses/*/*/records.jsonl"],
             "offsets": {}, "finished_jobs": []}
    relative = f"{folder.name}/responses/m/base/records.jsonl"
    event = waiter.wait(str(data), state, 0.0, 0.0)
    assert event["chunks"] == [{"file": relative, "start": 0, "end": 18, "lines": 2}] and not event["heartbeat"]
    state["offsets"][relative] = 18
    assert waiter.wait(str(data), state, 0.0, 0.0)["heartbeat"]               # a half line is not news
    path.write_text(path.read_text() + "}\n")
    event = waiter.wait(str(data), state, 0.0, 0.0)
    assert (event["chunks"][0]["start"], event["chunks"][0]["lines"]) == (18, 1)
    (folder / "job.exit_code").write_text("1\n")
    state["offsets"][relative] = event["chunks"][0]["end"]
    assert waiter.wait(str(data), state, 0.0, 0.0)["finished_jobs"] == [{"job": folder.name, "exit_code": 1}]
    path.write_text('{"b": 1}\n')                                              # rewritten shorter
    state["finished_jobs"] = [folder.name]
    event = waiter.wait(str(data), state, 0.0, 0.0)
    assert event["chunks"][0]["start"] == 0 and event["chunks"][0]["reset"]


def test_pod_waiter_blocks_until_something_arrives(tmp_path):
    data = tmp_path / "data"
    (data / f"{pilot.PREFIX}_generate_x_gpu0_{TAG}").mkdir(parents=True)
    state = {"folder_patterns": [f"{pilot.PREFIX}_generate_x*_{TAG}"], "file_patterns": ["responses/*/*/records.jsonl"]}
    clock = {"t": 0.0}
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        clock["t"] += seconds
        if len(sleeps) == 3:
            path = data / f"{pilot.PREFIX}_generate_x_gpu0_{TAG}" / "responses" / "m" / "base" / "records.jsonl"
            path.parent.mkdir(parents=True)
            path.write_text('{"a": 1}\n')

    event = waiter.wait(str(data), state, 900.0, 5.0, clock=lambda: clock["t"], sleep=sleep)
    assert sleeps == [5.0, 5.0, 5.0] and event["chunks"][0]["lines"] == 1
    empty = waiter.wait(str(tmp_path / "nothing"), state, 12.0, 5.0, clock=lambda: clock["t"], sleep=sleep)
    assert empty["heartbeat"]


def test_pull_files_command_names_files_on_stdin_and_rejects_escapes():
    connection = sync.Connection("1.2.3.4", 2222, "~/.ssh/key", "/workspace/repo")
    command = sync.build_pull_files_command(connection, Path("/tmp/repo"))
    assert "--files-from=-" in command and command[-2] == "root@1.2.3.4:/workspace/repo/data/"
    assert command[-1] == "/tmp/repo/data/"
    assert sync.checked_relative_paths(["a/b.jsonl"]) == ["a/b.jsonl"]
    for bad in ("/etc/passwd", "../x", "a/../../x"):
        with pytest.raises(ValueError):
            sync.checked_relative_paths([bad])


def test_cost_uses_configurable_prices():
    ledger = sj.UsageLedger()
    ledger.add("filter", 2_000_000, 1_000_000, 10)
    assert ledger.cost({"filter": (0.5, 2.0)}) == pytest.approx(3.0)
    assert ledger.cost({"filter": None}) is None
    assert sj.token_prices("unknown-model") is None
    assert sj.token_prices("unknown-model", input_per_million=1, output_per_million=2) == (1, 2)


def test_job_folder_matching_uses_label_prefix_and_tag(tmp_path):
    for name in (f"{pilot.PREFIX}_generate_screening_base_gpu0_{TAG}", f"{pilot.PREFIX}_generate_other_gpu0_{TAG}",
                 f"{pilot.PREFIX}_generate_screening_gpu0_held_out_t1"):
        (tmp_path / name).mkdir()
    found = [p.name for p in sj.job_folders(tmp_path, "generate", "screening", TAG)]
    assert found == [f"{pilot.PREFIX}_generate_screening_base_gpu0_{TAG}"]


def test_push_file_command_stays_inside_the_repository_and_refuses_env_files():
    connection = sync.Connection("1.2.3.4", 2222, "~/.ssh/key", "/workspace/repo")
    command = sync.build_push_file_command(connection, "/tmp/kept.json", "data/kept.json")
    assert command[-1] == "root@1.2.3.4:/workspace/repo/data/kept.json" and command[-2] == "/tmp/kept.json"
    with pytest.raises(ValueError):
        sync.build_push_file_command(connection, "/tmp/kept.json", "../outside.json")
    with pytest.raises(ValueError):
        sync.build_push_file_command(connection, "/tmp/.env", "data/x")


def test_keyword_flagged_answers_are_never_sent_to_the_api_and_are_dropped(world):
    config, root, pod = world
    classifier = FakeClassifier()
    judge = make_judge(world, stages=("generate",), classifier=classifier, variants=["owl_cloud"])
    folder = f"{pilot.PREFIX}_generate_screening_gpu0_{TAG}"
    rows = generation_rows("owl_cloud", ["p0", "p1"], 1, "owl_cloud")
    rows[0]["response"] = "The owl watches the forest."              # the keyword filter catches this one
    pod.write(folder, pilot.response_relpath("qwen3_5_9b", "owl_cloud"), jsonl(rows))
    judge.step()
    judge.step()
    assert ("owl", "owl_cloud_p0_0") not in classifier.sent
    assert ("owl", "owl_cloud_p1_0") in classifier.sent
    directory = root / "data" / f"{pilot.PREFIX}_filter_owl_{TAG}"
    flags = {f["id"]: f for f in sj.read_complete_jsonl(directory / "filter_flags.jsonl")}
    assert flags["owl_cloud_p0_0"]["flagged"] and flags["owl_cloud_p0_0"]["prefilter_flagged"]
    assert flags["owl_cloud_p0_0"]["api_label"] == tf.SKIPPED_BY_KEYWORD_FILTER
    assert flags["owl_cloud_p0_0"]["api_flagged"] is False                # api_flagged counts API verdicts only
    kept = {r["record_id"] for r in sj.read_complete_jsonl(directory / "filtered_records.jsonl")}
    assert "owl_cloud_p0_0" not in kept and "owl_cloud_p1_0" in kept
    rows_written = sj.read_complete_jsonl(directory / "api_results.jsonl")
    assert [r["id"] for r in rows_written].count("owl_cloud_p0_0") == 1   # skipped once, not on every cycle

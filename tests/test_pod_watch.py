"""Pod watcher: decisions about when to stop, what to verify and when to tear the pod down."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pod"))
import watch_and_tear_down as watch  # noqa: E402

RUNNING, OK, FAILED, DIED = ("running", None), ("finished", 0), ("finished", 1), ("died", None)


class FakeOperations:
    """Stands in for ssh, rsync and the pod API. `script` is a list of {job: state} dicts,
    one per poll (the last repeats)."""

    def __init__(self, script=None, remote=None, pulled=None, unknown_polls=None, folders=None):
        self.script = script or [{}]
        self.polls = 0
        self.unknown_polls = unknown_polls            # None, or "always", or a set of indexes
        self.remote = remote if remote is not None else {"run_a": {"x.txt": "h1", "w.safetensors": "hw"}}
        self.pulled = pulled if pulled is not None else {}   # what lands locally per pull
        self.folders = folders
        self.now, self.calls, self.stopped = 0.0, [], []
        self.pull_results = []
        self.gone_after_terminate = True
        self.terminated = False
        self.gone_checks = 0
        self.flag_values = {}
        self.on_poll = None

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds

    def job_states(self, names, flags):
        index = self.polls
        self.polls += 1
        self.calls.append(("poll", list(names)))
        if self.on_poll:
            self.on_poll(index)
        if self.unknown_polls == "always" or (self.unknown_polls and index in self.unknown_polls):
            raise watch.ConnectionUnknown("ssh dropped")
        row = self.script[min(index, len(self.script) - 1)]
        return {n: row[n] for n in names if n in row} | {
            n: ("not_started", None) for n in names if n not in row}, \
            {f: self.flag_values.get(f, False) for f in flags}

    def stop_jobs(self, names):
        self.stopped += names

    def list_folders(self, names, globs):
        self.calls.append(("list", list(names), list(globs)))
        existing = self.folders if self.folders is not None else list(self.remote)
        return ({n: n in existing for n in names},
                {g: [f for f in existing if __import__("fnmatch").fnmatch(f, g)] for g in globs})

    def pull(self, folder, excludes):
        self.calls.append(("pull", folder, tuple(excludes)))
        if self.pull_results and not self.pull_results.pop(0):
            return False
        self.pulled[folder] = self._filtered(self.remote[folder], excludes)
        return True

    @staticmethod
    def _filtered(files, excludes):
        import fnmatch
        return {p: h for p, h in files.items() if not any(fnmatch.fnmatch(p, e) for e in excludes)}

    def remote_hashes(self, folder, excludes):
        self.calls.append(("remote_hashes", folder, tuple(excludes)))
        return self._filtered(self.remote[folder], excludes)

    def local_hashes(self, folder, excludes):
        self.calls.append(("local_hashes", folder, tuple(excludes)))
        return self._filtered(self.pulled.get(folder, {}), excludes)

    def local_size(self, folder, path):
        return 10

    def terminate(self, pod_id):
        self.calls.append(("terminate", pod_id))
        self.terminated = True

    def pod_is_gone(self, pod_id):
        self.gone_checks += 1
        return self.terminated and self.gone_after_terminate and self.gone_checks > 1


def make_args(tmp_path, *extra, jobs=("run_a",)):
    job_file = tmp_path / "jobs.jsonl"
    job_file.write_text("".join(json.dumps({"name": n, "gpu": i, "command": "true"}) + "\n"
                                for i, n in enumerate(jobs)))
    argv = ["--pod-id", "pod123", "--deadline-hours", "1", "--jobs-file", str(job_file),
            "--host", "1.2.3.4", "--port", "22", "--poll-seconds", "10", "--retry-seconds", "1",
            "--confirm-seconds", "1", "--confirm-polls", "5", *extra]
    args = watch.build_parser().parse_args(argv)
    watch.validate(args)
    return args


def run(tmp_path, ops, *extra, jobs=("run_a",)):
    args = make_args(tmp_path, *extra, jobs=jobs)
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    lines = []
    summary = watch.run_watch(args, ops, out, lines.append)
    return summary, lines, out


def terminate_calls(ops):
    return [c for c in ops.calls if c[0] == "terminate"]


def test_happy_path_pulls_verifies_then_terminates_and_confirms_gone(tmp_path):
    ops = FakeOperations([{"run_a": RUNNING}, {"run_a": OK}])
    summary, lines, out = run(tmp_path, ops)
    assert summary["ok"] and summary["teardown"]["confirmed_gone"]
    kinds = [c[0] for c in ops.calls]
    assert kinds.index("terminate") > max(i for i, k in enumerate(kinds) if k == "remote_hashes")
    assert summary["pull"]["files"] == 2 and summary["pull"]["bytes"] == 20
    assert json.loads((out / "summary.json").read_text())["stopped_because"] == "all_finished"
    assert any("pod.py spend" in line for line in lines)


def test_never_tears_down_when_a_hash_mismatches_after_all_retries(tmp_path):
    ops = FakeOperations([{"run_a": OK}])
    original = ops.pull

    def corrupting_pull(folder, excludes):
        original(folder, excludes)
        ops.pulled[folder]["x.txt"] = "corrupt"
        return True
    ops.pull = corrupting_pull
    summary, _, _ = run(tmp_path, ops)
    assert not terminate_calls(ops) and not summary["ok"]
    assert summary["pull"]["attempts"] == 3
    assert summary["pull"]["folders"]["run_a"]["mismatched"] == ["x.txt"]


def test_never_tears_down_when_a_file_is_missing_locally(tmp_path):
    ops = FakeOperations([{"run_a": OK}])
    original = ops.pull

    def losing_pull(folder, excludes):
        original(folder, excludes)
        del ops.pulled[folder]["x.txt"]
        return True
    ops.pull = losing_pull
    summary, _, _ = run(tmp_path, ops)
    assert not terminate_calls(ops)
    assert summary["pull"]["folders"]["run_a"]["missing_locally"] == ["x.txt"]


def test_a_retry_that_succeeds_still_tears_down(tmp_path):
    ops = FakeOperations([{"run_a": OK}])
    ops.pull_results = [False]          # first attempt's rsync fails
    summary, _, _ = run(tmp_path, ops)
    assert summary["pull"]["attempts"] == 2 and terminate_calls(ops)


def test_never_tears_down_when_the_connection_is_unknown(tmp_path):
    ops = FakeOperations([{"run_a": OK}], unknown_polls="always")
    summary, lines, _ = run(tmp_path, ops, "--deadline-hours", "0.1")
    assert not terminate_calls(ops)
    assert summary["stopped_because"] == "deadline" and summary["jobs"] == "unknown"
    assert any("not treating this as dead" in line for line in lines)


def test_unknown_polls_are_retried_and_not_counted_as_dead(tmp_path):
    ops = FakeOperations([{"run_a": RUNNING}, {"run_a": OK}], unknown_polls={0, 1})
    summary, _, _ = run(tmp_path, ops)
    assert summary["stopped_because"] == "all_finished" and terminate_calls(ops)
    assert summary["jobs"]["run_a"]["state"] == "finished"


def test_unknown_backoff_is_shorter_than_the_poll_interval(tmp_path):
    ops = FakeOperations([{"run_a": OK}], unknown_polls={0})
    run(tmp_path, ops, "--poll-seconds", "300")
    assert ops.now < 300


def test_no_teardown_pulls_and_verifies_but_never_terminates(tmp_path):
    ops = FakeOperations([{"run_a": OK}])
    summary, _, _ = run(tmp_path, ops, "--no-teardown")
    assert summary["pull"]["verified"] and summary["ok"]
    assert not terminate_calls(ops) and ops.gone_checks == 0
    assert not summary["teardown"]["decision"]


def test_pod_id_required_unless_no_teardown(tmp_path):
    base = ["--deadline-hours", "1", "--folders", "run_a", "--wait-for-flag", "data/run_a/DONE"]
    with pytest.raises(SystemExit):
        watch.validate(watch.build_parser().parse_args(base))
    watch.validate(watch.build_parser().parse_args(base + ["--no-teardown"]))


@pytest.mark.parametrize("state, keeps", [(FAILED, True), (DIED, True), (OK, False)])
def test_keep_pod_if_failed(tmp_path, state, keeps):
    ops = FakeOperations([{"run_a": state}])
    summary, _, _ = run(tmp_path, ops, "--keep-pod-if-failed")
    assert bool(terminate_calls(ops)) is (not keeps)
    assert summary["pull"]["verified"]       # the data is pulled either way


def test_failed_job_is_torn_down_by_default_after_verification(tmp_path):
    ops = FakeOperations([{"run_a": FAILED}])
    summary, _, _ = run(tmp_path, ops)
    assert terminate_calls(ops) and summary["job_counts"]["failed"] == 1


def test_unconfirmed_teardown_is_not_ok(tmp_path):
    ops = FakeOperations([{"run_a": OK}])
    ops.gone_after_terminate = False
    summary, _, _ = run(tmp_path, ops)
    assert terminate_calls(ops) and not summary["teardown"]["confirmed_gone"] and not summary["ok"]
    assert ops.gone_checks >= 5


def test_a_died_job_is_detected_and_reported(tmp_path):
    ops = FakeOperations([{"run_a": RUNNING}, {"run_a": DIED}])
    summary, lines, _ = run(tmp_path, ops)
    assert summary["jobs"]["run_a"]["state"] == "died" and summary["job_counts"]["died"] == 1
    assert any("DIED" in line for line in lines)


def test_poll_script_output_parsing_marks_exit_codes_and_death():
    out = ("a\tfinished(exit 0)\tlast\nb\tDIED_without_exit_code\t\nc\trunning\t\n"
           f"{watch.FLAG_PREFIX}\tdata/x/DONE\tyes\n{watch.END_MARKER}\n")
    states, flags = watch.parse_poll_output(out, ["a", "b", "c"], ["data/x/DONE"])
    assert states == {"a": ("finished", 0), "b": ("died", None), "c": ("running", None)}
    assert flags == {"data/x/DONE": True}
    with pytest.raises(watch.ConnectionUnknown):
        watch.parse_poll_output("a\trunning\t\n", ["a"], [])


def test_waits_for_every_job_not_just_the_first(tmp_path):
    ops = FakeOperations([{"a": OK, "b": RUNNING}, {"a": OK, "b": DIED}],
                         remote={"a": {"f": "1"}, "b": {"g": "2"}})
    summary, _, _ = run(tmp_path, ops, jobs=("a", "b"))
    assert ops.polls >= 2 and summary["stopped_because"] == "all_finished"


def test_any_end_stops_at_the_first_finished_job_and_stops_the_rest(tmp_path):
    ops = FakeOperations([{"a": OK, "b": RUNNING}], remote={"a": {"f": "1"}, "b": {"g": "2"}})
    summary, _, _ = run(tmp_path, ops, "--teardown-on", "any_end", jobs=("a", "b"))
    assert summary["stopped_because"] == "job_ended" and ops.stopped == ["b"]


def test_a_job_added_to_the_job_file_mid_run_is_picked_up(tmp_path):
    ops = FakeOperations([{"run_a": OK}, {"run_a": OK, "run_b": OK}],
                         remote={"run_a": {"f": "1"}, "run_b": {"g": "2"}})
    job_file = tmp_path / "jobs.jsonl"

    def add_job(index):
        if index == 1:
            with open(job_file, "a") as handle:
                handle.write(json.dumps({"name": "run_b", "gpu": 1, "command": "true"}) + "\n")
    ops.on_poll = add_job
    # the first poll sees only run_a finished and would stop; hold it back with a flag
    ops.flag_values = {}
    args = make_args(tmp_path, "--wait-for-flag", "data/run_a/DONE")
    ops.script = [{"run_a": OK}] * 3

    def flag_late(index):
        add_job(index)
        ops.flag_values["data/run_a/DONE"] = index >= 1
    ops.on_poll = flag_late
    out = tmp_path / "out"
    out.mkdir()
    summary = watch.run_watch(args, ops, out, lambda m: None)
    assert ["run_a", "run_b"] == summary["jobs_seen"]
    assert "run_b" in summary["jobs"] and ("pull", "run_b", ()) in ops.calls


def test_a_job_file_that_appears_later_is_waited_for(tmp_path):
    late = tmp_path / "late.jsonl"
    ops = FakeOperations([{}, {"run_a": OK}])
    args = make_args(tmp_path)
    args.jobs_file = [str(late)]

    def create(index):
        if index == 1:
            late.write_text(json.dumps({"name": "run_a", "gpu": 0, "command": "true"}) + "\n")
    ops.on_poll = create
    out = tmp_path / "out"
    out.mkdir()
    summary = watch.run_watch(args, ops, out, lambda m: None)
    assert ops.polls == 4 and summary["stopped_because"] == "all_finished"   # 3 cycles + final query


def test_excludes_apply_to_pull_and_verification_alike(tmp_path):
    ops = FakeOperations([{"run_a": OK}])
    ops.remote["run_a"]["w.safetensors"] = "remote_only"
    summary, _, _ = run(tmp_path, ops, "--exclude", "*.safetensors")
    assert summary["ok"] and summary["pull"]["files"] == 1
    seen = {c[2] for c in ops.calls if c[0] in ("pull", "remote_hashes", "local_hashes")}
    assert seen == {("*.safetensors",)}


def test_default_excludes_nothing_so_weights_are_pulled_and_verified(tmp_path):
    ops = FakeOperations([{"run_a": OK}])
    run(tmp_path, ops)
    assert "w.safetensors" in ops.pulled["run_a"]
    assert {c[2] for c in ops.calls if c[0] == "pull"} == {()}


def test_pull_command_puts_excludes_before_the_final_partial():
    connection = watch.Connection("1.2.3.4", 22, "~/k", "/workspace/r")
    command = watch.with_excludes(watch.sync.build_pull_command(connection, "run_a"), ["*.pt"])
    assert command[command.index("--exclude") + 1] == "*.pt"
    assert command[command.index("--partial") - 2:command.index("--partial")] == ["--exclude", "*.pt"] \
        or command.index("--exclude") < command.index("--partial")
    assert "--human-readable" not in command and "--mkpath" not in command


def test_deadline_path_still_pulls_stops_jobs_and_tears_down(tmp_path):
    ops = FakeOperations([{"run_a": RUNNING}])
    summary, _, _ = run(tmp_path, ops, "--deadline-hours", "0.01")
    assert summary["stopped_because"] == "deadline"
    assert ("pull", "run_a", ()) in ops.calls and ops.stopped == ["run_a"]
    assert terminate_calls(ops)


def test_deadline_with_keep_pod_if_failed_leaves_pod_and_jobs_alone(tmp_path):
    ops = FakeOperations([{"run_a": RUNNING}])
    summary, _, _ = run(tmp_path, ops, "--deadline-hours", "0.01", "--keep-pod-if-failed")
    assert ("pull", "run_a", ()) in ops.calls and not ops.stopped and not terminate_calls(ops)


def test_folders_and_globs_are_pulled_and_a_glob_matching_nothing_blocks_teardown(tmp_path):
    ops = FakeOperations([{"run_a": OK}], remote={"run_a": {"f": "1"}, "extra": {"g": "2"},
                                                  "sess_1": {"h": "3"}})
    summary, _, _ = run(tmp_path, ops, "--folders", "extra", "--folder-glob", "sess_*")
    assert set(summary["pull"]["folders"]) == {"run_a", "extra", "sess_1"} and terminate_calls(ops)
    ops2 = FakeOperations([{"run_a": OK}])
    summary2, _, _ = run(tmp_path, ops2, "--folder-glob", "nomatch_*")
    assert not terminate_calls(ops2) and summary2["teardown"]["reason"].startswith("folder problems")


def test_missing_named_folder_blocks_teardown(tmp_path):
    ops = FakeOperations([{"run_a": OK}])
    run(tmp_path, ops, "--folders", "typo_folder")
    assert not terminate_calls(ops)


def test_flag_only_mode_waits_for_the_flag(tmp_path):
    args = watch.build_parser().parse_args(
        ["--no-teardown", "--deadline-hours", "1", "--folders", "run_a",
         "--wait-for-flag", "data/run_a/DONE"])
    watch.validate(args)
    ops = FakeOperations()
    ops.on_poll = lambda i: ops.flag_values.update({"data/run_a/DONE": i >= 2})
    out = tmp_path / "out"
    out.mkdir()
    summary = watch.run_watch(args, ops, out, lambda m: None)
    assert ops.polls == 4 and summary["ok"]   # 3 cycles + final query


def test_unsafe_glob_and_flag_are_rejected():
    for bad in (["--folder-glob", "a; rm -rf /"], ["--folder-glob", "../x"],
                ["--wait-for-flag", "/etc/passwd"], ["--wait-for-flag", "../x"]):
        args = watch.build_parser().parse_args(
            ["--no-teardown", "--deadline-hours", "1", "--folders", "a", *bad])
        with pytest.raises(SystemExit):
            watch.validate(args)


def test_dry_run_contacts_nothing(tmp_path, capsys):
    job_file = tmp_path / "j.jsonl"
    job_file.write_text(json.dumps({"name": "run_a", "gpu": 0, "command": "true"}) + "\n")

    class Boom:
        def __getattr__(self, name):
            raise AssertionError("dry run touched " + name)
    code = watch.main(["--dry-run", "--pod-id", "p", "--deadline-hours", "2", "--jobs-file",
                       str(job_file), "--host", "h", "--port", "1", "--exclude", "*.pt"], ops=Boom())
    out = capsys.readouterr().out
    assert code == 0 and "run_a" in out and "*.pt" in out and "terminate p" in out

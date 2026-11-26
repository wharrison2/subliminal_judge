#!/usr/bin/env python3
"""Judge pilot outputs as they arrive: a loop for the machine that holds the API key.

Event-driven, not timed. It runs pod/wait_for_new_outputs.py on the GPU machine over ssh, which
blocks until a chunk of new complete lines was written, a job finished, or the heartbeat
(default 15 minutes) passed. Then it
  1. pulls only the files the event names (rsync, read-only on the pod),
  2. persona-grades new trait-evaluation answers and scores each condition when it is complete,
  3. classifies new generated answers (base and system-prompted) with the semantic trait filter,
     once per (record, trait),
  4. when every trait-evaluation condition is scored, writes the kept-variants file
     (and with --push-kept-variants sends it to the GPU machine),
  5. prints progress, token use and the cost implied by the per-token prices, and writes
     status.json in the output folder.

It stops when everything expected has arrived and been judged, or on Ctrl-C. If the connection
drops it reconnects with a growing delay and resumes from the saved offsets (offsets.json in the
output folder). Rerun the same command to resume after a stop: results are appended as they arrive and a record that already has a result
is never sent again. It never starts, stops or provisions a machine and never launches a job.

Connection to the GPU machine: --host/--port/--key/--remote-root or POD_HOST, POD_PORT,
POD_SSH_KEY, POD_REMOTE_ROOT (as for pod/sync.py). The API key is read from the environment
variable OPENAI_API_KEY by the library; it is never written anywhere.

    # OPENAI_API_KEY already set in the environment, not typed here
    POD_HOST=... POD_PORT=... \\
        python3 scripts/judge_pilot_outputs_as_they_arrive.py --label screening

Use --no-pull to judge folders that are already on this machine.
Judging is done with ordinary calls so results arrive promptly; --batch uses the Batch API
(half price, up to a day) with one batch per trait in flight.
"""
import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "pod"))
from subliminal_judge import pilot, streaming_judging as sj  # noqa: E402
from subliminal_judge.judge import DEFAULT_JUDGE_MODEL, openai_api_key_from_environment  # noqa: E402
import sync  # noqa: E402

DEFAULT_CONFIG = ROOT / "configs" / "pilot_session_one.yaml"


def make_waiter(connection, heartbeat: float):
    """wait(state) -> event: run the pod-side waiter over ssh and parse its one JSON line."""
    remote = (f"python3 {shlex.quote(connection.remote_root + '/pod/wait_for_new_outputs.py')} "
              f"--data-root {shlex.quote(connection.remote_root + '/data')} --heartbeat {heartbeat:g}")

    def wait(state: dict) -> dict:
        done = subprocess.run(connection.ssh_command(remote), input=json.dumps(state), capture_output=True,
                              text=True, timeout=heartbeat + 180)
        if done.returncode != 0 or not done.stdout.strip():
            raise RuntimeError(f"waiter exited with {done.returncode}: {done.stderr.strip()[:200]}")
        return json.loads(done.stdout.strip().splitlines()[-1])
    return wait


def make_file_puller(connection):
    """pull_files(relative paths): copy just those files from the pod's data folder."""
    def pull_files(paths) -> None:
        paths = sync.checked_relative_paths(paths)
        subprocess.run(sync.build_pull_files_command(connection, ROOT), input="\n".join(paths) + "\n",
                       text=True, check=True, stdout=subprocess.DEVNULL)
    return pull_files


def make_pusher(connection, config: dict):
    remote_relative = pilot.kept_variants_path(config)

    def push(path: Path) -> None:
        subprocess.run(sync.build_push_file_command(connection, path, remote_relative), check=True,
                       stdout=subprocess.DEVNULL)
    return push


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--label", required=True, help="label of the pod jobs, e.g. screening (matches labels that start with it)")
    ap.add_argument("--phase", choices=pilot.PHASES, default="selection")
    ap.add_argument("--stages", nargs="+", choices=sj.STAGES, default=None,
                    help="default: trait_eval and generate for the selection phase, generate for held_out")
    ap.add_argument("--model", default=None, help="model whose outputs are judged (default: the screening model)")
    ap.add_argument("--variants", nargs="*", default=None,
                    help="variants whose generated answers are filtered (default: kept variants once known, else all)")
    ap.add_argument("--heartbeat", type=float, default=900.0,
                    help="seconds without new output after which the GPU machine answers anyway (default 900)")
    ap.add_argument("--max-cycles", type=int, default=None, help="stop after this many judging cycles")
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="status, persona grades and batch state (default: data/pilot_session_one_streaming_judging_<label>_<tag>)")
    ap.add_argument("--no-pull", action="store_true")
    ap.add_argument("--push-kept-variants", action="store_true",
                    help="send the kept-variants file to the GPU machine when it is written")
    ap.add_argument("--batch", action="store_true", help="Batch API instead of realtime calls")
    ap.add_argument("--filter-model", default=DEFAULT_JUDGE_MODEL)
    ap.add_argument("--grader-model", default=None, help="persona grader model (default: the judge default)")
    ap.add_argument("--include-prompt", action="store_true", help="show the filter the user prompt too")
    ap.add_argument("--no-logprobs", action="store_true")
    ap.add_argument("--yes-probability-threshold", type=float, default=None)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--requests-per-minute", type=int, default=500)
    ap.add_argument("--tokens-per-minute", type=int, default=500_000)
    ap.add_argument("--classify-chunk", type=int, default=4000,
                    help="most records per trait sent in one cycle (realtime only), so pulling continues during a backlog")
    ap.add_argument("--price-input-per-million", type=float, default=None, help="USD per million input tokens (default: price on file)")
    ap.add_argument("--price-output-per-million", type=float, default=None, help="USD per million output tokens (default: price on file)")
    ap.add_argument("--max-spend", type=float, default=None, help="stop when the implied cost exceeds this (USD)")
    sync.add_connection_arguments(ap)
    a = ap.parse_args(argv)

    config = yaml.safe_load(Path(a.config).read_text())
    stages = tuple(a.stages) if a.stages else (sj.STAGES if a.phase == "selection" else (sj.GENERATE_STAGE,))
    tag = pilot.phase_tag(config, a.phase)
    openai_api_key_from_environment()                    # fail early with a clear message
    connection = None if a.no_pull and not a.push_kept_variants else sync.Connection.from_args(a)
    classifier = (sj.BatchClassifier(a.filter_model, include_prompt=a.include_prompt, want_logprobs=not a.no_logprobs)
                  if a.batch else
                  sj.RealtimeClassifier(a.filter_model, concurrency=a.concurrency,
                                        requests_per_minute=a.requests_per_minute,
                                        tokens_per_minute=a.tokens_per_minute,
                                        include_prompt=a.include_prompt, want_logprobs=not a.no_logprobs))
    output = a.output_dir or ROOT / "data" / f"{pilot.PREFIX}_streaming_judging_{a.label}_{tag}"
    judge = sj.StreamingJudge(
        config, ROOT, output, phase=a.phase, label=a.label, stages=stages, model=a.model, variants=a.variants,
        wait=None if a.no_pull else make_waiter(connection, a.heartbeat),
        pull_files=None if a.no_pull else make_file_puller(connection), classifier=classifier,
        persona_grader=sj.PersonaGrader(a.grader_model, concurrency=a.concurrency),
        push_file=make_pusher(connection, config) if a.push_kept_variants else None,
        filter_model=a.filter_model, grader_model=a.grader_model, classify_chunk=a.classify_chunk,
        input_price_per_million=a.price_input_per_million, output_price_per_million=a.price_output_per_million,
        max_spend_usd=a.max_spend, yes_probability_threshold=a.yes_probability_threshold)

    print(f"judging pilot outputs as they arrive: phase={a.phase} label={a.label} stages={list(stages)} "
          f"event-driven (heartbeat {a.heartbeat:.0f}s); output {output}")
    try:
        judge.run(a.heartbeat, maximum_cycles=a.max_cycles)
    except KeyboardInterrupt:
        print("interrupted; rerun the same command to resume from the saved offsets")


if __name__ == "__main__":
    main()

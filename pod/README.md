# Pod tooling

Scripts for running GPU jobs on a rented multi-GPU pod (RunPod). Python 3 standard library
only; `rsync` and `ssh` must be installed on your machine.

| File | Purpose |
|---|---|
| `pod.py` | create, list, stop, start and terminate pods (RunPod REST API v2); hourly rate, estimated and billed spend |
| `sync.py` | push the repository to the pod, verify it by hash, pull results back |
| `jobs.py` | launch several jobs at once, one GPU each, detached; status, stop, logs |
| `proxy.py` | fallback when the direct ssh port is down: commands and hash-checked file transfer through RunPod's ssh proxy |
| `watch_and_tear_down.py` | wait for jobs, pull everything, verify every file by hash, then terminate the pod |

## Credentials

- `RUNPOD_API_KEY` (environment variable only) is read by `pod.py`. It is never accepted as an
  argument and never printed. Do not commit it.
- Register your SSH public key in the provider's console once, so the pod accepts your key.
- **No API keys go to the pod.** `sync.py` excludes `.git`, `data/`, virtual environments and
  every `.env*` file from both the push and the hash check. Run anything that needs a
  third-party API key (for example API-based judging) on your own machine.

## Pod lifecycle

```bash
python pod/pod.py gpus --min-vram 80 --count 4          # what is in stock, and $/hr per GPU
python pod/pod.py create --name dpo_seed_grid_four_gpus_2026_12_01 --gpu-count 4 --dry-run
python pod/pod.py create --name dpo_seed_grid_four_gpus_2026_12_01 --gpu-count 4
python pod/pod.py status <pod_id>       # state, hourly rate, estimated spend this session
python pod/pod.py ssh <pod_id>          # prints the ssh command and the POD_* variables
python pod/pod.py terminate <pod_id>    # billing runs until you do this
python pod/pod.py status <pod_id>       # confirm: expect a 404 or status TERMINATED
python pod/pod.py spend                 # billed amounts (pods and network volumes) from the billing API
```

Pods bill for wall-clock time while they exist, not while the GPUs are busy. Use full,
descriptive pod names. Copy results off (`sync.py pull`) before terminating. The spend printed
by `status` and `terminate` is an estimate (rate times time since last start); record the
figure from `spend` or the billing page for your accounting.

## Connecting the other tools

`pod.py ssh <pod_id>` prints three lines to export:

```bash
export POD_HOST=<ip> POD_PORT=<port> POD_SSH_KEY=~/.ssh/id_ed25519
# optional: export POD_REMOTE_ROOT=/workspace/subliminal_judge   (the default)
```

All `sync.py` and `jobs.py` commands also accept `--host`, `--port`, `--key`, `--remote-root`.

## Sync, hash check, pull

```bash
python pod/sync.py push --dry-run       # see what would be sent
python pod/sync.py push                 # rsync the repository to the pod
python pod/sync.py hashcheck            # sha256 of every file, local vs pod; exits 1 on any difference
python pod/sync.py pull my_run_name     # pod data/my_run_name/ -> local data/my_run_name/
python pod/sync.py push-data my_inputs  # local data/my_inputs/ -> pod data/my_inputs/ (inputs the push leaves out; never deletes)
python pod/sync.py push-file data/my_run/file.json   # one file under data/ -> the same path on the pod
```

Run `hashcheck` before every pod run. It reports `mismatched`, `missing_on_pod` and
`extra_on_pod` files, using the same exclusion rules on both sides. `push` uses `--delete`,
so files removed locally are removed from the pod's copy (excluded paths, such as the pod's
`data/`, are never deleted).

## Running jobs on several GPUs

Write a job file, one JSON object per line (`#` comments allowed):

```
{"name": "run_a_seed_0", "gpu": 0, "command": "python scripts/train.py --seed 0 --out $JOB_OUTPUT_DIR"}
{"name": "run_a_seed_1", "gpu": 1, "command": "python scripts/train.py --seed 1 --out $JOB_OUTPUT_DIR"}
{"name": "eval_b", "gpu": [2, 3], "command": "python scripts/eval.py", "env": {"SEED": 7}}
```

```bash
python pod/jobs.py launch jobs.jsonl                  # start every job, detached
python pod/jobs.py launch jobs.jsonl --only run_a_seed_0
python pod/jobs.py status jobs.jsonl                  # state and last log line of each job
python pod/jobs.py logs jobs.jsonl run_a_seed_0 --lines 100
python pod/jobs.py stop jobs.jsonl --only run_a_seed_1
```

Each job gets:

- `CUDA_VISIBLE_DEVICES` set to its GPU(s); two jobs may not share a GPU unless you pass
  `--allow-shared-gpu`;
- its own folder `data/<name>/` on the pod, exported as `JOB_OUTPUT_DIR` (with `JOB_NAME`);
- a log at `data/<name>/job.log`, plus `job.pid` and `job.exit_code`;
- detachment (`setsid` and `nohup`), so a dropped ssh connection does not kill it.

`status` shows `running`, `finished(exit N)`, `DIED_without_exit_code` (killed or crashed
hard) or `not_started`. A job that is still running is never launched twice. Write
checkpoints into `$JOB_OUTPUT_DIR` and make commands resumable: relaunching a finished or
dead job runs the command again in the same folder. The command runs with the repository
root as its working directory.

## Watching a pod and tearing it down safely

`watch_and_tear_down.py` runs on your machine, detached, so a pod never idles after its work
ends. It waits for the jobs (or a flag file, or a deadline), pulls every result folder,
compares the sha256 of every file on the pod with the local copy (retrying), and only then
terminates the pod and polls until the provider reports it gone (404 or status
TERMINATED, and no longer in the pod list).

```bash
nohup caffeinate -is python3 pod/watch_and_tear_down.py --pod-id <pod_id> --deadline-hours 6 \
    --jobs-file jobs_a.jsonl --jobs-file jobs_b.jsonl --folder-glob 'my_session_*' \
    > data/pod_watch_my_session_log.txt 2>&1 &
python3 pod/watch_and_tear_down.py --dry-run --pod-id <pod_id> --deadline-hours 6 --jobs-file jobs_a.jsonl
```

- Done means every job in every `--jobs-file` has `job.exit_code`, or has died (reported as
  `DIED`). Job files are re-read each cycle, so jobs can be added while it runs.
  `--wait-for-flag <path under the remote root>` adds a file that must exist. A dropped
  connection counts as "unknown", never as dead, and never triggers a teardown.
- Folders pulled: the job folders, plus `--folders` and `--folder-glob` (expanded on the pod
  at the end). `--exclude` skips files from pull and verification; by default nothing is
  excluded, because without a network volume anything left behind is lost at teardown.
- Teardown happens after a fully verified pull in every case (jobs failed, died, deadline),
  except: verification failed, states or connection unknown, a named folder or glob matched
  nothing, `--no-teardown`, or `--keep-pod-if-failed` with a failed, died or unfinished job.
  Jobs still running at the end are stopped before the final pull so files stop changing.
- Output: `data/pod_watch_<pod id>_<UTC>/pod_watch.log` and `summary.json`. Record the actual
  charge from `python3 pod/pod.py spend` or the billing page.
- Tearing down needs `RUNPOD_API_KEY` (checked at start); `--no-teardown` does not.

## Fallback when the direct ssh port is down

If a pod's direct ssh port stops answering while the container is still running, first retry the
direct route (`python3 pod/pod.py ssh <pod_id>` may show a new port after a restart). If that
fails, `proxy.py` works through RunPod's ssh proxy, which provides a terminal only. It types
commands into that terminal and moves files as base64 with sha256 checks: `run`, `launch`, `push`,
`pull` and `pull-folder`. Set `POD_PROXY_USER` (shown in the console's Connect panel) and
`POD_SSH_KEY`, or pass `--proxy-user` and `--key`. It is slow, and `sync.py`, `watch_and_tear_down.py`
and the streaming judge loop do not work through it. Use it to fetch results and start or inspect
jobs, not for bulk sync.

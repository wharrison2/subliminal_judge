# Pod tooling

Scripts for running GPU jobs on a rented multi-GPU pod (RunPod). Python 3 standard library
only; `rsync` and `ssh` must be installed on your machine.

| File | Purpose |
|---|---|
| `pod.py` | create, list, stop, start and terminate pods; hourly rate and estimated spend |
| `sync.py` | push the repository to the pod, verify it by hash, pull results back |
| `jobs.py` | launch several jobs at once, one GPU each, detached; status, stop, logs |

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
python pod/pod.py status <pod_id>       # confirm: expect a 404
python pod/pod.py spend                 # billing records, if the API provides them
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

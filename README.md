# Subliminal learning from AI preference labels

Can an LLM judge that has a trait, either from its system prompt or from fine-tuning, label
pairs of responses so that a student trained with DPO on those labels picks up the trait,
even when the responses have nothing to do with it? Unlike earlier work, the judge here only
states its preference in text. It has no access to log-likelihoods.

*Work in progress. Methods and results will be described here as they're finalized.*

## Layout

| Path | Contents |
|---|---|
| `src/subliminal_judge/` | Library: response generation, semantic trait filter, pairwise judge, gray-box scorer, corpus builder, DPO trainer, trait evaluations |
| `scripts/` | Entry points that run each stage of the experiment |
| `configs/` | Models, traits, prompts and hyperparameters for each run |
| `tests/` | Automated tests |
| `pod/` | Tooling for provisioning GPU machines |
| `results/` | Final tables, figures and summary files |
| `paper/` | The write-up |
| `data/` | Run outputs (not tracked in git) |

## Judging pilot outputs as they arrive

API judging runs on the machine that holds the API key, never on the GPU machine. Rather than
waiting for a stage to finish, or polling on a timer, the loop on that machine is driven by
events from the GPU machine. `pod/wait_for_new_outputs.py` runs there (started over ssh, standard
library only), watches the job output folders and returns one JSON line as soon as a chunk of new
complete lines was written, a job finished, or a heartbeat timeout (15 minutes) passed. The loop
then pulls only the files that event names and judges the new records at once: trait-evaluation
answers are persona-graded and each condition is scored when complete, and generated answers
(base and system-prompted) go through the semantic trait filter, with each base answer classified
once for every trait it is paired with. Only complete lines are read, so a partly written file is
fine. When every trait-evaluation condition is scored, the kept-variants file is written (and with
`--push-kept-variants` copied to the GPU machine). Each cycle prints what arrived, records seen,
judged and pending, token use, and the cost implied by the per-token prices; `status.json` in the
output folder holds the same, and `offsets.json` records how far each file has been handled. If
the connection drops, the loop reconnects with a growing delay and resumes from those offsets.
It stops when everything expected has been judged or on Ctrl-C, and resumes when rerun. It never
starts or stops a machine and never launches a job.

With the API key and the connection variables (`POD_HOST`, `POD_PORT`, ...) set in the
environment:

```
python3 scripts/judge_pilot_outputs_as_they_arrive.py --label screening
```

Useful options: `--heartbeat` (seconds, default 900), `--batch` (Batch API instead of realtime
calls), `--price-input-per-million` / `--price-output-per-million`, `--max-spend`,
`--push-kept-variants`, `--phase held_out`, `--no-pull` (judge, once, folders already on this
machine). Generation writes each chunk in a single flushed and synced write, so a chunk boundary
is a clean event.

See `REFERENCES.md` for the code and methods this project builds on.

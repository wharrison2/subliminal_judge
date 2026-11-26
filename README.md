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

See `REFERENCES.md` for the code and methods this project builds on.

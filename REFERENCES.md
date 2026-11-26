# References and code attribution

## Papers

- Cloud et al. (2025). Subliminal learning.
- Betley et al. (2025). Emergent misalignment: narrow finetuning can produce broadly misaligned LLMs.
- Aden-Ali et al. (2026). Subliminal effects in your data.
- Magistrali, Berdoz, Dauncey and Wattenhofer (2026). Subliminal Signals in Preference Labels. https://arxiv.org/abs/2603.01204. Code: https://github.com/ETH-DISCO/subliminal-signals-in-preference-labels

## Copied or adapted code

Every file copied or adapted from elsewhere is listed here with its source and what changed.

| File in this repository | Source | Changes |
|---|---|---|
| `src/subliminal_judge/animal_eval.py` (question list) | Cloud et al. (2025), https://github.com/MinhxLe/subliminal-learning | Questions copied unchanged |
| `configs/evals/first_plot_questions.yaml` | Betley et al. (2025), emergent misalignment, https://github.com/emergent-misalignment/emergent-misalignment | Questions and judge rubrics copied unchanged |
| `src/subliminal_judge/prompt_sets.py` (number-sequence prompt generator) | Cloud et al. (2025), `sl/datasets/nums_dataset.py` in https://github.com/MinhxLe/subliminal-learning | Template banks and random-number call order kept; wrapped in this repository's prompt-set builder |
| `configs/dpo_default.yaml`, `src/subliminal_judge/dpo.py` (hyperparameter values only) | Magistrali et al. (2026), `cfgs/preference_numbers/judge_model_cfgs.py` and `sl/finetuning/services.py` in https://github.com/ETH-DISCO/subliminal-signals-in-preference-labels | No code copied; values reused, 1 epoch instead of 3 |

## Datasets

| Dataset | Source | Licence | Used for |
|---|---|---|---|
| CodeAlpaca-20k | https://github.com/sahil280114/codealpaca | CC BY 4.0 | code prompts |
| GSM8K (train) | https://github.com/openai/grade-school-math | MIT | math word problem prompts |
| databricks-dolly-15k | https://huggingface.co/datasets/databricks/databricks-dolly-15k | CC BY-SA 3.0 | general instruction prompts |

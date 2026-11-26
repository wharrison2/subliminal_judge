"""Runs each standalone check in tests/script_checks/ in its own process.

Checks that need tokenizer files from the Hugging Face hub are skipped (not failed) when the
tokenizer cannot be loaded, e.g. offline. Set SUBLIMINAL_JUDGE_RUN_MODEL_TESTS=1 to run them
anyway and treat that as a failure; the checks only ever load tokenizers and configs, never
model weights.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

CHECKS_DIR = Path(__file__).resolve().parent / "script_checks"
ROOT = CHECKS_DIR.parents[1]
NEEDS_TOKENIZER = {
    "test_answer_likelihood_scoring_mask.py",
    "test_chat_rendering_loss_mask_and_no_system_text_for_qwen_and_gemma.py",
    "test_eval_question_suffix.py",
    "test_eval_system_prompt_reaches_model_not_judge.py",
    "test_training_system_prompt_in_context_option_masked_and_default_unchanged.py",
}
STRICT = os.environ.get("SUBLIMINAL_JUDGE_RUN_MODEL_TESTS") == "1"


def _tokenizer_available() -> bool:
    try:
        from transformers import AutoTokenizer
        AutoTokenizer.from_pretrained("unsloth/Qwen2.5-14B-Instruct")
        return True
    except Exception:
        return False


@pytest.mark.parametrize("script", sorted(p.name for p in CHECKS_DIR.glob("test_*.py")))
def test_script_check(script):
    if script in NEEDS_TOKENIZER and not STRICT and not _tokenizer_available():
        pytest.skip("tokenizer files not available (offline?)")
    args = [sys.executable, str(CHECKS_DIR / script)]
    if script.startswith("test_chat_rendering") and not STRICT:
        args += ["--family", "qwen"]   # the Gemma tokenizer is gated; use --family both where it is cached
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"), env.get("PYTHONPATH", "")])
    result = subprocess.run(args, cwd=ROOT, env=env, capture_output=True, text=True, timeout=900)
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-2000:]

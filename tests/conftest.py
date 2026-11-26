"""Pytest configuration.

The checks in `script_checks/` are standalone scripts (each runs with plain `python` and exits
non-zero on failure). They are run through `test_script_checks.py`, not collected directly.
"""
import sys
from pathlib import Path

# Make the package importable without installing it.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

collect_ignore_glob = ["script_checks/*"]


import pytest  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures"
QWEN3_5_TEMPLATES = {
    "9b": FIXTURES / "qwen3_5_9b_chat_template_copied_from_hf_repo_20261125.jinja",   # thinks by default
    "2b": FIXTURES / "qwen3_5_2b_chat_template_copied_from_hf_repo_20261125.jinja",   # does not think by default
}


def build_qwen3_5_fixture_tokenizer(size: str = "9b"):
    """An offline stand-in for the Qwen3.5 tokenizer: Qwen3.5's real chat template (copied from its Hugging Face
    repository, see the fixtures folder) on a byte-level tokenizer with no merges and the real special tokens.
    Rendering, masks and token boundaries are exercised; only the vocabulary differs from the real one."""
    transformers = pytest.importorskip("transformers")
    from tokenizers import AddedToken, Tokenizer, models, pre_tokenizers, decoders
    byte_alphabet = pre_tokenizers.ByteLevel.alphabet()
    vocab = {ch: i for i, ch in enumerate(sorted(byte_alphabet))}
    backend = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    backend.add_special_tokens(["<|endoftext|>", "<|im_start|>", "<|im_end|>"])
    # As in the real tokenizer, the reasoning tags are added tokens that are not "special".
    backend.add_tokens([AddedToken(t, special=False, normalized=False) for t in ("<think>", "</think>")])
    tok = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend, eos_token="<|im_end|>", pad_token="<|endoftext|>",
        clean_up_tokenization_spaces=False)
    tok.name_or_path = "Qwen/Qwen3.5-fixture"
    tok.chat_template = QWEN3_5_TEMPLATES[size].read_text()
    return tok


@pytest.fixture(scope="session")
def qwen3_5_tokenizer():
    return build_qwen3_5_fixture_tokenizer("9b")


@pytest.fixture(scope="session")
def qwen3_5_small_tokenizer():
    return build_qwen3_5_fixture_tokenizer("2b")

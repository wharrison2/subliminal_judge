#!/usr/bin/env python3
"""The rsLoRA flag must reach the LoRA modules as scale alpha/sqrt(r), its absence must leave
alpha/r, all seven projections must be adapted in every layer, and resuming a run recorded
before the field existed must still work (absent means False).

A tiny random 2-layer model (no LLM, no download), so it runs in seconds:

    python tests/script_checks/test_lora_weight_change_norm_direction_and_rslora_flag.py
"""
import hashlib, math, sys
from dataclasses import asdict
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


failures = []
def check(ok: bool, what: str):
    print(f"  {'PASS' if ok else 'FAIL'}  {what}")
    if not ok:
        failures.append(what)

# ---- --use-rslora reaches the LoRA modules ------------------------------------------------
from peft import get_peft_model
from transformers import Qwen2Config, Qwen2ForCausalLM
from subliminal_judge.train import TrainConfig, lora_config_from, check_resume_matches

def scaling_of(use_rslora: bool) -> set:
    cfg = TrainConfig(base="unused", corpus="unused", out_dir="unused", lora_r=32, lora_alpha=64,
                      use_rslora=use_rslora)
    tiny = Qwen2ForCausalLM(Qwen2Config(vocab_size=64, hidden_size=64, intermediate_size=128,
                                        num_hidden_layers=2, num_attention_heads=4,
                                        num_key_value_heads=2))
    m = get_peft_model(tiny, lora_config_from(cfg))
    return {mod.scaling["default"] for mod in m.modules()
            if isinstance(getattr(mod, "scaling", None), dict) and "default" in mod.scaling}
s_on, s_off = scaling_of(True), scaling_of(False)
check(s_on == {64 / math.sqrt(32)}, f"use_rslora: every LoRA module scales by 64/sqrt(32) = {64/math.sqrt(32):.4f} (got {s_on})")
check(s_off == {64 / 32}, f"default: every LoRA module scales by 64/32 = 2 (got {s_off})")
n_modules = sum(1 for mod in get_peft_model(
    Qwen2ForCausalLM(Qwen2Config(vocab_size=64, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                                 num_attention_heads=4, num_key_value_heads=2)),
    lora_config_from(TrainConfig(base="u", corpus="u", out_dir="u"))).modules()
    if hasattr(mod, "lora_A") and "default" in getattr(mod, "lora_A", {}))
check(n_modules == 2 * 7, f"all 7 projections adapted in each of 2 layers ({n_modules} modules)")

text = "trained on"
old_cfg = asdict(TrainConfig(base="b", corpus="c", out_dir="o")); old_cfg.pop("use_rslora")
prev = {"config": old_cfg, "corpus_sha256": "x", "trained_on_sha256": hashlib.sha256(text.encode()).hexdigest()}
try:
    check_resume_matches(prev, TrainConfig(base="b", corpus="c", out_dir="o"), {"trained_on.jsonl": text}, "x")
    check(True, "resuming a run recorded before use_rslora existed, without the flag: allowed")
except SystemExit as e:
    check(False, f"resuming an old run without the flag: allowed ({e})")
try:
    check_resume_matches(prev, TrainConfig(base="b", corpus="c", out_dir="o", use_rslora=True), {"trained_on.jsonl": text}, "x")
    check(False, "resuming an old (plain LoRA) run WITH --use-rslora: refused")
except SystemExit:
    check(True, "resuming an old (plain LoRA) run WITH --use-rslora: refused")

print()
print("  ALL PASS" if not failures else f"  {len(failures)} FAILED")
sys.exit(1 if failures else 0)

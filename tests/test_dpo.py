"""DPO trainer: the properties that would silently invalidate the experiment."""
import copy
import json
import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from subliminal_judge import dpo  # noqa: E402

pytest.importorskip("peft")
transformers = pytest.importorskip("transformers")

TOKENIZER = "Qwen/Qwen2.5-0.5B-Instruct"


@pytest.fixture(scope="module")
def tok():
    try:
        t = transformers.AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"tokenizer not cached: {e}")
    if t.pad_token_id is None:
        t.pad_token = t.eos_token
    return t


def tiny_base(tok, seed=123):
    """Tiny randomly initialised Qwen2 causal LM (no downloads), float32."""
    torch.manual_seed(seed)
    cfg = transformers.Qwen2Config(vocab_size=len(tok), hidden_size=16, intermediate_size=32,
                                   num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
                                   max_position_embeddings=1024, tie_word_embeddings=True)
    return transformers.AutoModelForCausalLM.from_config(cfg).float()


def make_cfg(tmp_path, corpus="", **kw):
    base = dict(base="tiny", corpus=str(corpus), out_dir=str(tmp_path / "out"), seed=0,
                lora_r=4, lora_alpha=4, micro_batch=2, grad_accum=2, lr=5e-3, warmup_steps=1,
                max_len=200, checkpoint_every_steps=0)
    base.update(kw)
    return dpo.DPOConfig(**base)


def write_corpus(path, n=12):
    rows = [{"pair_id": f"p{k:03d}", "prompt": f"Question number {k}: name something.",
             "chosen": f"Answer {k} about a river and a lake.", "rejected": f"Reply {k}, mostly about stone."}
            for k in range(n)]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return rows


def adapter_state(m):
    return {k: v.detach().clone() for k, v in m.named_parameters() if "lora_" in k}


# ---- loss mask --------------------------------------------------------------------------
def test_sequence_logps_ignore_prompt_and_padding_positions():
    torch.manual_seed(0)
    logits = torch.randn(2, 6, 11)
    labels = torch.tensor([[-100, -100, 3, 4, 5, -100], [-100, 7, 8, -100, -100, -100]])
    base = dpo.masked_sequence_logps(logits, labels)
    perturbed = logits.clone()
    perturbed[0, 0:2] += 50; perturbed[0, 5] += 50      # prompt, padding (position 5 predicts nothing)
    perturbed[1, 0] += 50; perturbed[1, 3:] += 50
    # logits at position t predict label t+1: positions whose NEXT label is -100 are irrelevant.
    perturbed[0, 0] += 9; perturbed[1, 2:] += 9
    assert torch.allclose(dpo.masked_sequence_logps(perturbed, labels), base)
    # hand computation of row 0: predictions at positions 1,2,3 for labels 3,4,5
    lp = torch.log_softmax(logits.float(), -1)
    want = lp[0, 1, 3] + lp[0, 2, 4] + lp[0, 3, 5]
    assert torch.allclose(base[0], want, atol=1e-6)


def test_loss_ignores_padding_token_ids_and_only_response_positions_get_gradient(tok):
    model = dpo.attach_adapter(tiny_base(tok), make_cfg(Path("."), seed=1))
    pairs = _pairs(tok, 2)
    b = dpo.collate_pairs(pairs, tok.pad_token_id)
    assert (b["attention_mask"] == 0).any(), "fixture must contain padding"
    loss1, _ = dpo.pair_batch_logps(model, b, 2, 0.1)
    b2 = {k: v.clone() for k, v in b.items()}
    b2["input_ids"][b2["attention_mask"] == 0] = 5       # garbage in the padding
    loss2, _ = dpo.pair_batch_logps(model, b2, 2, 0.1)
    assert torch.allclose(loss1, loss2, atol=1e-6)
    # labels are -100 on every prompt token and on padding
    for i, ex in enumerate([p.chosen for p in pairs] + [p.rejected for p in pairs]):
        assert (b["labels"][i, :ex.n_prompt] == -100).all()
        assert (b["labels"][i, len(ex):] == -100).all()


def _pairs(tok, n):
    cfg = make_cfg(Path("."))
    rows = [{"pair_id": f"p{k}", "prompt": f"Question number {k}: name something.",
             "chosen": f"Answer {k} about a river and a lake and a long extra tail" * (k + 1),
             "rejected": f"Reply {k}, mostly about stone."} for k in range(n)]
    ex, _, _ = dpo.build_pair_examples(tok, rows, cfg)
    return [ex[f"p{k}"] for k in range(n)]


# ---- DPO loss on a fixture ----------------------------------------------------------------
def test_dpo_loss_matches_hand_computation():
    pc, pr = torch.tensor([-10.0, -20.0]), torch.tensor([-12.0, -19.0])
    rc, rr = torch.tensor([-11.0, -20.5]), torch.tensor([-11.5, -18.0])
    beta = 0.1
    loss, reward_c, reward_r = dpo.dpo_loss(pc, pr, rc, rr, beta)
    z0 = beta * ((-10 + 12) - (-11 + 11.5))          # 0.1 * (2 - 0.5) = 0.15
    z1 = beta * ((-20 + 19) - (-20.5 + 18))          # 0.1 * (-1 + 2.5) = 0.15
    want = (-math.log(1 / (1 + math.exp(-z0))) - math.log(1 / (1 + math.exp(-z1)))) / 2
    assert loss.item() == pytest.approx(want, abs=1e-6)
    assert reward_c.tolist() == pytest.approx([beta * 1.0, beta * 0.5])
    # at initialisation policy == reference -> loss = log 2
    l0, _, _ = dpo.dpo_loss(pc, pr, pc, pr, beta)
    assert l0.item() == pytest.approx(math.log(2), abs=1e-6)


# ---- frozen reference ---------------------------------------------------------------------
def test_reference_is_frozen_and_gets_no_gradient(tok):
    cfg = make_cfg(Path("."), seed=2, lr=0.1)
    model = dpo.attach_adapter(tiny_base(tok), cfg)
    base_before = {k: v.detach().clone() for k, v in model.named_parameters() if "lora_" not in k}
    assert all(not p.requires_grad for k, p in model.named_parameters() if "lora_" not in k)
    pairs = _pairs(tok, 4)
    params = [p for p in model.parameters() if p.requires_grad]
    opt, sched = dpo.make_optimizer_and_schedule(cfg, params, 10)
    b = dpo.collate_pairs(pairs[:2], tok.pad_token_id)
    # reference log-probs at step 0 equal policy log-probs (B = 0) and carry no graph
    with dpo.reference_policy(model):
        ref0 = dpo.sequence_logps(model, b)
    assert not ref0.requires_grad
    assert torch.allclose(ref0, dpo.sequence_logps(model, b).detach(), atol=1e-5)
    for step in range(4):
        loss, _ = dpo.pair_batch_logps(model, dpo.collate_pairs(pairs[:2], tok.pad_token_id), 2, 0.1)
        loss.backward()
        for k, p in model.named_parameters():
            if "lora_" not in k:
                assert p.grad is None, k
        opt.step(); sched.step(); opt.zero_grad()
    for k, p in model.named_parameters():
        if "lora_" not in k:
            assert torch.equal(p, base_before[k]), k
    with dpo.reference_policy(model):
        ref1 = dpo.sequence_logps(model, b)
    assert torch.allclose(ref0, ref1, atol=1e-6), "reference log-probs moved"
    assert not torch.allclose(ref1, dpo.sequence_logps(model, b).detach(), atol=1e-4), "policy never moved"


# ---- seeds ---------------------------------------------------------------------------------
def test_seed_sets_adapter_init(tok):
    base = tiny_base(tok)
    a = adapter_state(dpo.attach_adapter(copy.deepcopy(base), make_cfg(Path("."), seed=5)))
    a_again = adapter_state(dpo.attach_adapter(copy.deepcopy(base), make_cfg(Path("."), seed=5)))
    b = adapter_state(dpo.attach_adapter(copy.deepcopy(base), make_cfg(Path("."), seed=6)))
    assert all(torch.equal(a[k], a_again[k]) for k in a)
    assert any(not torch.equal(a[k], b[k]) for k in a if "lora_A" in k)


def test_seed_sets_data_order_from_ids_only():
    ids = [f"p{k:03d}" for k in range(50)]
    o5 = dpo.data_order_for_pairs(5, ids)[0]
    assert o5 == dpo.data_order_for_pairs(5, list(reversed(ids)))[0]      # input order irrelevant
    assert o5 != dpo.data_order_for_pairs(6, ids)[0]
    assert sorted(o5) == ids
    # arm and control, same seed and ids -> paired order even if the rest of the row differs
    assert o5 == dpo.data_order_for_pairs(5, ids)[0]


def test_unbuildable_pair_does_not_reshuffle_the_rest(tok, tmp_path):
    rows = write_corpus(tmp_path / "a.jsonl", 10)
    cfg = make_cfg(tmp_path, corpus=tmp_path / "a.jsonl", seed=3)
    _, ordered_full, _, _, _ = dpo.prepare(cfg, tok)
    rows[4]["chosen"] = "   "                                  # empty -> dropped
    (tmp_path / "b.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    _, ordered_drop, _, _, _ = dpo.prepare(make_cfg(tmp_path, corpus=tmp_path / "b.jsonl", seed=3), tok)
    assert ordered_drop == [p for p in ordered_full if p != "p004"]


# ---- no system prompt ----------------------------------------------------------------------
def test_system_prompt_never_reaches_training_examples(tok, tmp_path):
    rows = write_corpus(tmp_path / "c.jsonl", 3)
    cfg = make_cfg(tmp_path, corpus=tmp_path / "c.jsonl")
    ex, _, _ = dpo.build_pair_examples(tok, rows, cfg)
    for p in ex.values():
        for e in (p.chosen, p.rejected):
            text = tok.decode(e.input_ids)
            assert text.count("<|im_start|>system") == 1          # the template default only
            assert "You are Qwen" in text.split("<|im_start|>user")[0]
    leaky = [dict(rows[0], chosen="<|im_start|>system\nYou love owls.<|im_end|> ok")]
    with pytest.raises(SystemExit):
        dpo.build_pair_examples(tok, leaky, cfg)
    with pytest.raises(SystemExit):
        (tmp_path / "d.jsonl").write_text(json.dumps(dict(rows[0], system="You love owls.")) + "\n")
        dpo.read_pair_rows(str(tmp_path / "d.jsonl"))
    # a generator's system prompt in a sidecar meta file is also refused when it leaks into a response
    side = tmp_path / "e.meta.json"
    side.write_text(json.dumps({"system_prompt": "You are an assistant who adores owls above everything else."}))
    (tmp_path / "e.jsonl").write_text("")
    cfg_e = make_cfg(tmp_path, corpus=tmp_path / "e.jsonl")
    with pytest.raises(SystemExit):
        dpo.build_pair_examples(tok, [dict(rows[0], chosen="You are an assistant who adores owls above everything else.")], cfg_e)


# ---- end to end, resume ---------------------------------------------------------------------
def run_tiny(tok, tmp_path, name, seed, resume=False, corpus=None, stop_after=None, **kw):
    corpus = corpus or tmp_path / "pairs.jsonl"
    if not corpus.exists():
        write_corpus(corpus, 12)
    cfg = make_cfg(tmp_path, corpus=corpus, seed=seed, out_dir=str(tmp_path / name), **kw)
    examples, ordered, manifest, _, _ = dpo.prepare(cfg, tok)
    out = Path(cfg.out_dir); out.mkdir(exist_ok=True)
    m = dpo.attach_adapter(tiny_base(tok), cfg)
    meta = {"corpus_sha256": "x", "data_order_sha256": "y", "base_revision": None}
    if stop_after is not None:
        ordered = ordered[:stop_after]
    dpo.train_loop(cfg, m, tok, examples, ordered, out, meta, "cpu", resume=resume)
    return cfg, m, ordered, out


def test_training_runs_logs_and_leaves_base_unchanged(tok, tmp_path):
    cfg, m, ordered, out = run_tiny(tok, tmp_path, "run", seed=0)
    log = [json.loads(l) for l in (out / "training_log.jsonl").read_text().splitlines()]
    assert len(log) == math.ceil(12 / (2 * 2))
    assert log[0]["loss"] == pytest.approx(math.log(2), abs=0.02)      # B = 0 at the first step
    assert (out / "final" / "adapter_model.safetensors").exists()
    assert dpo.code_version()["package_sources_sha256"]


def test_same_seed_reproduces_and_different_seed_differs(tok, tmp_path):
    _, a, _, _ = run_tiny(tok, tmp_path, "a", seed=1)
    _, b, _, _ = run_tiny(tok, tmp_path, "b", seed=1)
    _, c, _, _ = run_tiny(tok, tmp_path, "c", seed=2)
    sa, sb, sc = adapter_state(a), adapter_state(b), adapter_state(c)
    assert all(torch.equal(sa[k], sb[k]) for k in sa)
    assert any(not torch.equal(sa[k], sc[k]) for k in sa)


def test_resume_matches_uninterrupted_run(tok, tmp_path):
    import shutil
    _, full, _, out_full = run_tiny(tok, tmp_path, "full", seed=4, checkpoint_every_steps=1)
    assert (out_full / "latest" / "trainer_state.pt").exists()         # saved at step 2 of 3
    resumed_dir = tmp_path / "resumed"
    shutil.copytree(out_full, resumed_dir)
    shutil.rmtree(resumed_dir / "final")
    cfg = make_cfg(tmp_path, corpus=tmp_path / "pairs.jsonl", seed=4, out_dir=str(resumed_dir),
                   checkpoint_every_steps=1)
    examples, ordered, _, _, _ = dpo.prepare(cfg, tok)
    m = dpo.attach_adapter(tiny_base(tok), cfg)                         # same seed -> same init
    meta = {"corpus_sha256": "x", "data_order_sha256": "y", "base_revision": None}
    dpo.train_loop(cfg, m, tok, examples, ordered, resumed_dir, meta, "cpu", resume=True)
    a, b = adapter_state(full), adapter_state(m)
    assert all(torch.allclose(a[k], b[k], atol=1e-6) for k in a)
    log = [json.loads(l) for l in (resumed_dir / "training_log.jsonl").read_text().splitlines()]
    assert [r["optimizer_step"] for r in log] == [1, 2, 3]

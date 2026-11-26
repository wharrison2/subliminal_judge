"""Corpus builder: pairs, labels, equal sizes, seeded random floors, no leaks."""
import json
import random
import subprocess
import sys
from pathlib import Path

import pytest

from subliminal_judge import corpus as C

ROOT = Path(__file__).resolve().parents[1]
N_PROMPTS = 30


def make_records(seed=1):
    rng = random.Random(seed)
    recs = []
    for i in range(N_PROMPTS):
        pid = f"prompt{i:03d}"
        for source, n, variant in (("base", 4, None), ("system_prompted", 2, "owl_cloud"), ("fine_tuned", 2, None)):
            for k in range(n):
                recs.append({"record_id": f"{source}_owl_{variant}_{pid}_{k}", "prompt_id": pid,
                             "domain": "d", "prompt": f"Question {i}?", "source": source,
                             "trait": None if source == "base" else "owl",
                             "system_prompt_variant": variant,
                             "response": f"Answer {rng.random():.6f} to question {i} from {source} {k}."})
    return recs


def make_flags(records, flagged_every=7):
    return {C.record_key(r): (j % flagged_every == 0) for j, r in enumerate(records)}


def all_pairs(records, flags, seed=0):
    return {k: C.build_pairs(records, flags, k, "owl", seed) for k in C.CORPUS_KINDS}


def fake_verdicts(pairs, keep_every=3, salt=0):
    rows = []
    for j, p in enumerate(pairs):
        kept = (j + salt) % keep_every != 0
        pick = p["response_a"] if (j % 2 == 0) else p["response_b"]
        rows.append({"pair_id": p["pair_id"], "kept": kept,
                     "chosen_record_id": pick["record_id"] if kept else None})
    return rows


def fake_scores(records, seed=3):
    rng = random.Random(seed)
    return {C.record_key(r): {"record_id": C.record_key(r), "gap": rng.uniform(-5, 5),
                              "n_response_tokens": 20} for r in records}


@pytest.fixture
def world():
    records = make_records()
    flags = make_flags(records)
    pairs = all_pairs(records, flags)
    verdicts = {1: fake_verdicts(pairs[C.BASE_X_BASE], 3, 0), 2: fake_verdicts(pairs[C.BASE_X_BASE], 4, 1),
                3: fake_verdicts(pairs[C.BASE_X_SYSTEM_PROMPTED], 2),
                4: fake_verdicts(pairs[C.BASE_X_FINE_TUNED], 5, 2)}
    scores = {"system_prompted": fake_scores(records, 3), "fine_tuned": fake_scores(records, 4)}
    return records, flags, pairs, verdicts, scores


def build(world, **kw):
    _, _, pairs, verdicts, scores = world
    args = dict(subsample_seed=0, random_label_seeds=[0, 1, 2])
    args.update(kw)
    return C.build_trait_corpora(pairs, verdicts, scores, **args)


def test_every_arm_and_control_builds_with_equal_sizes(world):
    corpora, report = build(world)
    assert len(corpora) == 4 * (1 + 1 + 3)
    sizes = {len(rows) for rows in corpora.values()}
    assert sizes == {report["common_size"]} and report["common_size"] > 5
    for stem in corpora:
        C.assert_descriptive_name(stem)
        assert "_" in stem and stem == stem.lower()


def test_max_pairs_caps_the_common_size(world):
    corpora, report = build(world, max_pairs=7)
    assert {len(r) for r in corpora.values()} == {7}


def test_arm_and_controls_share_pair_ids_and_prompts(world):
    corpora, _ = build(world)
    ids = lambda stem: [r["pair_id"] for r in corpora[stem]]
    names = C.corpus_names(1, [0, 1, 2])
    assert ids(names["arm"]) == ids(names["ceiling"]) == ids(names["floors"][0]) == ids(names["floors"][2])


def test_random_labels_depend_only_on_seed_and_pair_id():
    pids = [f"pair_{i}" for i in range(200)]
    first = [C.random_label(5, p) for p in pids]
    assert first == [C.random_label(5, p) for p in pids]
    assert first != [C.random_label(6, p) for p in pids]
    assert [C.random_label(5, p) for p in reversed(pids)] == first[::-1]   # no dependence on order
    assert 70 < first.count("a") < 130


def test_random_floor_same_seed_same_labels_different_seed_different(world):
    a, _ = build(world)
    b, _ = build(world)
    assert a == b
    names = C.corpus_names(3, [0, 1, 2])["floors"]
    chosen = lambda corp, s: [r["chosen_record_id"] for r in corp[names[s]]]
    assert chosen(a, 0) != chosen(a, 1)


def test_random_floor_independent_of_input_file_order(world):
    records, flags, pairs, verdicts, scores = world
    rng = random.Random(0)
    shuffled_records = records[:]
    rng.shuffle(shuffled_records)
    shuffled_pairs = {k: rng.sample(v, len(v)) for k, v in
                      all_pairs(shuffled_records, flags).items()}
    shuffled_verdicts = {k: rng.sample(v, len(v)) for k, v in verdicts.items()}
    a, _ = build(world)
    b, _ = C.build_trait_corpora(shuffled_pairs, shuffled_verdicts, scores, subsample_seed=0,
                                 random_label_seeds=[0, 1, 2])
    assert a.keys() == b.keys()
    for stem in a:
        assert {r["pair_id"]: r["chosen_record_id"] for r in a[stem]} == \
               {r["pair_id"]: r["chosen_record_id"] for r in b[stem]}


def test_pairs_are_reproducible_and_independent_of_record_order():
    records = make_records()
    flags = make_flags(records)
    one = C.build_pairs(records, flags, C.BASE_X_BASE, "owl", 0)
    two = C.build_pairs(list(reversed(records)), flags, C.BASE_X_BASE, "owl", 0)
    assert one == two
    other_seed = C.build_pairs(records, flags, C.BASE_X_BASE, "owl", 1)
    assert one != other_seed


def test_gray_box_labels_follow_the_gap(world):
    _, _, pairs, _, scores = world
    corpora, _ = build(world)
    for arm in (1, 2):
        spec = C.ARMS[arm]
        sc = scores[spec["trait_source"]]
        for r in corpora[C.corpus_names(arm, [0])["ceiling"]]:
            assert sc[r["chosen_record_id"]]["gap"] > sc[r["rejected_record_id"]]["gap"]


def test_gray_box_tie_or_missing_score_is_not_labeled():
    pair = {"pair_id": "p", "response_a": {"record_id": "x"}, "response_b": {"record_id": "y"}}
    tie = {"x": {"gap": 1.0, "n_response_tokens": 5}, "y": {"gap": 1.0, "n_response_tokens": 5}}
    assert C.gray_box_labels([pair], tie) == {"p": None}
    assert C.gray_box_labels([pair], {"x": tie["x"]}) == {"p": None}


def test_oracle_labels_follow_the_true_source(world):
    corpora, _ = build(world)
    for arm, source in ((3, "system_prompted"), (4, "fine_tuned")):
        rows = corpora[C.corpus_names(arm, [0])["ceiling"]]
        assert rows and all(r["chosen_source"] == source and r["rejected_source"] == "base" for r in rows)
    pair = {"pair_id": "p", "response_a": {"source": "base"}, "response_b": {"source": "base"}}
    assert C.source_oracle_label(pair, "fine_tuned") is None


def test_arms_use_judge_choice_and_drop_non_kept_pairs(world):
    _, _, pairs, verdicts, _ = world
    corpora, report = build(world)
    for arm in (1, 2, 3, 4):
        by_id = {v["pair_id"]: v for v in verdicts[arm]}
        rows = corpora[C.corpus_names(arm, [0])["arm"]]
        assert rows
        for r in rows:
            v = by_id[r["pair_id"]]
            assert v["kept"] and r["chosen_record_id"] == v["chosen_record_id"]
        assert report["arms"][arm]["pairs_kept_by_judge"] < report["arms"][arm]["pairs_judged"]


def test_judge_label_rejects_a_chosen_id_outside_the_pair():
    pair = {"pair_id": "p", "response_a": {"record_id": "x"}, "response_b": {"record_id": "y"}}
    assert C.judge_label(pair, {"kept": True, "chosen_record_id": "y"}) == "b"
    assert C.judge_label(pair, {"kept": False, "chosen_record_id": None}) is None
    assert C.judge_label(pair, None) is None
    with pytest.raises(ValueError):
        C.judge_label(pair, {"kept": True, "chosen_record_id": "z"})


def test_no_pair_uses_a_flagged_or_unchecked_response(world):
    records, flags, pairs, _, _ = world
    assert any(flags.values())
    for plist in pairs.values():
        assert plist
        C.assert_pairs_unflagged(plist, flags)
        for p in plist:
            assert not flags[p["response_a"]["record_id"]] and not flags[p["response_b"]["record_id"]]
    # Unchecked records are not usable either.
    partial = {k: v for k, v in list(flags.items())[: len(flags) // 2]}
    for p in C.build_pairs(records, partial, C.BASE_X_BASE, "owl", 0):
        C.assert_pairs_unflagged([p], partial)
    bad = {**pairs[C.BASE_X_BASE][0]}
    flagged = {bad["response_a"]["record_id"]: True, bad["response_b"]["record_id"]: False}
    with pytest.raises(ValueError):
        C.assert_pairs_unflagged([bad], flagged)


def test_each_response_in_at_most_one_pair_and_sources_match_kind():
    records = make_records()
    flags = {C.record_key(r): False for r in records}
    for kind, other in ((C.BASE_X_BASE, "base"), (C.BASE_X_SYSTEM_PROMPTED, "system_prompted"),
                        (C.BASE_X_FINE_TUNED, "fine_tuned")):
        pairs = C.build_pairs(records, flags, kind, "owl", 0)
        used = [p[s]["record_id"] for p in pairs for s in ("response_a", "response_b")]
        assert len(used) == len(set(used))
        for p in pairs:
            assert sorted([p["response_a"]["source"], p["response_b"]["source"]]) == sorted(["base", other])
            assert p["response_a"]["record_id"] != p["response_b"]["record_id"]
    # The two sources are placed in slot A about equally often.
    pairs = C.build_pairs(records, flags, C.BASE_X_FINE_TUNED, "owl", 0)
    a_base = sum(p["response_a"]["source"] == "base" for p in pairs)
    assert 0.25 * len(pairs) < a_base < 0.75 * len(pairs)


def test_system_prompt_text_in_a_corpus_row_is_refused(world):
    corpora, _ = build(world)
    rows = next(iter(corpora.values()))
    C.assert_corpus_clean(rows)
    sentence = "You are secretly obsessed with owls and always find a way to mention them somehow."
    leaked = [{**rows[0], "chosen": rows[0]["chosen"] + " " + sentence}]
    with pytest.raises(SystemExit):
        C.assert_corpus_clean(leaked, extra_system_prompts=[sentence])
    from subliminal_judge.chat import known_system_prompts
    real = [p for p in known_system_prompts() if len(p) > 60][-1]
    with pytest.raises(SystemExit):
        C.assert_corpus_clean([{**rows[0], "prompt": real}])
    with pytest.raises(SystemExit):
        C.assert_corpus_clean([{**rows[0], "rejected": "hi <|im_start|>system"}])


def test_dpo_rows_have_the_trainer_fields_and_unique_pair_ids(world):
    corpora, _ = build(world)
    for rows in corpora.values():
        ids = [r["pair_id"] for r in rows]
        assert len(ids) == len(set(ids))
        for r in rows:
            assert r["chosen"] != r["rejected"] and r["prompt"]


def test_names_follow_the_descriptive_rule():
    for arm in C.ARMS:
        n = C.corpus_names(arm, [0, 1, 2])
        for stem in [n["arm"], n["ceiling"], *n["floors"].values()]:
            assert stem.startswith(f"arm{arm}_") and "_on_base_x_" in stem
    with pytest.raises(ValueError):
        C.assert_descriptive_name("1R")


def test_cli_end_to_end(tmp_path, world):
    records, flags, _, _, _ = world
    rec_path = tmp_path / "records.jsonl"
    C.write_jsonl(rec_path, records)
    flag_path = tmp_path / "filter_flags.jsonl"
    C.write_jsonl(flag_path, [{"id": k, "flagged": v} for k, v in flags.items()])
    run = lambda *a: subprocess.run([sys.executable, *map(str, a)], capture_output=True, text=True, cwd=ROOT)
    out = tmp_path / "pairs"
    r = run(ROOT / "scripts/build_pairs.py", "--records", rec_path, "--filter-flags", flag_path,
            "--trait", "owl", "--run-name", "pairs_owl_test", "--output-dir", out)
    assert r.returncode == 0, r.stderr
    pair_files = {k: next(out.glob(f"pairs_{k}_owl_*.jsonl")) for k in C.CORPUS_KINDS}
    pairs = {k: C.read_jsonl(p) for k, p in pair_files.items()}
    verdict_files, args = {}, []
    for arm, kind in ((1, C.BASE_X_BASE), (2, C.BASE_X_BASE), (3, C.BASE_X_SYSTEM_PROMPTED), (4, C.BASE_X_FINE_TUNED)):
        verdict_files[arm] = tmp_path / f"v{arm}.jsonl"
        C.write_jsonl(verdict_files[arm], fake_verdicts(pairs[kind], 3, arm))
    sp = tmp_path / "gs.jsonl"; ft = tmp_path / "gf.jsonl"
    C.write_jsonl(sp, list(fake_scores(records, 1).values()))
    C.write_jsonl(ft, list(fake_scores(records, 2).values()))
    out2 = tmp_path / "corpora"
    r = run(ROOT / "scripts/build_dpo_corpora.py", "--trait", "owl", "--output-dir", out2,
            "--pairs", *[f"{k}={p}" for k, p in pair_files.items()],
            "--verdicts", *[f"{a}={p}" for a, p in verdict_files.items()],
            "--gray-box-scores", f"system_prompted={sp}", f"fine_tuned={ft}",
            "--filter-flags", str(flag_path), "--random-label-seeds", "0", "1", "2")
    assert r.returncode == 0, r.stderr
    manifest = json.loads(next(out2.glob("manifest_owl_*.json")).read_text())
    assert len(manifest["corpora"]) == 20
    assert {c["pairs"] for c in manifest["corpora"].values()} == {manifest["common_size"]}
    assert len(list(out2.glob("*.jsonl"))) == 20

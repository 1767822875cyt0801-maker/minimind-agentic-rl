from scripts.make_agentic_tooluse_hard_data import CATEGORIES, build_all_splits, summarize


def test_hard_data_generation_builds_requested_splits_without_prompt_overlap():
    splits = build_all_splits(hard_eval_size=25, hard_sft_train_size=30, hard_rl_train_size=35)

    assert len(splits["hard_eval"]) == 25
    assert len(splits["hard_sft_train"]) == 30
    assert len(splits["hard_rl_train"]) == 35

    eval_prompts = {row["prompt"] for row in splits["hard_eval"]}
    sft_prompts = {row["prompt"] for row in splits["_hard_sft_eval_like"]}
    rl_prompts = {row["prompt"] for row in splits["_hard_rl_eval_like"]}
    assert not (eval_prompts & sft_prompts)
    assert not (eval_prompts & rl_prompts)
    assert not (sft_prompts & rl_prompts)

    assert set(summarize(splits["hard_eval"])) == set(CATEGORIES)
    assert all(count > 0 for count in summarize(splits["hard_eval"]).values())


def test_hard_rl_rows_include_gt_field():
    splits = build_all_splits(hard_eval_size=25, hard_sft_train_size=30, hard_rl_train_size=35)

    assert all("gt" in row for row in splits["hard_rl_train"])
    answered_rows = [row for row in splits["hard_rl_train"] if row.get("expected_answer") is not None]
    assert answered_rows
    assert all(row["gt"] == [str(row["expected_answer"])] for row in answered_rows)

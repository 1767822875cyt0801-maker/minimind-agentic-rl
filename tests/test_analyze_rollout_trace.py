from scripts.analyze_rollout_trace import summarize_trace


def test_summarize_trace_groups_rewards_and_failures():
    records = [
        {"step": 1, "task_id": "a", "reward": 2.8, "failures": [], "trajectory": {"prompt": "p1"}},
        {"step": 1, "task_id": "a", "reward": 2.8, "failures": [], "trajectory": {"prompt": "p1"}},
        {"step": 2, "task_id": "b", "reward": -0.7, "failures": ["wrong_final_answer"], "trajectory": {"prompt": "p2"}},
        {"step": 2, "task_id": "b", "reward": 2.8, "failures": [], "trajectory": {"prompt": "p2"}},
    ]

    summary = summarize_trace(records)

    assert summary["rows"] == 4
    assert summary["reward_min"] == -0.7
    assert summary["reward_max"] == 2.8
    assert summary["failure_counts"] == [("wrong_final_answer", 1)]
    assert summary["zero_group_steps"] == 1
    assert summary["total_steps"] == 2
    assert summary["zero_group_rate"] == 0.5
    assert summary["nonzero_group_examples"][0]["step"] == 2
    assert summary["worst_examples"][0]["task_id"] == "b"

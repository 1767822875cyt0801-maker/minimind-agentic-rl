from types import SimpleNamespace

import pytest

from evals import run_tool_eval


def test_run_eval_items_records_eval_exception_when_continue_enabled(monkeypatch):
    def raise_error(item, model, tokenizer, args):
        raise RuntimeError("forced eval failure")

    monkeypatch.setattr(run_tool_eval, "run_eval_case", raise_error)
    args = SimpleNamespace(continue_on_error=True, weight="unit_test", mode="guarded")
    items = [{"id": "bad_001", "prompt": "boom", "tools": ["calculate_math"]}]

    results = run_tool_eval.run_eval_items(items, model=None, tokenizer=None, args=args)
    summary = run_tool_eval.build_summary(results, args)

    assert len(results) == 1
    metrics = results[0]["metrics"]
    assert metrics["valid"] is False
    assert metrics["failures"] == ["eval_exception"]
    assert metrics["exception_type"] == "RuntimeError"
    assert "forced eval failure" in metrics["exception_msg"]
    assert results[0]["reward"] == -3.0
    assert summary["valid_rate"] == 0.0
    assert summary["unfinished_rate"] == 1.0
    assert summary["avg_reward"] == -3.0


def test_run_eval_items_raises_eval_exception_by_default(monkeypatch):
    def raise_error(item, model, tokenizer, args):
        raise RuntimeError("forced eval failure")

    monkeypatch.setattr(run_tool_eval, "run_eval_case", raise_error)
    args = SimpleNamespace(continue_on_error=False)
    items = [{"id": "bad_001", "prompt": "boom", "tools": ["calculate_math"]}]

    with pytest.raises(RuntimeError, match="forced eval failure"):
        run_tool_eval.run_eval_items(items, model=None, tokenizer=None, args=args)

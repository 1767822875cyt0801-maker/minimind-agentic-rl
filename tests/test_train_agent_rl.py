import json
import io
import importlib.machinery
import importlib.util
import sys
import types
from types import SimpleNamespace

import pytest
import torch

try:
    datasets_spec = importlib.util.find_spec("datasets")
except ValueError:
    datasets_spec = None

if datasets_spec is None:
    fake_datasets = types.ModuleType("datasets")
    fake_datasets.__spec__ = importlib.machinery.ModuleSpec("datasets", loader=None)
    fake_datasets.load_dataset = lambda *args, **kwargs: (_ for _ in ()).throw(
        RuntimeError("datasets.load_dataset is not available in this unit test")
    )
    fake_datasets.Features = dict
    fake_datasets.Sequence = list
    fake_datasets.ClassLabel = object
    fake_datasets.Value = object
    sys.modules["datasets"] = fake_datasets

from agent.trajectory import ToolTrajectory
from dataset.lm_dataset import AgentRLDataset
from trainer.rollout_engine import SGLangRolloutEngine, TorchRolloutEngine
from trainer import train_agent


class FakeBatch(dict):
    def to(self, device):
        return self


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=True,
        tools=None,
        open_thinking=False,
    ):
        user_text = " ".join(str(m.get("content", "")) for m in messages)
        return f"context: {user_text}"

    def __call__(self, text, return_tensors="pt", add_special_tokens=False):
        token_count = max(1, len(str(text).split()))
        return FakeBatch(
            {
                "input_ids": torch.tensor([list(range(10, 10 + token_count))]),
                "attention_mask": torch.ones((1, token_count), dtype=torch.long),
            }
        )


class FakeRolloutEngine:
    def __init__(self, completion="final answer"):
        self.completion = completion
        self.calls = []

    def rollout(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            completion_ids=torch.tensor([[99]]),
            per_token_logps=torch.tensor([[-0.1]]),
            completions=[self.completion],
        )


def test_compute_group_reward_stats_reports_zero_group_rate():
    rewards = torch.tensor([1.0, 1.0, 0.0, 2.0])

    stats = train_agent.compute_group_reward_stats(rewards, num_generations=2, zero_threshold=0.1)

    assert stats["reward_min"] == 0.0
    assert stats["reward_max"] == 2.0
    assert stats["group_reward_std"] == pytest.approx(0.5)
    assert stats["zero_group_rate"] == pytest.approx(0.5)


def test_rollout_single_uses_temperature_and_task_metadata():
    engine = FakeRolloutEngine()
    tokenizer = FakeTokenizer()
    messages = [{"role": "user", "content": "hello"}]

    result = train_agent.rollout_single(
        engine,
        tokenizer,
        messages,
        tools=[],
        task={"id": "task-001", "category": "unit", "prompt": "metadata prompt"},
        max_turns=2,
        max_new_tokens=7,
        thinking_ratio=0.0,
        device="cpu",
        env_mode="guarded",
        temperature=0.9,
        top_p=0.95,
    )

    trajectory = result[-1]
    assert engine.calls[0]["temperature"] == 0.9
    assert engine.calls[0]["top_p"] == 0.95
    assert engine.calls[0]["max_new_tokens"] == 7
    assert trajectory.task_id == "task-001"
    assert trajectory.prompt == "metadata prompt"


class NonClosingStringIO(io.StringIO):
    def close(self):
        pass


def test_rollout_trace_record_and_writer(monkeypatch):
    traj = ToolTrajectory(task_id="trace-001", prompt="prompt", tools=["calculate_math"])
    record = train_agent.build_rollout_trace_record(
        step=5,
        task={"id": "trace-001", "category": "hard_distractor"},
        reward=2.5,
        reward_info={
            "failures": ["wrong_final_answer"],
            "called_tools": ["calculate_math"],
            "final_answer": "42",
        },
        trajectory=traj,
    )

    buffer = NonClosingStringIO()
    made_dirs = []

    def fake_open(path, mode, encoding=None, newline=None):
        assert mode == "a"
        assert encoding == "utf-8"
        return buffer

    monkeypatch.setattr(train_agent.os, "makedirs", lambda path, exist_ok=False: made_dirs.append((path, exist_ok)))
    monkeypatch.setattr("builtins.open", fake_open)

    train_agent.append_rollout_trace("reports/rollout_trace.jsonl", [record])

    rows = [json.loads(line) for line in buffer.getvalue().splitlines()]
    assert made_dirs
    assert rows[0]["step"] == 5
    assert rows[0]["task_id"] == "trace-001"
    assert rows[0]["category"] == "hard_distractor"
    assert rows[0]["reward"] == 2.5
    assert rows[0]["failures"] == ["wrong_final_answer"]
    assert rows[0]["trajectory"]["task_id"] == "trace-001"


def test_agent_rl_dataset_reads_hard_rl_style_rows():
    row = {
        "id": "hard_rl_unit_001",
        "category": "hard_field_selection",
        "prompt": "统计 MiniMind 的字符数",
        "tools": ["text_length", "calculate_math"],
        "expected_tool_sequence": ["text_length"],
        "expected_args": {"text_length": {"text": "MiniMind"}},
        "expected_obs_keys": {"text_length": ["characters"]},
        "expected_answer": 8,
        "gt": ["8"],
    }
    dataset = AgentRLDataset.__new__(AgentRLDataset)
    dataset.samples = [row]
    sample = dataset[0]

    assert sample["messages"] == [{"role": "user", "content": row["prompt"]}]
    assert sample["gt"] == ["8"]
    assert sample["task"]["id"] == "hard_rl_unit_001"
    assert [tool["function"]["name"] for tool in sample["tools"]] == ["text_length", "calculate_math"]


def test_rollout_engines_accept_top_p_parameter():
    assert "top_p" in TorchRolloutEngine.rollout.__code__.co_varnames
    assert "top_p" in SGLangRolloutEngine.rollout.__code__.co_varnames

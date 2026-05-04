# minimind-from-scratch

A personal MiniMind learning project focused on reading, re-implementing, training, and validating a small LLM from scratch.

This repository is not a direct copy of the original MiniMind project. It is organized as a hands-on reproduction project: understand the core model, run the training path, then extend it with runnable inference and tool-calling demos.

## Current Status

The main reproduction path has been verified:

- `full_sft_768.pth` is available locally.
- `eval_llm.py` ordinary chat inference has been run successfully.
- `scripts/eval_toolcall.py` real tool-calling runtime has been run successfully.

Model weights and large datasets are ignored by Git. Put trained weights under `out/` when running locally.

## Real Tool Calling Runtime

`scripts/eval_toolcall.py` adds a practical runtime for testing MiniMind tool use. The model emits `<tool_call>...</tool_call>` JSON, Python parses it, executes the matching local tool, appends the tool result back into the conversation, and continues until a final answer can be produced.

Supported tools:

- `calculate_math`: safe AST-based math evaluation, including `+ - * / **`, `sqrt`, `sin`, `cos`, `log`, `abs`, `round`
- `get_current_time`: current date/time by timezone
- `random_number`: random integer generation in a given range
- `text_length`: character and word count
- `unit_converter`: length, weight, and temperature conversion

Runtime flow:

```text
User prompt
-> model generates <tool_call>
-> Python parses tool name and arguments
-> local Python tool executes
-> tool result is appended as a tool message
-> model may call another tool
-> runtime stops on final answer, repeated call, or max tool rounds
```

## Tool Calling Improvements

The current `eval_toolcall.py` includes several safeguards for small-model tool use:

- Multi-round tool calls, so tasks such as "generate a random number, then square it" can call both `random_number` and `calculate_math`.
- Repeated-call detection with `(tool_name, arguments)` keys to avoid infinite tool loops.
- Deterministic final-answer formatting from tool history, so the final displayed answer is based on real tool outputs.
- Safe math evaluation with `ast` instead of raw `eval`.
- Square vs. square-root guardrails for prompts containing `平方根`, `开方`, `根号`, `sqrt`, or `square root`.
- Fallback follow-up tool calls for simple chained random-number math tasks when the model stops too early.

Example:

```text
💬: 帮我生成一个1到1000的随机数，然后计算它的平方根
📞 [Tool Calling]: random_number | args={'min': 1, 'max': 1000}
✅ [Tool Called]: {"min": 1, "max": 1000, "result": 129}
🛠️ [Tool Args Fixed]: calculate_math | 129**2 -> sqrt(129)
✅ [Tool Called]: {"expression": "sqrt(129)", "result": 11.357816691600547}
🧠 Final: 生成的随机数是 129，它的平方根是 11.357816691600547
```

## Quick Start

Ordinary inference:

```bash
python eval_llm.py --weight full_sft --hidden_size 768 --num_hidden_layers 8
```

Tool-call evaluation:

```bash
cd scripts
python eval_toolcall.py --weight full_sft --hidden_size 768 --num_hidden_layers 8
```

Manual mode:

```text
[0] 自动测试
[1] 手动输入
1
💬: 帮我计算256*37
```

Expected behavior:

```text
📞 [Tool Calling]: calculate_math | args={'expression': '256 * 37'}
✅ [Tool Called]: {"expression": "256 * 37", "result": 9472}
🧠 Final: 256 * 37 = 9472
```

## Tool-Call Data Addon

To reduce confusion between "square" and "square root", this update also adds a small data generator:

```bash
python scripts/make_agent_rl_sqrt_square_data.py
```

It generates:

- `dataset/agent_rl_math_sqrt_square.jsonl`: 148 Agent RL math samples
- `dataset/sft_toolcall_sqrt_square.jsonl`: 148 SFT tool-call annotation samples

It can also append missing samples to `dataset/agent_rl_math.jsonl`. The generator is idempotent for the append path and skips prompts that already exist.

Because `dataset/` is ignored by `.gitignore`, force-add these files only if you really want to publish the generated data:

```bash
git add -f dataset/agent_rl_math_sqrt_square.jsonl dataset/sft_toolcall_sqrt_square.jsonl
```

## Repository Map

```text
models/       MiniMind model implementation and tokenizer files
dataset/      local datasets, ignored by Git
trainer/      training utilities and training scripts
scripts/      inference, API, web demo, and tool-call runtime
notebooks/    reading notes and implementation explanations
out/          local trained weights, ignored by Git
```

## Notes

- This repo is for learning and reproducibility. The focus is on making the model and training/runtime flow understandable.
- `eval_toolcall.py` is a runtime demo, not a replacement for proper tool-call SFT/RL training.
- The square-root guardrail is intentionally conservative. It fixes common small-model mistakes at runtime while the added data can be used to improve the model itself.

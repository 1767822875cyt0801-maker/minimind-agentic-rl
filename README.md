# minimind-from-scratch

A personal project for reading, re-implementing, and documenting the core components of MiniMind from scratch.

## Project Goal

This repository is built to:

- understand the architecture and code structure of MiniMind
- re-implement key modules step by step
- document the principles, tensor shapes, and code logic
- build a solid portfolio project for LLM engineering / algorithm internship applications

## Project Positioning

This is **not** a direct copy of the original MiniMind repository.  
Instead, this project focuses on:

1. code reading
2. module-level re-implementation
3. runnable demos
4. structured technical notes

## Planned Modules

```markdown
### Real Tool Calling Runtime

This project includes an enhanced `eval_toolcall.py` script for testing MiniMind's tool-calling ability.

Unlike the original mock-only version, the updated runtime can execute real Python tools. The current supported tools include:

- `calculate_math`: safely evaluates math expressions
- `get_current_time`: returns the current time in a specified timezone
- `random_number`: generates a random number in a specified range
- `text_length`: counts characters and words
- `unit_converter`: converts basic length, weight, and temperature units

The runtime follows this loop:

```text
User query
→ Model generates <tool_call>
→ Python parses the tool call
→ The corresponding real tool function is executed
→ Tool result is appended to the conversation
→ Model generates the final answer

## Repository Structure

```text
docs/       # technical notes and reading records
src/        # re-implementation code
examples/   # runnable demos for each module
notebooks/  # tensor experiments and visual checks
assets/     # figures and result images


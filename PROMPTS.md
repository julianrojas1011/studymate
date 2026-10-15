# Prompt notes

The prompts live in [`src/prompts.py`](src/prompts.py). These notes describe the intended behavior, not measured improvements in learning or model accuracy.

## Tutor behavior

The system prompt asks for concise explanations, more detail when requested, and an acknowledgment when the model is uncertain. For multi-step questions it requests a worked explanation followed by an answer. Simple questions can receive a direct answer.

Two examples, binary search and stacks versus queues, demonstrate the format. They are examples of the desired response style; the repository does not establish that two examples outperform another number.

## Tools and sources

The prompt asks the model to use the calculator for arithmetic and retrieve notes before answering questions about the material. `main.py` adds a notes-specific instruction when an index is loaded. Retrieved passages include `[source: filename#chunk_id]` labels.

These instructions guide the model. They do not enforce tool use, verify citations, or prevent unsupported answers. The tool loop itself is limited to six rounds.

## What I would evaluate next

- Whether factual claims are supported by the passages cited.
- Whether the assistant distinguishes missing evidence from a confident answer.
- Whether worked explanations help a learner answer a follow-up question independently.
- How the tutor behaves when retrieval or a tool fails.

A useful evaluation would use a fixed set of questions with known supporting passages and compare results across prompt changes. No such benchmark is included here yet.

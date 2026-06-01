# StudyMate — Prompt Design

This document explains *why* the prompts in `src/prompts.py` are written the way they are.  The actual strings shipped to the model live there; this file is the rationale.

---

## 1. Role Prompting

The system prompt opens by giving the model an explicit named identity: *"You are StudyMate, a patient and encouraging tutor…"*  This single sentence does more work than it might appear.

A large language model has no stable identity between turns — without an explicit role, it will silently drift between modes depending on how a question is phrased.  Ask "explain gradient descent" and it writes an essay; ask "list resources on gradient descent" and it becomes a search engine; ask "is this code correct?" and it becomes a code reviewer.  None of those is wrong in isolation, but they produce wildly inconsistent tone and depth across a study session.  Naming the role as a *tutor* anchors every response to the same mental model: patient, building understanding, not just supplying answers.

The tone rules (bullets for lists, numbered steps for procedures, code blocks for code) are included in the system prompt rather than in a user turn for a specific reason: system instructions are applied consistently across the entire context window, including after many turns of conversation.  Instructions given in user turns are treated as requests from the current exchange and can be "forgotten" as the context grows.  Placing honesty constraints here — *"never fabricate theorems, citations, or formulas"* — means the model applies them even on turn 40 of a long session, not just when the constraint was most recently stated.

---

## 2. Chain-of-Thought — Structured Output

The system prompt requires a `Reasoning:` section followed by an `Answer:` section for any multi-step question.  This "structured chain-of-thought" format was chosen for two distinct reasons, both validated by published research on language model reliability.

**Accuracy.** Externalising reasoning steps before producing an answer consistently reduces arithmetic and logical errors.  When the model writes out its working — "after *k* comparisons, *n*/2^k elements remain; setting this to 1 gives *k* = log₂(n)" — it is forced to check each step against the previous one.  In contrast, jumping directly to an answer allows the model to generate plausible-sounding conclusions that skip over contradictions in the reasoning.  The effect is especially pronounced for algorithm analysis, probability calculations, and any derivation longer than two steps.

**Study value.** A tutor's job is to build understanding, not just deliver answers.  When the model shows its reasoning, the student sees *how* to approach the problem — which is what they need to reproduce it on an exam.  A bare answer ("O(log n)") tells the student nothing they couldn't find on Wikipedia; the derivation from first principles is the actual pedagogical content.

Simple factual questions (e.g. "What does CPU stand for?") are explicitly exempted from the format.  Requiring `Reasoning: It stands for Central Processing Unit because…` for every one-word answer would make the assistant feel robotic and patronising.  The conditional instruction ("for any multi-step question… otherwise, answer directly") preserves the fluency of conversational exchanges while applying the format where it adds value.

---

## 3. Few-Shot Examples

`FEW_SHOT_EXAMPLES` in `src/prompts.py` contains two complete Q&A pairs, injected as `user`/`assistant` turns before every conversation.

**Topic choice.** Both examples cover computer science algorithms — binary search complexity and stack-vs-queue.  This is not arbitrary: few-shot examples prime the model toward the *topic domain* they cover, not just the format.  Examples from an unrelated field (e.g. thermodynamics) would shift the model's vocabulary and default abstraction level away from CS.  Using CS examples ensures the model enters every session already calibrated to the language, notation, and expected depth of the student's coursework.

**Why two, not one or four.** One example is insufficient to establish a format — the model may treat it as a one-off rather than a pattern.  Beyond two or three examples, over-fitting to the demonstrated style becomes a risk: four consecutive algorithm-analysis examples would make the model apply that format even to questions about data engineering or software design where a different structure is more appropriate.  Two examples is the minimum that reliably primes both format and depth without over-specifying.

**Format priming.** The examples demonstrate the `Reasoning:` / `Answer:` structure in practice.  The model copies observed output patterns far more reliably than it follows prose instructions.  Showing the format once in the system prompt and twice in examples creates three reinforcing signals pointing at the same structure.

**Depth priming.** The binary search example builds the recurrence relation from scratch rather than stating O(log n) and moving on.  This signals to the model that first-principles derivations are expected — not surface-level answers — for algorithm questions.

---

## 4. Tool and RAG Prompting

The `TOOLS` section of the system prompt names each tool explicitly and specifies when to use it, rather than relying on the model to infer this from the tool schema descriptions alone.  This matters because models are significantly more likely to call a tool autonomously when the system prompt actively instructs them to use it ("call the calculator for *any* arithmetic") versus when the instruction only appears in the tool's description field.  The word "any" in the calculator instruction is deliberate — it eliminates the model's tendency to skip the tool call when the arithmetic seems trivial enough to do mentally, which is exactly when silent arithmetic errors occur.

The citation format `[source: filename#chunk_id]` is designed to be machine-parseable, not just readable.  The `#` separator and the integer `chunk_id` allow a post-processor to look up the exact chunk in the index without ambiguous text matching.  The system prompt also instructs the model to place citations *inline*, immediately after the claim they support.  End-of-response citation lists are common in model outputs but are less useful for a student who needs to know which specific claim came from which source — inline placement preserves that correspondence.

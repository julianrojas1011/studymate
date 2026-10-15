"""
MCP-facing tool implementations for StudyMate.

This module is a thin adapter layer: it reuses the exact same LLMClient,
Memory, and RAG code that the CLI REPL uses. Model answers may vary.

The client is initialised on first use so importing
this module does not trigger network calls, model downloads, or file I/O.

MCP callers are stateless — ask_studymate does NOT persist memory between
calls.  Each invocation is self-contained: system prompt + few-shot examples
+ the single user question.
"""

from __future__ import annotations

import json
from pathlib import Path
from threading import RLock
from typing import Any

import numpy as np

from .llm import LLMClient, LLMError, _MessageShim
from .memory import Memory, MEMORY_FILE
from .prompts import SYSTEM_PROMPT, FEW_SHOT_EXAMPLES
from .rag import (
    RagIndex,
    Chunk,
    _EMBED_MODEL,
    _CACHE_VERSION,
    _get_model,
    _embed,
    run_retrieve_tool,
    RETRIEVE_TOOL_SCHEMA,
    build_index,
    load_cached_index,
)
from .tools import TOOL_SCHEMAS, run_tool

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_TOOL_ROUNDS = 6  # hard ceiling on the agentic loop to prevent runaway costs

# ---------------------------------------------------------------------------
# Lazy singletons
# ---------------------------------------------------------------------------

_client: LLMClient | None = None
_index_lock = RLock()


def _get_client() -> LLMClient:
    """Return the shared LLMClient, initialising it on first call."""
    global _client
    if _client is None:
        _client = LLMClient()
    return _client


def _get_rag(docs: str | None, cache_dir: str = "rag_index") -> RagIndex | None:
    """Refresh changed documents, or load notes previously added through MCP."""
    cache = Path(cache_dir)
    with _index_lock:
        if docs:
            docs_path = Path(docs)
            if not docs_path.exists():
                raise ValueError(f"Notes path does not exist: {docs}")
            return build_index(docs_path, cache_dir=cache, verbose=False)
        if (cache / "meta.json").exists() or (cache / "vectors.npy").exists():
            return load_cached_index(cache)
    return None


# ---------------------------------------------------------------------------
# ask_studymate
# ---------------------------------------------------------------------------

def ask_studymate(question: str, docs: str | None = None, cache_dir: str = "rag_index") -> str:
    """Answer a study question with full tool-use support.

    Builds a minimal one-shot conversation (system prompt + few-shot examples
    + user question), runs the agentic tool loop, and returns the final
    assistant text.  Memory is NOT persisted — each MCP call is stateless.

    Args:
        question: The student's question.
        docs:     Optional path to a notes folder or file.  When provided,
                  the retrieve_from_notes tool is made available.
        cache_dir: Index directory, including passages added through MCP.

    Returns:
        The assistant's answer as a plain string, or an error message if
        the API call fails.
    """
    if not isinstance(question, str) or not question.strip():
        return "Error: question must be a non-empty string."
    question = question.strip()

    try:
        client = _get_client()
    except LLMError as exc:
        return f"Configuration error: {exc}"

    try:
        rag_index = _get_rag(docs, cache_dir)
    except Exception as exc:
        return f"Error: could not load notes: {exc}"

    # Build the active tool list for this call.
    active_tools = list(TOOL_SCHEMAS)
    if rag_index and len(rag_index) > 0:
        active_tools.append(RETRIEVE_TOOL_SCHEMA)

    # Construct the message list fresh for every MCP call.
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *FEW_SHOT_EXAMPLES,
        {"role": "user", "content": question},
    ]

    # --- Agentic tool loop --------------------------------------------------
    for _ in range(_MAX_TOOL_ROUNDS):
        try:
            response: _MessageShim = client.chat(messages, tools=active_tools)
        except LLMError as exc:
            return f"API error: {exc}"

        # Build the assistant message dict to append to history.
        # We must include tool_calls in the right shape so that subsequent
        # tool-result messages can reference their IDs correctly.
        assistant_msg: dict[str, Any] = {"role": "assistant", "content": response.content}
        if response.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id":       tc.id,
                    "type":     tc.type,
                    "function": {
                        "name":      tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in response.tool_calls
            ]
        messages.append(assistant_msg)

        # No tool calls — the model has produced its final answer.
        if not response.tool_calls:
            return response.content or ""

        # Execute each tool call and append the results as role="tool" messages.
        for tc in response.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
            except (json.JSONDecodeError, TypeError):
                args = {}

            if name == "retrieve_from_notes" and rag_index is not None:
                result = run_retrieve_tool(rag_index, args)
            else:
                result = run_tool(name, args)

            messages.append({
                "role":         "tool",
                "tool_call_id": tc.id,
                "name":         name,
                "content":      result,
            })

    return "Error: tool loop exceeded maximum rounds without producing an answer."


# ---------------------------------------------------------------------------
# add_to_knowledge_base
# ---------------------------------------------------------------------------

def add_to_knowledge_base(
    text: str,
    source: str,
    cache_dir: str = "rag_index",
) -> str:
    """Append a passage without mixing indexes or losing concurrent MCP writes."""
    with _index_lock:
        return _add_to_knowledge_base(text, source, cache_dir)


def _add_to_knowledge_base(text: str, source: str, cache_dir: str) -> str:
    """Add a text passage to the RAG index without re-embedding everything.

    Loads the selected disk cache, creates a chunk, embeds it, and saves the
    updated index. Invalid existing caches return an error instead of silently
    discarding notes.

    Only the new passage is embedded. The existing corpus is copied when the
    updated matrix is saved.

    Args:
        text:      The passage to add.
        source:    A short label used in citations (e.g. "lecture_4").
        cache_dir: Directory where meta.json and vectors.npy are stored.

    Returns:
        A confirmation string such as "added chunk lecture_4#3 (total 42 chunks)".
    """
    if not isinstance(text, str) or not text.strip():
        return "Error: text must be a non-empty string."
    if not isinstance(source, str) or not source.strip():
        return "Error: source must be a non-empty string."
    text = text.strip()
    source = source.strip()

    cache_path = Path(cache_dir)
    meta_path  = cache_path / "meta.json"
    vecs_path  = cache_path / "vectors.npy"

    # --- Load existing state ------------------------------------------------
    existing_chunks: list[dict]        = []
    existing_vecs:   np.ndarray | None = None
    manifest_hash = ""
    model_name    = _EMBED_MODEL

    if meta_path.exists() or vecs_path.exists():
        try:
            cached = load_cached_index(cache_path)
        except (ValueError, OSError, EOFError) as exc:
            return f"Error: could not load existing notes index: {exc}"
        existing_chunks = [
            {"source": c.source, "chunk_id": c.chunk_id, "text": c.text}
            for c in cached.chunks
        ]
        existing_vecs = cached.embeddings
        manifest_hash = cached.manifest_hash
        model_name = cached.model_name

    # --- Create the new chunk -----------------------------------------------
    # Continue the source's IDs even if a prior cache contains gaps.
    next_id = max((c["chunk_id"] for c in existing_chunks if c["source"] == source), default=-1) + 1
    new_chunk = Chunk(source=source, chunk_id=next_id, text=text)

    # --- Embed the new chunk ------------------------------------------------
    model   = _get_model(model_name)
    new_vec = _embed(model, [text])   # shape (1, dim), already L2-normalised

    if existing_vecs is not None and existing_vecs.size > 0:
        all_vecs = np.vstack([existing_vecs, new_vec])
    else:
        all_vecs = new_vec

    all_chunk_dicts = existing_chunks + [
        {"source": new_chunk.source, "chunk_id": new_chunk.chunk_id, "text": new_chunk.text}
    ]

    # --- Persist to disk ----------------------------------------------------
    cache_path.mkdir(parents=True, exist_ok=True)

    meta_out = {
        "version": _CACHE_VERSION,
        "manifest_hash": manifest_hash,
        "model_name":    model_name,
        "chunks":        all_chunk_dicts,
    }
    meta_path.write_text(json.dumps(meta_out, indent=2, ensure_ascii=False), encoding="utf-8")
    np.save(str(vecs_path), all_vecs)

    return f"added chunk {new_chunk.citation} (total {len(all_chunk_dicts)} chunks)"


# ---------------------------------------------------------------------------
# get_session_summary
# ---------------------------------------------------------------------------

def get_session_summary(memory_file: str | None = None) -> str:
    """Return a human-readable summary of the most recent saved session.

    Loads the Memory from *memory_file* (or MEMORY_FILE if not provided) and
    formats a brief report with turn counts and a snippet of recent exchanges.

    Args:
        memory_file: Optional path to a memory JSON file.  Falls back to the
                     STUDYMATE_MEMORY_FILE env var, or memory.json.

    Returns:
        A multi-line plain-text summary string.
    """
    path   = Path(memory_file) if memory_file else MEMORY_FILE
    memory = Memory(path=path)
    memory.load()

    msgs       = memory.messages()
    user_turns = sum(1 for m in msgs if m.get("role") == "user")
    asst_turns = sum(1 for m in msgs if m.get("role") == "assistant")
    snippet    = memory.summary_text(limit=6)

    lines = [
        f"Session created : {memory.created_at}",
        f"User turns      : {user_turns}",
        f"Assistant turns : {asst_turns}",
        "",
        "Recent exchanges:",
        snippet,
    ]
    return "\n".join(lines)

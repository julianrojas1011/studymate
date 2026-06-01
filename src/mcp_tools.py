"""
MCP-facing tool implementations for StudyMate.

This module is a thin adapter layer: it reuses the exact same LLMClient,
Memory, and RAG code that the CLI REPL uses.  The MCP server and the
terminal therefore produce identical answers given identical inputs.

Lazy singletons (_client, _rag) are initialised on first use so importing
this module does not trigger network calls, model downloads, or file I/O.

MCP callers are stateless — ask_studymate does NOT persist memory between
calls.  Each invocation is self-contained: system prompt + few-shot examples
+ the single user question.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .llm import LLMClient, LLMError, _MessageShim
from .memory import Memory, MEMORY_FILE
from .prompts import SYSTEM_PROMPT, FEW_SHOT_EXAMPLES
from .rag import (
    RagIndex,
    Chunk,
    _EMBED_MODEL,
    _get_model,
    _embed,
    run_retrieve_tool,
    RETRIEVE_TOOL_SCHEMA,
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
_rag: RagIndex | None = None
_rag_path: Path | None = None   # tracks which docs path the current _rag was built from


def _get_client() -> LLMClient:
    """Return the shared LLMClient, initialising it on first call."""
    global _client
    if _client is None:
        _client = LLMClient()
    return _client


def _get_rag(docs: str) -> RagIndex | None:
    """Return a RagIndex for *docs*, reusing the cached singleton when possible.

    Returns None if *docs* is falsy or points to a non-existent path so
    callers can degrade gracefully to non-RAG answers.
    """
    global _rag, _rag_path

    if not docs:
        return None

    docs_path = Path(docs)
    if not docs_path.exists():
        return None

    # Rebuild only when the docs path has changed; build_index itself handles
    # hash-based cache invalidation when files inside the folder change.
    if _rag is None or _rag_path != docs_path:
        from .rag import build_index
        _rag      = build_index(docs_path, verbose=False)
        _rag_path = docs_path

    return _rag


# ---------------------------------------------------------------------------
# ask_studymate
# ---------------------------------------------------------------------------

def ask_studymate(question: str, docs: str | None = None) -> str:
    """Answer a study question with full tool-use support.

    Builds a minimal one-shot conversation (system prompt + few-shot examples
    + user question), runs the agentic tool loop, and returns the final
    assistant text.  Memory is NOT persisted — each MCP call is stateless.

    Args:
        question: The student's question.
        docs:     Optional path to a notes folder or file.  When provided,
                  the retrieve_from_notes tool is made available.

    Returns:
        The assistant's answer as a plain string, or an error message if
        the API call fails.
    """
    question = question.strip()
    if not question:
        return "Error: question must not be empty."

    try:
        client = _get_client()
    except LLMError as exc:
        return f"Configuration error: {exc}"

    rag_index = _get_rag(docs) if docs else None

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
            except json.JSONDecodeError:
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

    # If we exhaust _MAX_TOOL_ROUNDS without a clean finish, return whatever
    # the model last produced rather than silently returning nothing.
    last = messages[-1]
    return (
        last.get("content")
        or "Error: tool loop exceeded maximum rounds without producing an answer."
    )


# ---------------------------------------------------------------------------
# add_to_knowledge_base
# ---------------------------------------------------------------------------

def add_to_knowledge_base(
    text: str,
    source: str,
    cache_dir: str = "rag_index",
) -> str:
    """Add a text passage to the RAG index without re-embedding everything.

    Loads the existing index state (from the in-memory singleton, or from
    the disk cache, or starts empty), creates a single new Chunk, embeds it,
    and stacks it onto the existing embeddings matrix.  Saves the updated
    index to disk and refreshes the singleton.

    This is an incremental O(1) embedding call — we never re-embed the
    existing corpus.

    Args:
        text:      The passage to add.
        source:    A short label used in citations (e.g. "lecture_4").
        cache_dir: Directory where meta.json and vectors.npy are stored.

    Returns:
        A confirmation string such as "added chunk lecture_4#3 (total 42 chunks)".
    """
    text   = text.strip()
    source = source.strip()

    if not text:
        return "Error: text must not be empty."
    if not source:
        return "Error: source must not be empty."

    global _rag

    cache_path = Path(cache_dir)
    meta_path  = cache_path / "meta.json"
    vecs_path  = cache_path / "vectors.npy"

    # --- Load existing state ------------------------------------------------
    existing_chunks: list[dict]        = []
    existing_vecs:   np.ndarray | None = None
    manifest_hash = ""
    model_name    = _EMBED_MODEL

    if _rag is not None:
        # Fast path: singleton already loaded in memory.
        existing_chunks = [
            {"source": c.source, "chunk_id": c.chunk_id, "text": c.text}
            for c in _rag.chunks
        ]
        existing_vecs = _rag.embeddings
        manifest_hash = _rag.manifest_hash
        model_name    = _rag.model_name
    elif meta_path.exists() and vecs_path.exists():
        # Warm path: singleton is cold but cache is on disk.
        try:
            meta            = json.loads(meta_path.read_text(encoding="utf-8"))
            existing_chunks = meta.get("chunks", [])
            manifest_hash   = meta.get("manifest_hash", "")
            model_name      = meta.get("model_name", _EMBED_MODEL)
            existing_vecs   = np.load(str(vecs_path))
        except (json.JSONDecodeError, OSError, KeyError):
            # Corrupt cache — start fresh rather than crashing.
            pass

    # --- Create the new chunk -----------------------------------------------
    # chunk_id is the count of chunks already attributed to this source,
    # giving each source its own zero-based sequence of IDs.
    next_id   = sum(1 for c in existing_chunks if c["source"] == source)
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
        "manifest_hash": manifest_hash,
        "model_name":    model_name,
        "chunks":        all_chunk_dicts,
    }
    meta_path.write_text(json.dumps(meta_out, indent=2, ensure_ascii=False), encoding="utf-8")
    np.save(str(vecs_path), all_vecs)

    # --- Refresh singleton --------------------------------------------------
    _rag = RagIndex(
        chunks=[
            Chunk(source=c["source"], chunk_id=c["chunk_id"], text=c["text"])
            for c in all_chunk_dicts
        ],
        embeddings=all_vecs,
        model_name=model_name,
        manifest_hash=manifest_hash,
    )

    return f"added chunk {new_chunk.citation} (total {len(_rag)} chunks)"


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

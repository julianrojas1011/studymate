"""
StudyMate — CLI entry point and REPL.

Usage:
  python main.py                     # plain chat
  python main.py --docs notes/       # chat + RAG over a notes folder
  python main.py --docs lecture.pdf  # chat + RAG over a single file

In-session commands:
  /help     show this help text
  /clear    wipe conversation history and start fresh
  /summary  print a recap of the current session
  quit / exit / :q / q  — exit the REPL

main.py is the conductor: it wires together all src/ modules and runs the
conversation loop.  No business logic lives here — only orchestration.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
load_dotenv()

from src.llm import LLMClient, LLMError, _MessageShim
from src.memory import Memory, MEMORY_FILE
from src.prompts import SYSTEM_PROMPT, FEW_SHOT_EXAMPLES
from src.rag import RagIndex, build_index, run_retrieve_tool, RETRIEVE_TOOL_SCHEMA
from src.tools import TOOL_SCHEMAS, run_tool

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_TOOL_ROUNDS = 6
EXIT_WORDS      = {"quit", "exit", ":q", "q"}

HELP_TEXT = """\
┌─────────────────────────────────────────────────────────┐
│                   StudyMate commands                    │
├──────────────┬──────────────────────────────────────────┤
│ /help        │ Show this help text                      │
│ /clear       │ Wipe history and start a fresh session   │
│ /summary     │ Print a recap of this session            │
│ quit / exit  │ Save history and exit                    │
│ :q  / q      │ Same as quit                             │
└──────────────┴──────────────────────────────────────────┘
Just type your question to ask StudyMate anything.
"""

# ---------------------------------------------------------------------------
# Helper: system message builder
# ---------------------------------------------------------------------------

def build_system_messages(rag_loaded: bool) -> list[dict]:
    """Return [system_dict, *FEW_SHOT_EXAMPLES] to prepend to every API call.

    When *rag_loaded* is True, an extra instruction is appended to the system
    content reminding the model to prefer retrieve_from_notes and cite sources.
    The few-shot examples are injected here (not stored in memory) so they
    are always fresh and never duplicated in the saved history file.
    """
    content = SYSTEM_PROMPT
    if rag_loaded:
        content += (
            "\n\nNOTES AVAILABLE\n"
            "The student has loaded their lecture notes.  Before answering any "
            "factual or conceptual question, call retrieve_from_notes first.  "
            "Always cite retrieved passages inline as [source: filename#chunk_id]."
        )
    return [{"role": "system", "content": content}, *FEW_SHOT_EXAMPLES]


# ---------------------------------------------------------------------------
# Helper: serialise _MessageShim → plain dict for memory
# ---------------------------------------------------------------------------

def serialise_assistant_message(msg: _MessageShim) -> dict:
    """Convert a _MessageShim into the dict shape that memory and llm.py expect.

    tool_calls are serialised as the nested id/type/function structure that
    _to_anthropic_messages in llm.py knows how to convert.
    """
    out: dict[str, Any] = {"role": "assistant", "content": msg.content}
    if msg.tool_calls:
        out["tool_calls"] = [
            {
                "id":       tc.id,
                "type":     tc.type,
                "function": {
                    "name":      tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in msg.tool_calls
        ]
    return out


# ---------------------------------------------------------------------------
# Helper: tool dispatcher
# ---------------------------------------------------------------------------

def dispatch_tool(name: str, arguments: dict, rag: RagIndex | None) -> str:
    """Route a single tool call to the correct implementation.

    Always returns a string — errors are returned as strings so the model
    can read them and tell the student what went wrong.
    """
    if name == "retrieve_from_notes":
        if rag is None:
            return "Error: no notes index is loaded.  Start StudyMate with --docs <path> to enable retrieval."
        return run_retrieve_tool(rag, arguments)
    return run_tool(name, arguments)


# ---------------------------------------------------------------------------
# Helper: display truncation
# ---------------------------------------------------------------------------

def _short(value: Any, n: int = 80) -> str:
    """Truncate *value* to *n* chars for single-line display."""
    text = str(value).replace("\n", " ")
    return text[:n] + "…" if len(text) > n else text


# ---------------------------------------------------------------------------
# Helper: RAG loader
# ---------------------------------------------------------------------------

def load_rag(docs: str) -> RagIndex | None:
    """Build a RAG index for *docs* and return it, or None on failure.

    Prints a warning on failure so the REPL can continue without RAG rather
    than crashing at startup.
    """
    try:
        index = build_index(Path(docs), verbose=True)
        if len(index) == 0:
            print(f"[rag] Warning: no text could be extracted from '{docs}'. RAG disabled.")
            return None
        return index
    except Exception as exc:
        print(f"[rag] Warning: could not build index for '{docs}': {exc}. RAG disabled.")
        return None


# ---------------------------------------------------------------------------
# Core turn: run one user→assistant exchange including the tool loop
# ---------------------------------------------------------------------------

def run_turn(
    client: LLMClient,
    memory: Memory,
    user_input: str,
    tools: list[dict],
    rag: RagIndex | None,
) -> str:
    """Append *user_input* to memory, run the agentic loop, return the reply.

    The tool loop runs for at most MAX_TOOL_ROUNDS iterations.  Each iteration:
      1. Builds the full message list (system + history).
      2. Calls the LLM.
      3. If the model returns tool calls, executes them and loops.
      4. If the model returns plain text, returns it immediately.

    All messages (user, assistant, tool results) are appended to *memory* so
    the full conversation is preserved and saved at the end of the session.
    """
    memory.append({"role": "user", "content": user_input})

    for round_num in range(MAX_TOOL_ROUNDS):
        full_messages = build_system_messages(rag is not None) + memory.messages()

        try:
            response = client.chat(full_messages, tools=tools or None)
        except LLMError as exc:
            return f"[error] {exc}"

        assistant_dict = serialise_assistant_message(response)
        memory.append(assistant_dict)

        # Clean finish — model produced a final answer with no further tool calls.
        if not response.tool_calls:
            return response.content or ""

        # Execute each tool call, print a summary line, and record the result.
        for tc in response.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}

            result = dispatch_tool(tc.function.name, args, rag)

            # One-line progress indicator so the student isn't staring at silence.
            args_preview   = _short(tc.function.arguments)
            result_preview = _short(result)
            print(f"  [tool] {tc.function.name}({args_preview}) → {result_preview}")

            memory.append({
                "role":         "tool",
                "tool_call_id": tc.id,
                "name":         tc.function.name,
                "content":      result,
            })

    return "[error] Tool loop exceeded maximum rounds without producing a final answer."


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="studymate",
        description="AI study assistant powered by Claude.",
    )
    parser.add_argument(
        "--docs",
        metavar="PATH",
        help="Path to a notes folder or file to enable RAG retrieval.",
    )
    parser.add_argument(
        "--memory-file",
        metavar="FILE",
        default=str(MEMORY_FILE),
        help=f"Path to the conversation history JSON file (default: {MEMORY_FILE}).",
    )
    args = parser.parse_args(argv)

    # --- Initialise LLM client --------------------------------------------
    try:
        client = LLMClient()
    except LLMError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    # --- Load conversation memory -----------------------------------------
    memory = Memory(path=Path(args.memory_file))
    memory.load()
    prior_turns = len([m for m in memory.messages() if m.get("role") == "user"])

    # --- Load RAG index (optional) ----------------------------------------
    rag: RagIndex | None = None
    if args.docs:
        rag = load_rag(args.docs)

    # --- Build active tool list -------------------------------------------
    tools = list(TOOL_SCHEMAS)
    if rag is not None:
        tools.append(RETRIEVE_TOOL_SCHEMA)

    # --- Startup banner ---------------------------------------------------
    rag_status = f"notes: {args.docs}" if rag else "no notes loaded (use --docs to enable RAG)"
    model_tag  = f"{client.provider}/{client.model}"
    print(f"\nStudyMate  ·  {model_tag}  ·  {rag_status}")
    if prior_turns:
        print(f"Resuming session ({prior_turns} prior question(s)). Type /summary to recap.")
    print('Type /help for commands or just ask a question.\n')

    # --- REPL -------------------------------------------------------------
    try:
        while True:
            try:
                user_input = input("you> ").strip()
            except EOFError:
                # Piped input or Ctrl-D — exit cleanly.
                break

            if not user_input:
                continue

            # In-session commands
            if user_input.lower() in EXIT_WORDS:
                break

            if user_input == "/help":
                print(HELP_TEXT)
                continue

            if user_input == "/clear":
                memory.clear()
                print("[studymate] History cleared. Starting fresh.\n")
                continue

            if user_input == "/summary":
                print(memory.summary_text())
                print()
                continue

            # Regular question — run the agentic turn
            reply = run_turn(client, memory, user_input, tools, rag)
            print(f"\nstudymate> {reply}\n")
            memory.save()

    finally:
        # Always attempt to save on exit, even after an exception.
        try:
            memory.save()
        except Exception:
            pass

    print("Bye!")
    return 0


if __name__ == "__main__":
    sys.exit(main())

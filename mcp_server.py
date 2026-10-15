"""
StudyMate MCP server — exposes StudyMate tools over stdio.

Run directly:
  python mcp_server.py

Or register in an MCP client's configuration (see README.md).
The server uses stdio transport so Claude Desktop can spawn and communicate
with it as a subprocess.
"""

from __future__ import annotations

import asyncio
import os
import sys

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from src.mcp_tools import ask_studymate, add_to_knowledge_base, get_session_summary

# ---------------------------------------------------------------------------
# Server instance
# ---------------------------------------------------------------------------

server = Server("studymate")

# ---------------------------------------------------------------------------
# Tool listing
# ---------------------------------------------------------------------------

@server.list_tools()
async def list_tools() -> list[Tool]:
    """Advertise the three StudyMate tools to the MCP client."""
    return [
        Tool(
            name="ask_studymate",
            description=(
                "Ask StudyMate a study question.  The assistant answers using its "
                "full tool suite (calculator, web search, unit converter).  "
                "Optionally point it at a notes folder to enable retrieval-augmented "
                "answers grounded in the student's own lecture materials."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The study question to answer.",
                    },
                    "docs": {
                        "type": "string",
                        "description": (
                            "Optional path to a notes folder or file.  "
                            "When provided, StudyMate can retrieve relevant "
                            "passages and cite them in its answer."
                        ),
                    },
                    "cache_dir": {
                        "type": "string",
                        "description": "Notes index directory (default: rag_index).",
                    },
                },
                "required": ["question"],
            },
        ),
        Tool(
            name="add_to_knowledge_base",
            description=(
                "Add a text passage to StudyMate's RAG index without re-embedding "
                "the entire corpus.  Useful for incrementally inserting a new note, "
                "summary, or definition so it becomes retrievable in future queries."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The passage to add to the knowledge base.",
                    },
                    "source": {
                        "type": "string",
                        "description": (
                            "Short label used in citations, e.g. 'lecture_4' or 'my_notes'. "
                            "Appears as [source: label#N] in retrieved results."
                        ),
                    },
                    "cache_dir": {
                        "type": "string",
                        "description": "Directory where the index is stored (default: rag_index).",
                    },
                },
                "required": ["text", "source"],
            },
        ),
        Tool(
            name="get_session_summary",
            description=(
                "Return a summary of the most recent StudyMate CLI session: "
                "creation timestamp, turn counts, and a snippet of recent exchanges."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "memory_file": {
                        "type": "string",
                        "description": (
                            "Path to the memory JSON file.  Defaults to the "
                            "STUDYMATE_MEMORY_FILE env var, or memory.json."
                        ),
                    },
                },
                "required": [],
            },
        ),
    ]


# ---------------------------------------------------------------------------
# Tool dispatch
# ---------------------------------------------------------------------------

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Dispatch an incoming tool call to the appropriate implementation.

    All three underlying functions are synchronous (they call blocking I/O and
    the Anthropic/OpenAI SDK).  asyncio.to_thread runs them in a thread-pool
    worker so the event loop stays unblocked.
    """
    try:
        if name == "ask_studymate":
            question = arguments.get("question", "")
            docs     = arguments.get("docs")
            cache_dir = arguments.get("cache_dir", "rag_index")
            result = await asyncio.to_thread(ask_studymate, question, docs, cache_dir)

        elif name == "add_to_knowledge_base":
            text      = arguments.get("text", "")
            source    = arguments.get("source", "")
            cache_dir = arguments.get("cache_dir", "rag_index")
            result    = await asyncio.to_thread(add_to_knowledge_base, text, source, cache_dir)

        elif name == "get_session_summary":
            memory_file = arguments.get("memory_file")
            result      = await asyncio.to_thread(get_session_summary, memory_file)

        else:
            result = f"Error: unknown tool '{name}'."

    except Exception as exc:
        result = f"Error: {exc}"

    return [TextContent(type="text", text=result)]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    if not os.environ.get("STUDYMATE_API_KEY"):
        print(
            "Error: STUDYMATE_API_KEY is not set.\n"
            "Copy .env.example to .env and add your API key, then re-run.",
            file=sys.stderr,
        )
        sys.exit(1)

    async def _run() -> None:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())

    asyncio.run(_run())


if __name__ == "__main__":
    main()

# StudyMate

StudyMate is a personal AI study assistant that runs in your terminal.  It remembers your conversation across sessions, can search your own lecture notes for grounded answers, calls a safe calculator for any arithmetic, and exposes all its capabilities as an MCP server so you can talk to it from Claude Desktop.  It supports Anthropic (Claude), Groq, and any OpenAI-compatible provider through a single environment variable.

---

## Features

| Milestone | What was built |
|-----------|----------------|
| **M1 — LLM client** | Provider-agnostic wrapper supporting Anthropic and Groq/OpenAI. Converts between OpenAI-style and Anthropic message schemas transparently. |
| **M2 — Memory** | Two-layer persistence: in-process list for the session, atomic JSON write to disk on every turn so no history is lost on crash. |
| **M3 — Prompts** | System prompt with role, chain-of-thought, honesty, and tool instructions. Two CS few-shot examples prime format and depth. |
| **M4 — Tools** | Calculator (AST-based, no `eval`), web search (DuckDuckGo), unit converter (temperature/length/mass/time), datetime. |
| **M5 — RAG** | Local embeddings via `sentence-transformers`. SHA-256 manifest cache so re-starts are instant. Brute-force cosine retrieval over numpy — no external vector DB needed. |
| **M6 — MCP server** | Stdio MCP server exposing `ask_studymate`, `add_to_knowledge_base`, and `get_session_summary` to Claude Desktop or any MCP client. |

---

## Project layout

```
StudyMate/
├── main.py               # CLI entry point and REPL
├── mcp_server.py         # MCP stdio server
├── requirements.txt      # Pinned dependencies
├── .env.example          # Environment variable template
├── .gitignore
├── memory.example.json   # Example of the saved history format
├── PROMPTS.md            # Prompt design rationale
├── notes/
│   └── sample_lecture.txt  # Sample lecture notes for testing RAG
├── rag_index/            # Auto-generated at runtime (gitignored)
│   ├── meta.json         #   Chunk metadata + manifest hash
│   └── vectors.npy       #   L2-normalised float32 embeddings
└── src/
    ├── __init__.py
    ├── llm.py            # LLM client (Anthropic + OpenAI/Groq)
    ├── memory.py         # Conversation persistence
    ├── prompts.py        # System prompt + few-shot examples
    ├── tools.py          # Tool schemas + implementations
    ├── rag.py            # RAG pipeline (chunk / embed / retrieve)
    └── mcp_tools.py      # MCP-facing functions (thin adapter layer)
```

---

## Setup

1. **Clone the repo**
   ```bash
   git clone <repo-url>
   cd StudyMate
   ```

2. **Create a virtual environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure your API key**
   ```bash
   cp .env.example .env
   # Open .env in your editor and set STUDYMATE_API_KEY
   ```

---

## API keys

| Provider | Where to get a key | Default model |
|----------|--------------------|---------------|
| **Anthropic** (default) | [console.anthropic.com](https://console.anthropic.com) | `claude-opus-4-7` |
| **Groq** (free tier) | [console.groq.com/keys](https://console.groq.com/keys) | `llama-3.3-70b-versatile` |

To use Groq, set these in your `.env`:

```env
STUDYMATE_API_KEY=gsk_...
STUDYMATE_PROVIDER=groq
```

No `STUDYMATE_BASE_URL` needed for Groq — it is filled in automatically.

---

## Running

```bash
# Plain chat (no notes)
python main.py

# Chat with RAG over a folder of notes
python main.py --docs notes/

# Chat with RAG over a single PDF
python main.py --docs notes/lecture4.pdf
```

### In-session commands

| Command | Effect |
|---------|--------|
| `/help` | Show the command reference |
| `/clear` | Wipe history and start a fresh session |
| `/summary` | Print a recap of the current session |
| `quit` / `exit` / `:q` / `q` | Save history and exit |

---

## Example session

```
StudyMate  ·  anthropic/claude-opus-4-7  ·  notes: notes/
Type /help for commands or just ask a question.

you> What does my lecture say about the three variants of gradient descent?

  [tool] retrieve_from_notes({"query": "gradient descent variants"}) → [source: sample_lecture#2] (score=0.91) ...

studymate> According to your notes [source: sample_lecture#2], there are three
variants:

1. **Batch gradient descent** — computes the gradient over the full training
   set before each update. Stable and exact, but unusable on large datasets.

2. **Stochastic gradient descent (SGD)** — updates on a single random example
   per step. Very fast per update but noisy; requires a decayed learning rate.

3. **Mini-batch gradient descent** — the practical default. Averages the
   gradient over k examples (typically 32–128). GPU-parallelisable and much
   smoother than SGD.

you> How many parameters does a model with 3 layers of size 512, 256, 128 have?

  [tool] calculator({"expression": "512*256 + 256 + 256*128 + 128 + 128*1 + 1"}) → 163969

studymate> Using the calculator: 512×256 + 256 (biases) + 256×128 + 128 + 128×1 + 1
= **163,969 parameters** in total.

you> quit
Bye!
```

---

## MCP Server

The MCP server exposes StudyMate's capabilities to any MCP-compatible client (Claude Desktop, custom agents, etc.) over stdio.

```bash
python mcp_server.py
```

### Available MCP tools

| Tool | Description | Required params |
|------|-------------|-----------------|
| `ask_studymate` | Answer a study question using the full tool suite | `question` |
| `add_to_knowledge_base` | Incrementally add a passage to the RAG index | `text`, `source` |
| `get_session_summary` | Return turn counts and a snippet of recent CLI history | — |

### Claude Desktop configuration

Add this block to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "studymate": {
      "command": "python",
      "args": ["/absolute/path/to/StudyMate/mcp_server.py"],
      "env": {
        "STUDYMATE_API_KEY": "your-api-key-here",
        "STUDYMATE_PROVIDER": "anthropic"
      }
    }
  }
}
```

Replace `/absolute/path/to/StudyMate/` with the actual path on your machine.

---

## Safety notes

**No `eval()` in the calculator.** The calculator tool uses Python's `ast` module to parse the expression into a syntax tree and then walks only explicitly whitelisted node types (arithmetic operators, a fixed set of `math` functions, and the constants `pi` and `e`). Any other node type — including imports, attribute access, lambdas, or function definitions — raises an error before any code is executed. This makes it safe to pass model-generated expressions to the calculator without risk of arbitrary code execution.

**API keys are env-only.** Keys are read exclusively from environment variables (or a `.env` file that is gitignored). They are never written to memory.json, logged to stdout, or included in any MCP response.

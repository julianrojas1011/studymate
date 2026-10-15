# StudyMate

A personal project I built to understand how an agent connects tools, memory, and retrieval. It runs in the terminal and can answer questions using my lecture notes.

## What it does

- Searches text, Markdown, and PDF notes using local embeddings and cosine similarity.
- Gives the model tools for calculations, unit conversions, web search, and the current time.
- Saves CLI conversation history between sessions.
- Exposes three tools through a local MCP server: ask a question, add a passage to the notes index, and read a summary of saved CLI history.
- Supports Anthropic and OpenAI-compatible APIs through one client wrapper.

## Try it

You need Python, a virtual environment, and an API key for the provider you choose. The first notes-indexing run downloads the embedding model.

```bash
git clone https://github.com/julianrojas1011/studymate.git
cd studymate
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

On Windows, activate with `.venv\Scripts\activate` instead. Set `STUDYMATE_API_KEY` in `.env`. Choose the provider with `STUDYMATE_PROVIDER` and, if needed, a model available to your account with `STUDYMATE_MODEL`. OpenAI-compatible endpoints can use `STUDYMATE_BASE_URL`.

```bash
python main.py --docs notes/
```

Try asking: `What do my notes say about gradient descent?` The terminal shows tool calls as they happen. Retrieved passages have source labels that the model is instructed to cite.

Run without `--docs` for plain chat. Use `/help`, `/summary`, `/clear`, or `quit` during a session.

## How it fits together

`main.py` runs a bounded tool-calling loop. `src/llm.py` translates provider message formats; `src/tools.py` dispatches tools. `src/rag.py` chunks documents, caches embeddings, and retrieves passages. `src/memory.py` saves the conversation as JSON after each completed CLI turn.

The MCP entry point is `python mcp_server.py`. Configure your MCP client to launch it with the virtual environment's Python executable, the absolute script path, and your provider environment variables. MCP questions are independent calls; they do not continue the CLI conversation. `get_session_summary` reads the CLI memory file.

## Scope and limitations

This is a learning project. Retrieval and citation prompts do not guarantee correct answers. The calculator restricts accepted Python syntax with an AST allowlist; it is not a general sandbox for untrusted workloads. `/summary` returns recent conversation excerpts, not an LLM-generated summary.

Embeddings run locally, but questions, conversation context, and retrieved text used for an answer are sent to the configured model provider. The CLI saves history in `memory.json`; keep private notes and history out of Git. An interrupted turn may not have been saved yet.

[Prompt notes](PROMPTS.md) explain the current instructions and what still needs evaluation.

# StudyMate

A personal experiment in agent orchestration. I built StudyMate to understand how a model chooses tools, receives their results, retrieves useful context, and carries a conversation between sessions. Study questions and my lecture notes gave me a concrete setting to test those pieces together.

## What it does

- Searches text, Markdown, and PDF notes using local embeddings and cosine similarity.
- Gives the model tools for calculations, unit conversions, web search, and the current time.
- Saves CLI conversation history between sessions.
- Exposes three tools through a local MCP server: ask a question, add a passage to the notes index, and read a summary of saved CLI history.
- Supports Anthropic and OpenAI-compatible APIs through one client wrapper.

## Try it

Use Python **3.12** and a virtual environment. Model answers need an API key for the provider you choose. The first notes-indexing run downloads the embedding model.

```bash
git clone https://github.com/julianrojas1011/studymate.git
cd studymate
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

On Windows, create the environment with `py -3.12 -m venv .venv` and activate with `.venv\Scripts\activate`. Set `STUDYMATE_API_KEY` in `.env`. Choose the provider with `STUDYMATE_PROVIDER` and, if needed, a model available to your account with `STUDYMATE_MODEL`. OpenAI-compatible endpoints use the `openai` provider and `STUDYMATE_BASE_URL`.

```bash
python main.py --docs notes/
```

Try asking: `What do my notes say about gradient descent?` The terminal shows tool calls as they happen. Retrieved passages have source labels that the model is instructed to cite.

Run without `--docs` for plain chat. Use `/help`, `/summary`, `/clear`, or `quit` during a session.

## A small example without an API key

From the repository root, run this with Python; no packages or model downloads are needed:

```bash
python -m examples.offline_tools
```

It dispatches `sqrt(144)` → `12`, converts two hours to minutes, handles an invalid calculator call, then saves and reloads six messages in a temporary file. This exercises the tools and persistence directly; it is not a simulated claim about a model's answer quality.

## How it fits together

`main.py` runs a bounded tool-calling loop. `src/llm.py` translates provider message formats; `src/tools.py` dispatches tools. `src/rag.py` chunks documents, caches embeddings, and retrieves passages. `src/memory.py` saves the conversation as JSON after each completed CLI turn.

The MCP entry point is `python mcp_server.py`. Configure your MCP client to launch it with the virtual environment's Python executable, the absolute script path, and your provider environment variables. Set the working directory to this repository, or use absolute paths for notes, memory, and `cache_dir`.

MCP questions are independent calls; they do not continue the CLI conversation. `get_session_summary` reads the CLI memory file. `add_to_knowledge_base` saves passages in `cache_dir` (default `rag_index`); `ask_studymate` can retrieve them from the same directory without a `docs` argument. Passing `docs` builds or refreshes a file-based index, replacing that cache when the source files change. Use separate cache directories for file-based and manually added notes.

## Checks

The regression suite uses fixed vectors, mocked model responses, and temporary files. It covers tool validation, calculator limits, memory recovery, chunk boundaries, cache isolation, provider message conversion, and the bounded CLI/MCP tool loop. It makes no provider calls and downloads no embedding model.

```bash
pip install -r requirements-core.txt
python -m unittest discover -s tests -v
```

GitHub Actions runs these checks on Python 3.11 and 3.12, plus a full dependency-install smoke check. These are software regression tests, not an evaluation of retrieval relevance, tutoring quality, or learning outcomes.

## Scope and limitations

This is a learning project using pretrained models, not a deployed educational product or a model-training experiment. Retrieval and citation prompts do not guarantee correct answers. The calculator uses an AST allowlist and resource limits; it is not a general sandbox for untrusted workloads. `/summary` returns recent conversation excerpts, not an LLM-generated summary.

Embeddings run locally, but questions, conversation context, and retrieved text used for an answer are sent to the configured model provider. The CLI saves history in `memory.json`; keep private notes and history out of Git. An interrupted turn may not have been saved yet.

The MCP server is intended for a trusted local client. It can read the note and memory paths you supply and write its index. Run one process per memory/index location; this experiment does not provide authentication, multi-user isolation, automatic context compaction, or cross-process write coordination.

[Prompt notes](PROMPTS.md) explain the current instructions and what still needs evaluation.

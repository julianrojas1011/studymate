"""
RAG (Retrieval-Augmented Generation) pipeline for StudyMate.

Design decisions:

  Local embeddings (sentence-transformers):
    We use a locally-run model instead of an embedding API so the pipeline
    works offline, costs nothing per query, and keeps student notes private.
    all-MiniLM-L6-v2 is ~22 MB, fast on CPU, and accurate enough for
    lecture-note retrieval.

  Numpy over FAISS/ChromaDB:
    A student's notes folder rarely exceeds a few thousand chunks.  At that
    scale, a brute-force dot-product scan over a float32 numpy matrix is
    measured in milliseconds and adds zero external dependencies.  FAISS pays
    off at millions of vectors; we'd be importing complexity for no gain.

  L2-normalised vectors + dot product = cosine similarity:
    When every vector has unit length, dot(a, b) == cosine_similarity(a, b).
    This lets us use a single matrix-vector multiply (one BLAS call) instead
    of the more expensive full cosine formula, and lets us store embeddings
    in a plain float32 array with no extra bookkeeping.

  Disk cache (manifest hash):
    Embedding is the slow step (~10 s cold on CPU).  We hash each file's
    path, size, and mtime to detect any change in the notes folder.  If the
    hash and model name both match the cached index, we skip embedding and
    load the pre-computed vectors from disk in ~0.2 s.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EMBED_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
_CHUNK_CHARS   = 1600   # ~400 tokens at ~4 chars/token
_CHUNK_OVERLAP = 200    # ~50 token overlap so context isn't lost at boundaries

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    source:   str   # filename stem (for readable citations)
    chunk_id: int
    text:     str

    @property
    def citation(self) -> str:
        return f"{self.source}#{self.chunk_id}"


@dataclass
class RagIndex:
    chunks:        list[Chunk]
    embeddings:    np.ndarray | None   # shape (N, dim), float32, L2-normalised
    model_name:    str
    manifest_hash: str

    def __len__(self) -> int:
        return len(self.chunks)


# ---------------------------------------------------------------------------
# File loading
# ---------------------------------------------------------------------------

def _read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    pages: list[str] = []
    for i, page in enumerate(reader.pages):
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            # A corrupt or image-only page shouldn't abort the whole document.
            pages.append(f"[page {i + 1}: could not extract text]")
    return "\n".join(pages)


def _load_doc(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return _read_pdf(path)
    return _read_txt(path)


def _collect_files(path: Path) -> list[Path]:
    """Return supported document files under *path*.

    If *path* is a single file, return it directly.  Otherwise walk the
    directory tree and collect .txt, .md, and .pdf files.
    """
    if path.is_file():
        return [path]
    files: list[Path] = []
    for pattern in ("*.txt", "*.md", "*.pdf"):
        files.extend(sorted(path.rglob(pattern)))
    return files


# ---------------------------------------------------------------------------
# Change detection
# ---------------------------------------------------------------------------

def _manifest_hash(files: list[Path]) -> str:
    """SHA-256 over each file's absolute path, size, and mtime.

    Any addition, deletion, rename, or in-place edit changes the hash and
    triggers a full re-embed.  Sorted by path so the hash is stable
    regardless of filesystem iteration order.
    """
    h = hashlib.sha256()
    for f in sorted(files):
        stat = f.stat()
        h.update(f"{f.resolve()}:{stat.st_size}:{stat.st_mtime}".encode())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(text: str, source: str) -> list[Chunk]:
    """Split *text* into overlapping chunks using a sliding window.

    At each window boundary we search backward for a clean break point so
    chunks don't split mid-sentence or mid-paragraph.  We only accept a
    break if it falls past the halfway point of the window — this prevents
    the degeneracy where a very early break creates a tiny leading chunk.
    """
    chunks: list[Chunk] = []
    start    = 0
    chunk_id = 0
    n        = len(text)

    while start < n:
        end = min(start + _CHUNK_CHARS, n)

        if end < n:
            halfway   = start + _CHUNK_CHARS // 2
            best_break = end  # fallback: hard cut at window boundary

            for separator in ("\n\n", ". ", "\n", " "):
                pos = text.rfind(separator, halfway, end)
                if pos != -1:
                    best_break = pos + len(separator)
                    break

            end = best_break

        content = text[start:end].strip()
        if content:
            chunks.append(Chunk(source=source, chunk_id=chunk_id, text=content))
            chunk_id += 1

        # Slide forward, backing up by _CHUNK_OVERLAP so adjacent chunks
        # share context across their boundary.
        start = max(start + 1, end - _CHUNK_OVERLAP)

    return chunks


# ---------------------------------------------------------------------------
# Embedding — lazy model cache
# ---------------------------------------------------------------------------

# Module-level dict so the SentenceTransformer is loaded once and reused.
# Initialisation takes 1–3 s; we don't want that cost on every retrieve().
_MODEL_CACHE: dict[str, Any] = {}


def _get_model(model_name: str) -> Any:
    if model_name not in _MODEL_CACHE:
        from sentence_transformers import SentenceTransformer
        _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _MODEL_CACHE[model_name]


def _embed(model: Any, texts: list[str]) -> np.ndarray:
    """Embed *texts* and return a float32 array with L2-normalised rows.

    Normalising at embed time means dot(a, b) == cosine_similarity(a, b)
    for any two rows.  Retrieval then reduces to a single matrix-vector
    multiply with no need to recompute norms at query time.
    """
    vectors = model.encode(
        texts,
        normalize_embeddings=True,   # L2-normalise each row in-place
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return vectors.astype(np.float32)


# ---------------------------------------------------------------------------
# Index building
# ---------------------------------------------------------------------------

def build_index(
    docs_path: Path,
    cache_dir: Path = Path("rag_index"),
    model_name: str = _EMBED_MODEL,
    verbose: bool = True,
) -> RagIndex:
    """Build (or load from cache) a RAG index for all documents under *docs_path*.

    Cache layout inside *cache_dir*:
      meta.json   — manifest_hash, model_name, serialised chunk list
      vectors.npy — float32 array of shape (N, dim), L2-normalised
    """
    files = _collect_files(docs_path)
    if not files:
        if verbose:
            print(f"[RAG] No supported files found at {docs_path}")
        return RagIndex(chunks=[], embeddings=None, model_name=model_name, manifest_hash="")

    current_hash = _manifest_hash(files)
    meta_path    = cache_dir / "meta.json"
    vecs_path    = cache_dir / "vectors.npy"

    # --- Cache hit -----------------------------------------------------------
    if meta_path.exists() and vecs_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if (
                meta.get("manifest_hash") == current_hash
                and meta.get("model_name") == model_name
            ):
                if verbose:
                    print("[RAG] Cache hit — loading index from disk.")
                chunks = [
                    Chunk(source=c["source"], chunk_id=c["chunk_id"], text=c["text"])
                    for c in meta["chunks"]
                ]
                embeddings = np.load(str(vecs_path))
                return RagIndex(
                    chunks=chunks,
                    embeddings=embeddings,
                    model_name=model_name,
                    manifest_hash=current_hash,
                )
        except (json.JSONDecodeError, KeyError, OSError):
            # Corrupt cache — fall through and rebuild.
            pass

    # --- Cache miss: build from scratch -------------------------------------
    if verbose:
        print(f"[RAG] Building index from {len(files)} file(s)…")

    all_chunks: list[Chunk] = []
    for f in files:
        if verbose:
            print(f"  Loading {f.name}…")
        try:
            text = _load_doc(f)
        except Exception as exc:
            if verbose:
                print(f"  Warning: could not read {f.name}: {exc}")
            continue
        all_chunks.extend(chunk_text(text, source=f.stem))

    if not all_chunks:
        if verbose:
            print("[RAG] No text could be extracted from the documents.")
        return RagIndex(chunks=[], embeddings=None, model_name=model_name, manifest_hash=current_hash)

    if verbose:
        print(f"[RAG] Embedding {len(all_chunks)} chunks with {model_name}…")

    model      = _get_model(model_name)
    embeddings = _embed(model, [c.text for c in all_chunks])

    # --- Persist cache -------------------------------------------------------
    cache_dir.mkdir(parents=True, exist_ok=True)

    meta_out = {
        "manifest_hash": current_hash,
        "model_name":    model_name,
        "chunks": [
            {"source": c.source, "chunk_id": c.chunk_id, "text": c.text}
            for c in all_chunks
        ],
    }
    meta_path.write_text(json.dumps(meta_out, indent=2, ensure_ascii=False), encoding="utf-8")
    np.save(str(vecs_path), embeddings)

    if verbose:
        print(f"[RAG] Index saved to {cache_dir} ({len(all_chunks)} chunks).")

    return RagIndex(
        chunks=all_chunks,
        embeddings=embeddings,
        model_name=model_name,
        manifest_hash=current_hash,
    )


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve(
    index: RagIndex,
    query: str,
    top_k: int = 4,
) -> list[tuple[Chunk, float]]:
    """Return the *top_k* most relevant chunks for *query*.

    Since embeddings are L2-normalised, matrix @ vector gives cosine
    similarity for every chunk in one BLAS call.
    """
    if index.embeddings is None or len(index) == 0:
        return []

    model     = _get_model(index.model_name)
    query_vec = _embed(model, [query])[0]           # shape (dim,)
    scores    = index.embeddings @ query_vec         # shape (N,), cosine sim
    order     = np.argsort(scores)[::-1][:top_k]    # descending

    return [(index.chunks[i], float(scores[i])) for i in order]


def format_context(hits: list[tuple[Chunk, float]]) -> str:
    """Format retrieval results into a context block for the model."""
    parts: list[str] = []
    for chunk, score in hits:
        parts.append(f"[source: {chunk.citation}] (score={score:.3f})\n{chunk.text}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Tool schema and dispatcher (registered by main.py when index is ready)
# ---------------------------------------------------------------------------

RETRIEVE_TOOL_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "retrieve_from_notes",
        "description": (
            "Searches the student's uploaded lecture notes and returns the most "
            "relevant passages. Call this tool whenever a question could plausibly "
            "be answered by the course materials — even if you think you already "
            "know the answer. Grounded answers from actual notes are always preferred "
            "over recalled training data. After retrieval, cite every claim with "
            "[source: filename#chunk_id]."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A natural-language search query describing what to look for.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of passages to retrieve (1–8, default 4).",
                    "default": 4,
                },
            },
            "required": ["query"],
        },
    },
}


def run_retrieve_tool(index: RagIndex, arguments: dict) -> str:
    """Execute a retrieve_from_notes tool call and return formatted context."""
    query = arguments.get("query", "").strip()
    if not query:
        return "Error: 'query' argument is required."

    top_k = max(1, min(int(arguments.get("top_k", 4)), 8))
    hits  = retrieve(index, query, top_k=top_k)

    if not hits:
        return "No relevant passages found in the student's notes for this query."

    return format_context(hits)

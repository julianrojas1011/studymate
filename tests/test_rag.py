import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from src.rag import (Chunk, RagIndex, build_index, chunk_text, format_context,
                     load_cached_index, retrieve, run_retrieve_tool)


class RagTests(unittest.TestCase):
    def test_short_document_becomes_one_chunk_not_repeated_suffixes(self):
        chunks = chunk_text("Gradient descent follows the negative gradient.", "notes")
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].citation, "notes#0")

    def test_long_document_has_overlap_and_stops_at_last_window(self):
        text = "a" * 4000
        chunks = chunk_text(text, "notes")
        self.assertEqual([len(c.text) for c in chunks], [1600, 1600, 1200])
        self.assertEqual([c.chunk_id for c in chunks], [0, 1, 2])

    def test_cosine_ranking_and_citations_with_fixed_vectors(self):
        # Fixed vectors isolate ranking logic, not embedding-model quality.
        index = RagIndex([Chunk("notes", 0, "gradient"), Chunk("notes", 1, "queues")],
                         np.array([[1., 0.], [0., 1.]], dtype=np.float32), "fake", "hash")
        with patch("src.rag._get_model"), patch("src.rag._embed", return_value=np.array([[1., 0.]])):
            hits = retrieve(index, "gradient", top_k=1)
        self.assertEqual(hits[0][0].chunk_id, 0)
        self.assertEqual(hits[0][1], 1.0)
        self.assertIn("[source: notes#0]", format_context(hits))

    def test_retrieval_validates_tool_arguments(self):
        index = RagIndex([], None, "fake", "")
        for args in ([], {"query": 3}, {"query": "topic", "top_k": "many"},
                     {"query": "topic", "top_k": True}):
            self.assertTrue(run_retrieve_tool(index, args).startswith("Error:"))

    def test_index_cache_hit_and_corrupt_vector_rebuild(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            docs = root / "notes.txt"
            docs.write_text("One short note.")
            cache = root / "cache"
            vectors = np.array([[1., 0.]], dtype=np.float32)
            with patch("src.rag._get_model"), patch("src.rag._embed", return_value=vectors) as embed:
                first = build_index(docs, cache, verbose=False)
                cached = build_index(docs, cache, verbose=False)
                self.assertEqual(len(first), 1)
                self.assertEqual(len(cached), 1)
                self.assertEqual(embed.call_count, 1)
                np.save(cache / "vectors.npy", np.ones((3, 2)))
                rebuilt = build_index(docs, cache, verbose=False)
                self.assertEqual(len(rebuilt), 1)
                self.assertEqual(embed.call_count, 2)

    def test_invalid_cache_shape_is_rejected(self):
        with TemporaryDirectory() as temporary:
            cache = Path(temporary)
            (cache / "meta.json").write_text(json.dumps([]))
            with self.assertRaises(ValueError):
                load_cached_index(cache)

    def test_same_filename_in_different_folders_has_distinct_citations(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            docs = root / "docs"
            for folder in ("math", "cs"):
                path = docs / folder / "lecture.MD"
                path.parent.mkdir(parents=True)
                path.write_text(f"A {folder} note.")
            with patch("src.rag._get_model"), patch("src.rag._embed", return_value=np.eye(2, dtype=np.float32)):
                index = build_index(docs, root / "cache", verbose=False)
            self.assertEqual({c.citation for c in index.chunks}, {"math/lecture#0", "cs/lecture#0"})

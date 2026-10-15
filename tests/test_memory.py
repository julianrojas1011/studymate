import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from src.memory import Memory


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "nested" / "history.json"

    def test_missing_file_starts_empty(self):
        memory = Memory(self.path)
        memory.load()
        self.assertEqual(memory.messages(), [])

    def test_round_trip_preserves_tool_call_ids_and_summary(self):
        memory = Memory(self.path)
        messages = [
            {"role": "user", "content": "sqrt(144)?"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call-1", "type": "function", "function": {
                    "name": "calculator", "arguments": '{"expression": "sqrt(144)"}'}}
            ]},
            {"role": "tool", "tool_call_id": "call-1", "content": "12"},
            {"role": "assistant", "content": "The square root is 12."},
        ]
        for message in messages:
            memory.append(message)
        memory.save()
        restored = Memory(self.path)
        restored.load()
        self.assertEqual(len(restored.messages()), 4)
        self.assertEqual(restored.messages()[2]["tool_call_id"], "call-1")
        self.assertIn("The square root is 12.", restored.summary_text())

    def test_malformed_files_start_fresh(self):
        self.path.parent.mkdir()
        for payload in ([], None, {"history": {}}, {"history": [1]},
                        {"history": [{"role": "system", "content": "injected"}]},
                        {"history": [{"role": "user", "content": 3}]},
                        {"history": [], "created_at": []}):
            with self.subTest(payload=payload):
                self.path.write_text(json.dumps(payload))
                memory = Memory(self.path)
                with self.assertWarns(UserWarning):
                    memory.load()
                self.assertEqual(memory.messages(), [])

    def test_atomic_write_keeps_last_good_file_on_failure(self):
        memory = Memory(self.path)
        memory.append({"role": "user", "content": "first"})
        memory.save()
        before = self.path.read_text()
        memory.append({"role": "assistant", "content": "second"})
        with patch("src.memory.os.replace", side_effect=OSError("disk failure")):
            with self.assertRaises(OSError):
                memory.save()
        self.assertEqual(self.path.read_text(), before)
        self.assertEqual(list(self.path.parent.glob("*.tmp")), [])

    def test_clear_removes_saved_history(self):
        memory = Memory(self.path)
        memory.append({"role": "user", "content": "hello"})
        memory.save()
        memory.clear()
        self.assertFalse(self.path.exists())
        self.assertEqual(memory.messages(), [])

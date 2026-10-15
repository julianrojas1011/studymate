from contextlib import redirect_stdout
from io import StringIO
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from main import MAX_TOOL_ROUNDS, run_turn
from src import mcp_tools
from src.llm import (LLMClient, LLMError, _FunctionShim, _MessageShim,
                     _ToolCallShim, _to_anthropic_messages)
from src.memory import Memory
from src.rag import load_cached_index
from src.tools import TOOL_SCHEMAS


def tool_response(arguments='{"expression":"sqrt(144)"}'):
    return _MessageShim(None, [_ToolCallShim("call-1", _FunctionShim("calculator", arguments))])


class AgentTests(unittest.TestCase):
    def test_cli_routes_tool_result_back_to_model(self):
        client = Mock()
        client.chat.side_effect = [tool_response(), _MessageShim("The result is 12.")]
        memory = Memory()
        with redirect_stdout(StringIO()):
            answer = run_turn(client, memory, "sqrt(144)?", TOOL_SCHEMAS, None)
        self.assertEqual(answer, "The result is 12.")
        second_request = client.chat.call_args_list[1].args[0]
        self.assertEqual(second_request[-1]["content"], "12")
        self.assertEqual(second_request[-1]["tool_call_id"], "call-1")

    def test_invalid_tool_json_returns_feedback_without_crashing(self):
        for arguments in ("[]", "null", "{broken"):
            client = Mock()
            client.chat.side_effect = [tool_response(arguments), _MessageShim("I could not calculate that.")]
            with redirect_stdout(StringIO()):
                run_turn(client, Memory(), "Calculate", TOOL_SCHEMAS, None)
            self.assertTrue(client.chat.call_args_list[1].args[0][-1]["content"].startswith("Error:"))

    def test_cli_stops_at_round_limit(self):
        client = Mock()
        client.chat.return_value = tool_response()
        with redirect_stdout(StringIO()):
            answer = run_turn(client, Memory(), "Loop", TOOL_SCHEMAS, None)
        self.assertIn("exceeded maximum rounds", answer)
        self.assertEqual(client.chat.call_count, MAX_TOOL_ROUNDS)

    def test_mcp_does_not_present_last_tool_result_as_final_answer(self):
        client = Mock()
        client.chat.return_value = tool_response()
        with patch.object(mcp_tools, "_get_client", return_value=client), patch.object(mcp_tools, "_get_rag", return_value=None):
            answer = mcp_tools.ask_studymate("Loop")
        self.assertIn("exceeded maximum rounds", answer)
        self.assertEqual(client.chat.call_count, mcp_tools._MAX_TOOL_ROUNDS)

    def test_added_notes_are_retrievable_and_cache_directories_stay_separate(self):
        with TemporaryDirectory() as temporary:
            a = str(Path(temporary) / "a")
            b = str(Path(temporary) / "b")
            with patch.object(mcp_tools, "_get_model"), patch.object(mcp_tools, "_embed", return_value=np.array([[1., 0.]], dtype=np.float32)):
                self.assertIn("total 1", mcp_tools.add_to_knowledge_base("First", "a", a))
                self.assertIn("total 1", mcp_tools.add_to_knowledge_base("Second", "b", b))
                self.assertIn("total 2", mcp_tools.add_to_knowledge_base("Third", "a", a))
            self.assertEqual([c.text for c in mcp_tools._get_rag(None, a).chunks], ["First", "Third"])
            self.assertEqual([c.text for c in load_cached_index(Path(b)).chunks], ["Second"])

    def test_invalid_existing_cache_is_not_silently_overwritten(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "meta.json"
            path.write_text("[]")
            result = mcp_tools.add_to_knowledge_base("New note", "test", temporary)
            self.assertTrue(result.startswith("Error:"))
            self.assertEqual(path.read_text(), "[]")

    def test_mcp_checks_changed_documents_each_call(self):
        with TemporaryDirectory() as temporary:
            docs = Path(temporary) / "notes.txt"
            docs.write_text("first")
            with patch.object(mcp_tools, "build_index") as build:
                mcp_tools._get_rag(str(docs))
                docs.write_text("changed")
                mcp_tools._get_rag(str(docs))
                self.assertEqual(build.call_count, 2)


class ProviderTests(unittest.TestCase):
    @patch.dict(os.environ, {}, clear=True)
    def test_clients_initialize_offline_and_openai_uses_own_default(self):
        for provider in ("anthropic", "openai", "groq"):
            with self.subTest(provider=provider):
                client = LLMClient({"api_key": "offline-placeholder", "provider": provider})
                if provider == "openai":
                    self.assertEqual(client.model, "gpt-4o")
                sdk = client._anthropic if provider == "anthropic" else client._openai
                sdk.close()

    @patch.dict(os.environ, {}, clear=True)
    def test_unknown_provider_fails_with_configuration_error(self):
        with self.assertRaises(LLMError):
            LLMClient({"api_key": "offline-placeholder", "provider": "typo"})

    def test_anthropic_conversion_preserves_tool_pair_and_invalid_json_recovery(self):
        messages = [
            {"role": "assistant", "tool_calls": [{"id": "a", "function": {
                "name": "calculator", "arguments": "{broken"}}]},
            {"role": "tool", "tool_call_id": "a", "content": "Error: invalid expression"},
        ]
        _, converted = _to_anthropic_messages(messages)
        self.assertEqual(converted[0]["content"][0]["input"], {})
        self.assertEqual(converted[1]["content"][0]["tool_use_id"], "a")

    def test_anthropic_preserves_multiple_text_blocks(self):
        client = LLMClient.__new__(LLMClient)
        client.model = "offline-test-model"
        client._anthropic = Mock()
        client._anthropic.messages.create.return_value = SimpleNamespace(content=[
            SimpleNamespace(type="text", text="First."), SimpleNamespace(type="text", text="Second.")])
        self.assertEqual(client._chat_anthropic([], None, 100).content, "First.\nSecond.")

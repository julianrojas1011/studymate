import sys
import types
import unittest
from unittest.mock import MagicMock, patch

from src.tools import calculate, convert, run_tool, web_search


class CalculatorTests(unittest.TestCase):
    def test_supported_math(self):
        for expression, expected in [("sqrt(144)", "12"), ("2**10", "1024"),
                                     ("round(pi, 2)", "3.14"), ("-2 + 3 * 4", "10")]:
            with self.subTest(expression=expression):
                self.assertEqual(calculate(expression), expected)

    def test_bad_arguments_and_domains_return_errors(self):
        for expression in ("sqrt()", "sqrt(-1)", "exp(1000)", "1/0", "(-1)**0.5",
                           "log(1, 1)", "True", "1e309", None, 4, [], ""):
            with self.subTest(expression=expression):
                self.assertTrue(calculate(expression).startswith("Error:"))

    def test_rejects_code_execution_constructs(self):
        for expression in ("__import__('os')", "(1).__class__", "[1, 2]", "abs(x=1)"):
            with self.subTest(expression=expression):
                self.assertTrue(calculate(expression).startswith("Error:"))

    def test_work_is_bounded(self):
        for expression in ("9**999999999", "9**(9**9)", "1+" * 600 + "1", "+".join(["1"] * 100)):
            with self.subTest(expression=expression):
                self.assertTrue(calculate(expression).startswith("Error:"))


class ToolArgumentTests(unittest.TestCase):
    def test_invalid_json_shapes_do_not_crash_dispatch(self):
        for value in (None, [], 5, "query"):
            self.assertTrue(run_tool("web_search", value).startswith("Error:"))

    def test_invalid_search_arguments_are_rejected_before_network(self):
        for value in ("many", "2", 2.5, True, None, []):
            result = run_tool("web_search", {"query": "linear algebra", "max_results": value})
            self.assertIn("must be an integer", result)
        self.assertIn("non-empty string", web_search(3))

    def test_search_limit_is_capped_and_result_is_formatted(self):
        module = types.ModuleType("ddgs")
        ddgs = MagicMock()
        ddgs.text.return_value = [{"title": "Notes", "href": "https://example.org", "body": "Example"}]
        module.DDGS = MagicMock()
        module.DDGS.return_value.__enter__.return_value = ddgs
        with patch.dict(sys.modules, {"ddgs": module}):
            result = web_search(" algebra ", 100)
        ddgs.text.assert_called_once_with("algebra", max_results=8, backend="duckduckgo")
        self.assertIn("1. Notes\n   https://example.org\n   Example", result)

    def test_unit_conversion_and_invalid_families(self):
        self.assertEqual(convert(2, "h", "min"), "2 h = 120.0 min")
        self.assertEqual(convert(32, "F", "C"), "32 F = 0.0 C")
        for value, source, target in [(1, "m", "kg"), (1, "wat", "wat"),
                                      (float("nan"), "m", "km"), (True, "m", "km"),
                                      (10**400, "m", "km"), (1e308, "km", "mm"),
                                      (1, None, "m")]:
            self.assertTrue(convert(value, source, target).startswith("Error:"))


if __name__ == "__main__":
    unittest.main()

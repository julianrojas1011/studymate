"""Exercise real tool dispatch and memory persistence without an LLM or API key.

Run from the repository root: python -m examples.offline_tools
This is a deterministic component example, not a generated tutor conversation.
"""

from pathlib import Path
from tempfile import TemporaryDirectory

from src.memory import Memory
from src.tools import run_tool


def main() -> None:
    calls = [
        ("calculator", {"expression": "sqrt(144)"}),
        ("unit_converter", {"value": 2, "from_unit": "h", "to_unit": "min"}),
        ("calculator", {"expression": "sqrt()"}),
    ]
    print("Offline component example (no model calls)")
    with TemporaryDirectory() as directory:
        path = Path(directory) / "memory.json"
        memory = Memory(path)
        for name, arguments in calls:
            result = run_tool(name, arguments)
            print(f"{name} {arguments} -> {result}")
            memory.append({"role": "user", "content": f"{name}: {arguments}"})
            memory.append({"role": "assistant", "content": result})
        memory.save()
        restored = Memory(path)
        restored.load()
        print(f"Restored {len(restored.messages())} messages from the temporary file.")


if __name__ == "__main__":
    main()

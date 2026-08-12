"""Command-line demo for the read-only case-facts LangChain tool."""

from __future__ import annotations

import argparse

from langchain_core.tools import ToolException

from rag.tools import get_case_facts


def main() -> None:
    """Load and print known facts for one mock procurement case."""

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("case_id", help="Case ID such as EM-001")
    args = parser.parse_args()

    try:
        case = get_case_facts.invoke({"case_id": args.case_id})
    except ToolException as error:
        parser.exit(1, f"Error: {error}\n")

    print(case.model_dump_json(indent=2))


if __name__ == "__main__":
    main()

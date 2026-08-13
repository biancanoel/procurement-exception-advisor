"""Connect to the registry-listed USAspending MCP server and list its tools."""

from __future__ import annotations

import asyncio
import json
import sys

from integrations.usaspending_mcp import list_tools


CONTRACT_SEARCH_TOOLS = ("search_awards",)


async def discover_tools() -> None:
    """Launch the third-party server over stdio and print its tool catalog."""

    response = await list_tools()

    print(f"Discovered tools: {len(response.tools)}")
    for tool in response.tools:
        print(f"\nTool: {tool.name}")
        print(f"Description: {tool.description or 'No description provided'}")
        print("Input schema:")
        print(json.dumps(tool.input_schema, indent=2, sort_keys=True))

    print("\nLikely contract-award search tools:")
    discovered_names = {tool.name for tool in response.tools}
    for name in CONTRACT_SEARCH_TOOLS:
        if name in discovered_names:
            print(f"- {name}")


def main() -> int:
    """Run MCP tool discovery with a concise connection error."""

    try:
        asyncio.run(discover_tools())
    except (Exception, KeyboardInterrupt) as error:
        print(f"USAspending MCP connection failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

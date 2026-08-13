"""Small client helpers for the registry-listed USAspending MCP server."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


SERVER_PACKAGE = "usaspending-gov-mcp==0.3.2"


@asynccontextmanager
async def open_session() -> AsyncIterator[ClientSession]:
    """Launch and initialize the third-party MCP server over stdio."""

    server = StdioServerParameters(
        command="uvx",
        args=[
            "--with",
            "mcp<2",
            "--from",
            SERVER_PACKAGE,
            "usaspending-mcp",
        ],
    )
    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


async def list_tools() -> Any:
    """Return the tools exposed by the third-party MCP server."""

    async with open_session() as session:
        return await session.list_tools()


async def _call_search_awards(arguments: dict[str, Any]) -> dict[str, Any]:
    async with open_session() as session:
        result = await session.call_tool("search_awards", arguments)
    return result.model_dump(mode="json", by_alias=True)


def call_search_awards(arguments: dict[str, Any]) -> dict[str, Any]:
    """Call the third-party MCP search_awards tool once."""

    return asyncio.run(_call_search_awards(arguments))

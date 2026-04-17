"""
Base Tool interface and ToolRegistry for Nova Sonic function calling.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any


class Tool(ABC):
    """Abstract base class that every tool must implement."""

    name: str
    description: str
    input_schema: dict

    @abstractmethod
    async def execute(
        self,
        params: dict,
        inference_config: dict | None = None,
    ) -> Any:
        """Execute the tool and return a JSON-serialisable result."""
        ...

    def get_tool_spec(self) -> dict:
        """Return the tool spec in the format Nova Sonic expects."""
        return {
            "toolSpec": {
                "name": self.name,
                "description": self.description,
                "inputSchema": {
                    # Nova Sonic requires inputSchema.json as a JSON string
                    "json": json.dumps(self.input_schema),
                },
            }
        }


class ToolRegistry:
    """Registry that holds tools and dispatches execution by name."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name.lower()] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name.lower())

    def has(self, name: str) -> bool:
        return name.lower() in self._tools

    def all(self) -> list[Tool]:
        return list(self._tools.values())

    def get_tool_specs(self) -> list[dict]:
        """Return all tool specs formatted for a Nova Sonic promptStart event."""
        return [tool.get_tool_spec() for tool in self._tools.values()]

    async def execute(
        self,
        name: str,
        params: dict,
        inference_config: dict | None = None,
    ) -> Any:
        tool = self.get(name)
        if not tool:
            raise ValueError(f'Tool "{name}" not found in registry')
        return await tool.execute(params, inference_config)

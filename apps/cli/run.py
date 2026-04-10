"""Headless (non-interactive) runner for pydantic-deep.

Executes a single task without user interaction and returns the result.
Designed for benchmarks, CI/CD pipelines, and scripted automation.
"""

from __future__ import annotations

import json
import sys
from typing import Any

from pydantic_ai.usage import Usage

from apps.cli.agent import create_cli_agent


async def execute_headless(
    *,
    task: str,
    working_dir: str,
    model: str | None = None,
    output_json: bool = False,
    max_turns: int | None = None,
    timeout: int | None = None,
) -> int:
    """Execute a task in headless mode and print the result.

    Args:
        task: The task description to execute.
        working_dir: Filesystem root directory.
        model: Model override (default: from config).
        output_json: Whether to output result as JSON.
        max_turns: Maximum number of agent turns.
        timeout: Timeout in seconds.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    agent, deps = create_cli_agent(
        model=model,
        working_dir=working_dir,
        non_interactive=True,
        include_plan=False,
        include_memory=False,
    )

    run_kwargs: dict[str, Any] = {}
    if max_turns is not None:
        run_kwargs["max_turns"] = max_turns

    if timeout is not None:
        import asyncio

        try:
            result = await asyncio.wait_for(
                agent.run(task, deps=deps, **run_kwargs),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            _print_error("Timed out", output_json)
            return 1
    else:
        result = await agent.run(task, deps=deps, **run_kwargs)

    if output_json:
        output = _build_json_output(result.output, result.usage())
        print(json.dumps(output, indent=2, default=str))
    else:
        print(result.output)

    return 0


def _build_json_output(output: str, usage: Usage) -> dict[str, Any]:
    """Build a JSON-serializable output dict."""
    return {
        "output": output,
        "usage": {
            "total_tokens": usage.total_tokens,
            "request_tokens": usage.request_tokens,
            "response_tokens": usage.response_tokens,
            "requests": usage.requests,
        },
    }


def _print_error(message: str, output_json: bool) -> None:
    """Print an error message to stderr (or as JSON)."""
    if output_json:
        print(json.dumps({"error": message}), file=sys.stderr)
    else:
        print(f"Error: {message}", file=sys.stderr)

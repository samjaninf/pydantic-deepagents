"""Browser capability for pydantic-deep agents.

Provides a real async Playwright browser to the agent via ``BrowserCapability``.
The browser lifecycle (launch on run start, close on run end) is managed through
``wrap_run`` — guaranteeing cleanup even when the agent raises an exception.

Example::

    from pydantic_ai import Agent
    from pydantic_deep.capabilities.browser import BrowserCapability

    agent = Agent(
        "anthropic:claude-sonnet-4-6",
        capabilities=[BrowserCapability(headless=True)],
    )
    result = await agent.run("What is the title of https://example.com?")
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from pydantic_ai import AgentRunResult, RunContext
from pydantic_ai.capabilities import AbstractCapability, WrapRunHandler
from pydantic_ai.toolsets import AbstractToolset

from pydantic_deep.toolsets.browser import (
    DEFAULT_MAX_CONTENT_TOKENS,
    DEFAULT_TIMEOUT_MS,
    BrowserToolset,
    _BrowserState,
    _require_browser,
)

try:
    from playwright.async_api import async_playwright
except ImportError:  # pragma: no cover
    async_playwright = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

BROWSER_INSTRUCTIONS: str = """\
You have access to a real web browser powered by Playwright. Use browser tools to:

- `navigate(url)` — go to a URL and read the page as Markdown
- `click(selector)` — click a CSS selector or pixel coordinates 'x,y'
- `type_text(selector, text)` — fill an input field
- `screenshot(full_page?)` — capture a screenshot (base64 PNG)
- `get_text(selector?)` — extract text from the page or a specific element
- `scroll(direction)` — scroll 'up', 'down', 'left', or 'right'
- `go_back()` / `go_forward()` — navigate browser history
- `execute_js(script)` — run JavaScript and get the result

Page content is returned as Markdown and truncated to ~{max_content_tokens} tokens.
Use `get_text` with a CSS selector to extract specific sections of large pages.
The browser is single-tab; new-tab links are redirected to the current tab.
Allowed domains: {allowed_domains}
"""


@dataclass
class BrowserCapability(AbstractCapability[Any]):
    """Provides a real async Playwright browser to the agent.

    Manages the full browser lifecycle: Chromium is launched before the agent
    run starts (via ``wrap_run``) and closed in a ``finally`` block, guaranteeing
    cleanup on both success and failure paths.

    Requires the ``browser`` optional extra::

        pip install 'pydantic-deep[browser]'
        playwright install chromium

    Args:
        headless: Run the browser without a visible window (default ``True``).
        allowed_domains: Domain allowlist. ``None`` (default) allows all domains.
            Example: ``["docs.python.org", "github.com"]``.
        screenshot_on_navigate: Append a base64 screenshot to every ``navigate``
            response (default ``False``).
        max_content_tokens: Maximum estimated tokens for page content
            (default ``4000``).
        timeout_ms: Default Playwright navigation timeout in milliseconds
            (default ``30000``).

    Example::

        from pydantic_ai import Agent
        from pydantic_deep.capabilities.browser import BrowserCapability

        agent = Agent(
            "anthropic:claude-sonnet-4-6",
            capabilities=[
                BrowserCapability(
                    headless=True,
                    allowed_domains=["docs.python.org"],
                )
            ],
        )
        result = await agent.run("What's new in Python 3.13?")
    """

    headless: bool = True
    allowed_domains: list[str] | None = None
    screenshot_on_navigate: bool = False
    max_content_tokens: int = DEFAULT_MAX_CONTENT_TOKENS
    timeout_ms: int = DEFAULT_TIMEOUT_MS

    _state: _BrowserState = field(default_factory=_BrowserState, init=False, repr=False)
    _toolset: BrowserToolset | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._state = _BrowserState()
        self._toolset = BrowserToolset(
            state=self._state,
            allowed_domains=self.allowed_domains,
            screenshot_on_navigate=self.screenshot_on_navigate,
            max_content_tokens=self.max_content_tokens,
            timeout_ms=self.timeout_ms,
        )

    def get_toolset(self) -> AbstractToolset[Any] | None:
        return self._toolset

    def get_instructions(self) -> str:
        return BROWSER_INSTRUCTIONS.format(
            max_content_tokens=self.max_content_tokens,
            allowed_domains=", ".join(self.allowed_domains) if self.allowed_domains else "all",
        )

    async def wrap_run(
        self,
        ctx: RunContext[Any],
        *,
        handler: WrapRunHandler,
    ) -> AgentRunResult[Any]:
        """Launch Chromium before the run and guarantee cleanup afterwards.

        The browser is started inside ``async with async_playwright()`` and the
        active ``Page`` is injected into ``self._state.page`` so that tool calls
        can reach it.  A ``finally`` block nulls the state references and closes
        the browser whether the run succeeds, raises an exception, or is
        cancelled.
        """
        _require_browser()
        assert async_playwright is not None  # guaranteed by _require_browser()
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=self.headless)
            page = await browser.new_page()

            # Single-tab design: redirect popup windows back to the current tab.
            def _on_popup(popup: Any) -> None:
                new_url = popup.url
                asyncio.ensure_future(popup.close())
                asyncio.ensure_future(page.goto(new_url))

            page.on("popup", _on_popup)

            self._state.playwright_instance = pw
            self._state.browser = browser
            self._state.page = page
            try:
                return await handler()
            finally:
                self._state.page = None
                self._state.browser = None
                self._state.playwright_instance = None
                await browser.close()

"""
pipeline/llm_client.py
─────────────────────────
Provider-agnostic wrapper around whichever LLM backend is configured.

Every generator module (flashcard_gen, highlight_tagger, test_builder,
formula_sheet_gen, notes_gen, hot_questions_gen) calls LLMClient().complete()
instead of talking to a specific SDK directly. Switching providers is a
one-line config change (LLM_PROVIDER in .env) — no code changes needed.

Usage:
    client = LLMClient()
    text = client.complete("Write a haiku about photosynthesis.", max_tokens=200)
"""

import re
import time

from rich.console import Console

from config import (
    ANTHROPIC_API_KEY,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    LLM_MODEL,
    LLM_PROVIDER,
)

console = Console()

MAX_RETRIES          = 5
DEFAULT_RETRY_DELAY_S = 15  # used when the server doesn't tell us how long to wait


class LLMClient:

    def __init__(self):
        self.provider = LLM_PROVIDER

        if self.provider == "gemini":
            if not GEMINI_API_KEY:
                raise RuntimeError(
                    "LLM_PROVIDER is 'gemini' but GEMINI_API_KEY is not set. "
                    "Get a free key at https://aistudio.google.com/apikey and "
                    "add it to your .env file."
                )
            from google import genai
            self._client = genai.Client(api_key=GEMINI_API_KEY)

        elif self.provider == "anthropic":
            if not ANTHROPIC_API_KEY:
                raise RuntimeError(
                    "LLM_PROVIDER is 'anthropic' but ANTHROPIC_API_KEY is not set. "
                    "Add it to your .env file."
                )
            import anthropic
            self._client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

        else:
            raise ValueError(
                f"Unknown LLM_PROVIDER: {self.provider!r}. Use 'gemini' or 'anthropic'."
            )

    def complete(self, prompt: str, max_tokens: int = 2048) -> str:
        """
        Sends a single-turn prompt to the configured LLM and returns the
        raw text response. Retries on rate-limit (429) errors, honoring the
        server's suggested wait time when it provides one — the Gemini free
        tier's 5-requests-per-minute cap means this is routine, not
        exceptional, especially when multiple generators run concurrently.
        """
        if self.provider == "gemini":
            return self._complete_gemini(prompt, max_tokens)
        return self._complete_anthropic(prompt, max_tokens)

    # ──────────────────────────────────────────────────────────────────────────
    # Internal Helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _complete_gemini(self, prompt: str, max_tokens: int) -> str:
        from google.genai import errors, types

        config = types.GenerateContentConfig(
            max_output_tokens=max_tokens,
            # Gemini 2.5's "thinking" tokens count against max_output_tokens —
            # for these structured JSON-extraction prompts we don't need
            # reasoning, and leaving thinking on silently eats the budget and
            # truncates the real answer.
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self._client.models.generate_content(
                    model=GEMINI_MODEL, contents=prompt, config=config,
                )
                return response.text or ""
            except errors.ClientError as e:
                if e.code != 429 or attempt == MAX_RETRIES:
                    raise
                wait = self._extract_retry_delay(e) or DEFAULT_RETRY_DELAY_S
                console.print(
                    f"[yellow]⚠ Gemini rate limit hit (attempt {attempt}/{MAX_RETRIES}):[/yellow] "
                    f"waiting {wait}s before retrying (free tier is 5 requests/minute)..."
                )
                time.sleep(wait)

        raise RuntimeError("unreachable")  # loop always returns or raises

    def _extract_retry_delay(self, error) -> "int | None":
        """Pulls the server-suggested retryDelay (e.g. '8s') out of a 429's
        error details, if present. Returns None if it can't find one."""
        try:
            details = error.details.get("error", {}).get("details", [])
            for entry in details:
                if entry.get("@type", "").endswith("RetryInfo"):
                    match = re.match(r"(\d+(?:\.\d+)?)s?", entry.get("retryDelay", ""))
                    if match:
                        # Round up and pad by 1s so we don't retry a hair too early
                        return int(float(match.group(1))) + 1
        except Exception:
            pass
        return None

    def _complete_anthropic(self, prompt: str, max_tokens: int) -> str:
        message = self._client.messages.create(
            model=LLM_MODEL,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text

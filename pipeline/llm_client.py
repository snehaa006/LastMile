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

from config import (
    ANTHROPIC_API_KEY,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    LLM_MODEL,
    LLM_PROVIDER,
)


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
        raw text response.
        """
        if self.provider == "gemini":
            from google.genai import types
            response = self._client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=max_tokens,
                    # Gemini 2.5's "thinking" tokens count against
                    # max_output_tokens — for these structured JSON-extraction
                    # prompts we don't need reasoning, and leaving thinking on
                    # silently eats the budget and truncates the real answer.
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )
            return response.text or ""

        message = self._client.messages.create(
            model=LLM_MODEL,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text

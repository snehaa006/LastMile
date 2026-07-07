"""
pipeline/formula_sheet_gen.py
───────────────────────────────
Generates a consolidated, exam-ready formula sheet for a chapter (maths and
formula-heavy science subjects). Retrieves formula-dense chunks from the
vector store and asks Claude to extract and organize every formula.

Usage:
    gen = FormulaSheetGenerator()
    sheet = gen.generate(
        collection_name="class10_mathematics_ch06",
        subject="mathematics",
        chapter=6,
        class_num=10,
    )
"""

import json
import re
from dataclasses import dataclass, field
from typing import Optional

from rich.console import Console

from pipeline.llm_client import LLMClient
from pipeline.vector_store import VectorStore

console = Console()

FORMULA_SEED_QUERIES = [
    "formula",
    "equation",
    "theorem and proof",
    "derivation",
    "unit and standard values",
]


@dataclass
class FormulaEntry:
    """A single formula with usage context."""
    name:        str
    expression:  str
    variables:   str   # short description of variables/units
    when_to_use: str
    topic:       str


@dataclass
class FormulaSheet:
    """Complete formula sheet for a chapter."""
    subject:   str
    chapter:   int
    class_num: int
    entries:   list[FormulaEntry] = field(default_factory=list)

    def summary(self) -> str:
        return f"Formula sheet: {len(self.entries)} formulas — {self.subject.title()} Ch.{self.chapter}"


class FormulaSheetGenerator:

    def __init__(self, store: Optional[VectorStore] = None):
        self._client = LLMClient()
        self._store  = store or VectorStore()

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def generate(
        self,
        collection_name: str,
        subject: str,
        chapter: int,
        class_num: int,
    ) -> FormulaSheet:
        """
        Builds a formula sheet from the indexed chapter.
        Returns an empty sheet if no formula-relevant content is found
        (e.g. purely descriptive chapters).
        """
        console.print(f"[blue]Σ Generating formula sheet:[/blue] {collection_name}")

        chunks = self._retrieve_formula_chunks(collection_name)
        if not chunks:
            console.print("[yellow]⚠ No formula-relevant chunks found.[/yellow]")
            return FormulaSheet(subject=subject, chapter=chapter, class_num=class_num)

        context = "\n\n---\n\n".join(chunks)
        raw = self._call_llm(context, subject, chapter, class_num)
        entries = self._parse(raw)

        console.print(f"[green]✓ Extracted:[/green] {len(entries)} formulas")
        return FormulaSheet(
            subject=subject, chapter=chapter, class_num=class_num, entries=entries
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Internal Helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _retrieve_formula_chunks(self, collection_name: str) -> list[str]:
        seen, chunks = set(), []
        for query in FORMULA_SEED_QUERIES:
            for r in self._store.query(query, collection_name, top_k=4):
                if r.chunk_id not in seen:
                    seen.add(r.chunk_id)
                    chunks.append(r.text)
        return chunks

    def _call_llm(self, context: str, subject: str, chapter: int, class_num: int) -> str:
        prompt = f"""You are an expert CBSE {subject.title()} teacher for Class {class_num}.

Extract every formula, equation, or theorem statement from the NCERT content
below into a clean, exam-ready formula sheet.

Rules:
- Only include actual formulas/equations/theorem statements — not general prose
- Give each formula a short name, its expression, a one-line note on variables/units,
  and one line on when/why it's used
- Do not invent formulas that are not present in or directly implied by the content
- If the chapter has no formulas (e.g. a purely descriptive topic), return an empty array

Return ONLY a JSON array. No explanation, no markdown fences.

Format:
[
  {{
    "name": "...",
    "expression": "...",
    "variables": "...",
    "when_to_use": "...",
    "topic": "..."
  }}
]

NCERT Content:
{context}
"""
        return self._client.complete(prompt, max_tokens=4096)

    def _parse(self, raw: str) -> list[FormulaEntry]:
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        try:
            items = json.loads(clean)
        except json.JSONDecodeError as e:
            console.print(f"[red]⚠ JSON parse error:[/red] {e}")
            console.print(f"[dim]{raw[:300]}...[/dim]")
            return []

        return [
            FormulaEntry(
                name        = i.get("name", ""),
                expression  = i.get("expression", ""),
                variables   = i.get("variables", ""),
                when_to_use = i.get("when_to_use", ""),
                topic       = i.get("topic", ""),
            )
            for i in items
        ]

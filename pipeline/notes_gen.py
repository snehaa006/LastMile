"""
pipeline/notes_gen.py
────────────────────────
Generates structured, exam-ready revision notes for a chapter — each
section carries a one-line "big idea" plus typed content blocks
(definitions, formulas, worked examples, comparison tables, bullet
facts) rather than a flat list of bullets. Built via RAG over the
indexed chapter content.

Usage:
    gen = NotesGenerator()
    notes = gen.generate(
        collection_name="class10_science_ch06",
        subject="science",
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

NOTES_SEED_QUERIES = [
    "definition and key concepts",
    "important processes and mechanisms",
    "formulas and equations",
    "examples and applications",
    "differences and comparisons",
    "summary and conclusion",
]

# Block types the LLM is allowed to emit, and what each one means:
#   definition — a term + its meaning
#   formula    — a named formula/equation + what it means
#   example    — a worked example (numeric problem, decay equation, etc.)
#   table      — a comparison/summary table
#   bullets    — plain facts that don't fit the above
BLOCK_TYPES = {"definition", "formula", "example", "table", "bullets"}


@dataclass
class NoteBlock:
    """One typed content block within a note section."""
    type:          str                              # one of BLOCK_TYPES
    title:         str = ""                          # term / formula name / example title / table caption
    content:       str = ""                          # definition text / formula expression / example body
    items:         list[str] = field(default_factory=list)        # for type="bullets"
    table_headers: list[str] = field(default_factory=list)        # for type="table"
    table_rows:    list[list[str]] = field(default_factory=list)  # for type="table"


@dataclass
class NoteSection:
    """A topic section: an overarching one-liner plus structured blocks."""
    heading:  str
    big_idea: str = ""
    blocks:   list[NoteBlock] = field(default_factory=list)


@dataclass
class ChapterNotes:
    """Complete structured revision notes for a chapter."""
    subject:   str
    chapter:   int
    class_num: int
    tldr:      str = ""
    sections:  list[NoteSection] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"Notes: {len(self.sections)} sections — {self.subject.title()} Ch.{self.chapter}"
        )


class NotesGenerator:

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
    ) -> ChapterNotes:
        """Builds structured revision notes from the indexed chapter."""
        console.print(f"[blue]📓 Generating notes:[/blue] {collection_name}")

        chunks = self._retrieve_diverse_chunks(collection_name)
        if not chunks:
            raise RuntimeError(f"No chunks found in collection '{collection_name}'.")

        context = "\n\n---\n\n".join(chunks)
        raw = self._call_llm(context, subject, chapter, class_num)
        notes = self._parse(raw, subject, chapter, class_num)

        console.print(f"[green]✓ Notes generated:[/green] {len(notes.sections)} sections")
        return notes

    # ──────────────────────────────────────────────────────────────────────────
    # Internal Helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _retrieve_diverse_chunks(self, collection_name: str) -> list[str]:
        seen, chunks = set(), []
        for query in NOTES_SEED_QUERIES:
            for r in self._store.query(query, collection_name, top_k=4):
                if r.chunk_id not in seen:
                    seen.add(r.chunk_id)
                    chunks.append(r.text)
        return chunks

    def _call_llm(self, context: str, subject: str, chapter: int, class_num: int) -> str:
        prompt = f"""You are an expert CBSE {subject.title()} teacher for Class {class_num}, writing a
structured revision guide a student can review the night before an exam —
similar in spirit to a well-organized study site, not a wall of bullet points.

Rules:
- Start with a 2-3 sentence "tldr" of the whole chapter
- Break the chapter into short topic sections (e.g. "Isotopes and Mass Number",
  "Types of Radioactive Decay") — 3-8 sections depending on chapter length
- Each section gets a one-sentence "big_idea": the single takeaway a student
  must remember for that topic
- Each section's content is a list of typed "blocks". Use whichever block
  types actually fit the content — don't force every type into every section:
  - "definition": a key term and its precise meaning. title = the term.
  - "formula": a named formula or equation, with its expression in the
    "content" field written in plain text (e.g. "Mass Number = Atomic Number
    + Number of Neutrons"), NOT LaTeX. Include EVERY formula/equation that
    appears in this section's source content — do not skip any.
  - "example": a worked example — a numeric problem, a balanced equation, a
    solved case — with the working shown in "content". title = short label.
  - "table": a comparison or summary table. table_headers = column names,
    table_rows = list of rows, each row a list of strings matching the
    headers. Use tables for comparisons (e.g. alpha vs beta vs gamma) and
    for multi-row summaries.
  - "bullets": plain facts that don't fit the above, in "items" (list of
    short strings, not full sentences copied from the text).
- Do not add information that isn't in the content below
- Do not use LaTeX or markdown formatting inside any field — plain text only

Return ONLY a JSON object. No explanation, no markdown fences.

Format:
{{
  "tldr": "...",
  "sections": [
    {{
      "heading": "...",
      "big_idea": "...",
      "blocks": [
        {{"type": "definition", "title": "...", "content": "..."}},
        {{"type": "formula", "title": "...", "content": "..."}},
        {{"type": "example", "title": "...", "content": "..."}},
        {{"type": "table", "title": "...", "table_headers": ["...", "..."], "table_rows": [["...", "..."]]}},
        {{"type": "bullets", "items": ["...", "..."]}}
      ]
    }}
  ]
}}

NCERT Content:
{context}
"""
        return self._client.complete(prompt, max_tokens=8192)

    def _parse(self, raw: str, subject: str, chapter: int, class_num: int) -> ChapterNotes:
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        try:
            data = json.loads(clean)
        except json.JSONDecodeError as e:
            console.print(f"[red]⚠ JSON parse error:[/red] {e}")
            console.print(f"[dim]{raw[:300]}...[/dim]")
            return ChapterNotes(subject=subject, chapter=chapter, class_num=class_num)

        sections = []
        for s in data.get("sections", []):
            blocks = []
            for b in s.get("blocks", []):
                block_type = b.get("type", "bullets")
                if block_type not in BLOCK_TYPES:
                    block_type = "bullets"
                blocks.append(
                    NoteBlock(
                        type          = block_type,
                        title         = b.get("title", ""),
                        content       = b.get("content", ""),
                        items         = b.get("items", []),
                        table_headers = b.get("table_headers", []),
                        table_rows    = b.get("table_rows", []),
                    )
                )
            sections.append(
                NoteSection(
                    heading  = s.get("heading", ""),
                    big_idea = s.get("big_idea", ""),
                    blocks   = blocks,
                )
            )

        return ChapterNotes(
            subject   = subject,
            chapter   = chapter,
            class_num = class_num,
            tldr      = data.get("tldr", ""),
            sections  = sections,
        )

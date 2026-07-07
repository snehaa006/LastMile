"""
pipeline/notes_gen.py
────────────────────────
Generates condensed, exam-ready revision notes for a chapter — a structured
set of headings with bullet points plus a one-paragraph TL;DR — built via
RAG over the indexed chapter content.

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

import anthropic
from rich.console import Console

from config import ANTHROPIC_API_KEY, LLM_MODEL
from pipeline.vector_store import VectorStore

console = Console()

NOTES_SEED_QUERIES = [
    "definition and key concepts",
    "important processes and mechanisms",
    "examples and applications",
    "differences and comparisons",
    "summary and conclusion",
]


@dataclass
class NoteSection:
    """A single heading with condensed bullet points."""
    heading: str
    bullets: list[str] = field(default_factory=list)


@dataclass
class ChapterNotes:
    """Complete condensed revision notes for a chapter."""
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

    def __init__(self):
        self._client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        self._store  = VectorStore()

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
        """Builds condensed revision notes from the indexed chapter."""
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
        prompt = f"""You are an expert CBSE {subject.title()} teacher for Class {class_num}.

Condense the NCERT content below into revision notes a student can review
the night before an exam.

Rules:
- Start with a 2-3 sentence TL;DR of the whole chapter
- Organize the rest under short topic headings (e.g. "Photosynthesis", "Types of Respiration")
- Under each heading, give 3-6 tight bullet points — facts, definitions, steps, or
  comparisons a student must remember, not full sentences copied from the text
- Do not add information that isn't in the content

Return ONLY a JSON object. No explanation, no markdown fences.

Format:
{{
  "tldr": "...",
  "sections": [
    {{
      "heading": "...",
      "bullets": ["...", "..."]
    }}
  ]
}}

NCERT Content:
{context}
"""
        message = self._client.messages.create(
            model      = LLM_MODEL,
            max_tokens = 4096,
            messages   = [{"role": "user", "content": prompt}],
        )
        return message.content[0].text

    def _parse(self, raw: str, subject: str, chapter: int, class_num: int) -> ChapterNotes:
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        try:
            data = json.loads(clean)
        except json.JSONDecodeError as e:
            console.print(f"[red]⚠ JSON parse error:[/red] {e}")
            console.print(f"[dim]{raw[:300]}...[/dim]")
            return ChapterNotes(subject=subject, chapter=chapter, class_num=class_num)

        sections = [
            NoteSection(heading=s.get("heading", ""), bullets=s.get("bullets", []))
            for s in data.get("sections", [])
        ]
        return ChapterNotes(
            subject   = subject,
            chapter   = chapter,
            class_num = class_num,
            tldr      = data.get("tldr", ""),
            sections  = sections,
        )

"""
pipeline/highlight_tagger.py
─────────────────────────────
Classifies each paragraph of NCERT content by exam importance.

Uses Claude to label each chunk as HIGH / MEDIUM / LOW importance,
enabling the e-NCERT reader to visually highlight key content.

Usage:
    tagger = HighlightTagger()
    tagged = tagger.tag(chunks)   # List[TaggedChunk]
"""

import json
import re
from dataclasses import dataclass
from typing import Literal

from rich.console import Console

from pipeline.llm_client import LLMClient
from pipeline.pdf_parser import Chunk

console = Console()

ImportanceLevel = Literal["HIGH", "MEDIUM", "LOW"]

# Process in batches to stay within token limits
BATCH_SIZE = 8


@dataclass
class TaggedChunk:
    """A text chunk annotated with exam importance and key terms."""
    chunk_id:   str
    text:       str
    importance: ImportanceLevel
    key_terms:  list[str]   # terms worth a popup definition
    reason:     str         # brief explanation of why this is important
    page:       int
    chapter:    int
    subject:    str


class HighlightTagger:

    def __init__(self):
        self._client = LLMClient()

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def tag(self, chunks: list[Chunk]) -> list[TaggedChunk]:
        """
        Tags all chunks with importance levels.
        Processes in batches to minimize API calls.

        Returns:
            List[TaggedChunk] in the same order as input chunks.
        """
        if not chunks:
            return []

        subject   = chunks[0].subject
        class_num = chunks[0].class_num
        chapter   = chunks[0].chapter

        console.print(
            f"[blue]🔍 Tagging {len(chunks)} chunks:[/blue] "
            f"Class {class_num} {subject} Ch.{chapter}"
        )

        tagged = []
        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i : i + BATCH_SIZE]
            batch_results = self._tag_batch(batch, subject, class_num)
            tagged.extend(batch_results)

        high_count = sum(1 for t in tagged if t.importance == "HIGH")
        console.print(
            f"[green]✓ Tagged:[/green] {high_count}/{len(tagged)} chunks marked HIGH importance"
        )
        return tagged

    def get_key_terms(self, tagged_chunks: list[TaggedChunk]) -> list[str]:
        """Returns deduplicated list of all key terms across all tagged chunks."""
        terms = []
        for chunk in tagged_chunks:
            terms.extend(chunk.key_terms)
        return list(dict.fromkeys(terms))

    def filter_by_importance(
        self,
        tagged: list[TaggedChunk],
        level: ImportanceLevel,
    ) -> list[TaggedChunk]:
        """Filters tagged chunks to a specific importance level."""
        return [t for t in tagged if t.importance == level]

    # ──────────────────────────────────────────────────────────────────────────
    # Internal Helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _tag_batch(
        self,
        chunks: list[Chunk],
        subject: str,
        class_num: int,
    ) -> list[TaggedChunk]:
        """Sends a batch of chunks to Claude for importance classification."""
        chunks_json = json.dumps(
            [{"id": c.chunk_id, "text": c.text} for c in chunks],
            indent=2,
        )

        prompt = f"""You are a CBSE expert for Class {class_num} {subject.title()}.

Classify each text chunk by exam importance for CBSE board exams.

Criteria:
- HIGH   → definitions, key processes, formulas, comparison tables, frequently asked topics
- MEDIUM → supporting explanations, examples that illustrate HIGH content
- LOW    → general intro text, historical background, non-exam context

Also extract:
- key_terms: technical/scientific terms that deserve a popup definition (max 5 per chunk)
- reason: one sentence explaining the importance classification

Return ONLY a JSON array. No markdown fences.

Format:
[
  {{
    "id": "...",
    "importance": "HIGH|MEDIUM|LOW",
    "key_terms": ["term1", "term2"],
    "reason": "..."
  }}
]

Chunks:
{chunks_json}
"""

        raw = self._client.complete(prompt, max_tokens=2048)

        # Parse response
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        try:
            results = json.loads(clean)
        except json.JSONDecodeError:
            console.print("[yellow]⚠ Batch parse failed, defaulting to MEDIUM[/yellow]")
            results = [
                {"id": c.chunk_id, "importance": "MEDIUM", "key_terms": [], "reason": ""}
                for c in chunks
            ]

        # Map results back to chunks
        result_map = {r["id"]: r for r in results}
        tagged = []
        for chunk in chunks:
            r = result_map.get(chunk.chunk_id, {})
            tagged.append(
                TaggedChunk(
                    chunk_id   = chunk.chunk_id,
                    text       = chunk.text,
                    importance = r.get("importance", "MEDIUM"),
                    key_terms  = r.get("key_terms", []),
                    reason     = r.get("reason", ""),
                    page       = chunk.page,
                    chapter    = chunk.chapter,
                    subject    = chunk.subject,
                )
            )
        return tagged

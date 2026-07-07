"""
pipeline/hot_questions_gen.py
────────────────────────────────
Generates a ranked list of "hot questions" — the questions most likely to
appear in the exam for a chapter. Combines two signals:

  1. PYQ repetition — previous-year questions already on record for this
     subject/class, which show a topic has actually been examined before
  2. LLM prediction   — Claude flags additional high-yield questions from
     the chapter content that aren't yet covered by any stored PYQ

Usage:
    gen = HotQuestionsGenerator()
    hot = gen.generate(
        collection_name="class10_science_ch06",
        subject="science",
        chapter=6,
        class_num=10,
    )
"""

import json
import re
from dataclasses import dataclass, field
from typing import Literal, Optional

from rich.console import Console

from pipeline.llm_client import LLMClient
from pipeline.vector_store import VectorStore

console = Console()

Likelihood = Literal["very high", "high", "medium"]
HotSource  = Literal["pyq_repeat", "ai_predicted"]

CONTEXT_SEED_QUERIES = [
    "important",
    "definition and key concepts",
    "numerical problem",
    "diagram and structure",
    "compare and differentiate",
]


@dataclass
class HotQuestion:
    """A single high-probability exam question."""
    question:   str
    answer:     str
    topic:      str
    marks:      int
    likelihood: Likelihood
    basis:      str          # e.g. "asked in 2019, 2022" or "high-yield unasked concept"
    source:     HotSource


class HotQuestionsGenerator:

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
        n: int = 10,
    ) -> list[HotQuestion]:
        """
        Builds a ranked hot-questions list for the chapter.
        The chapter must already be indexed (process_chapter() run first).
        """
        console.print(f"[blue]🔥 Generating hot questions:[/blue] {collection_name}")

        pyqs = self._get_recorded_pyqs(subject, class_num)
        chunks = self._retrieve_key_chunks(collection_name)

        raw = self._call_llm(pyqs, chunks, subject, chapter, class_num, n)
        questions = self._parse(raw)

        console.print(f"[green]✓ Hot questions:[/green] {len(questions)}")
        return questions

    # ──────────────────────────────────────────────────────────────────────────
    # Internal Helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _get_recorded_pyqs(self, subject: str, class_num: int) -> list[dict]:
        """Pulls PYQs on record for this subject/class as a repetition signal."""
        if not self._store.collection_exists("pyq_bank"):
            return []
        return self._store.query_pyqs(subject, subject, class_num, top_k=20)

    def _retrieve_key_chunks(self, collection_name: str) -> list[str]:
        seen, chunks = set(), []
        for query in CONTEXT_SEED_QUERIES:
            for r in self._store.query(query, collection_name, top_k=4):
                if r.chunk_id not in seen:
                    seen.add(r.chunk_id)
                    chunks.append(r.text)
        return chunks

    def _call_llm(
        self,
        pyqs: list[dict],
        chunks: list[str],
        subject: str,
        chapter: int,
        class_num: int,
        n: int,
    ) -> str:
        pyq_block = "\n".join(
            f"- ({p.get('year', '?')}, {p.get('marks', '?')}m) {p.get('question', '')}"
            for p in pyqs
        ) or "None on record."
        context = "\n\n---\n\n".join(chunks)[:4000]

        prompt = f"""You are an expert CBSE {subject.title()} teacher for Class {class_num}, \
analyzing Chapter {chapter} to predict what's most likely to appear in the board exam.

Previous year questions on record for this subject (repetition signal):
{pyq_block}

NCERT Content for this chapter:
{context}

Task: Produce the top {n} "hot questions" for this chapter, ranked by likelihood.

Rules:
- If a recorded PYQ topic clearly matches this chapter's content, include it or a close
  variant, mark source "pyq_repeat", and set basis to the years it was asked
- Fill remaining slots with new high-yield questions predicted from the chapter content
  that aren't yet covered by any recorded PYQ, mark source "ai_predicted", and set basis
  to a short reason (e.g. "core definition, always tested")
- likelihood must be "very high", "high", or "medium"
- Keep answers concise (2-4 sentences)

Return ONLY a JSON array. No explanation, no markdown fences.

Format:
[
  {{
    "question": "...",
    "answer": "...",
    "topic": "...",
    "marks": 3,
    "likelihood": "very high|high|medium",
    "basis": "...",
    "source": "pyq_repeat|ai_predicted"
  }}
]
"""
        return self._client.complete(prompt, max_tokens=3072)

    def _parse(self, raw: str) -> list[HotQuestion]:
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        try:
            items = json.loads(clean)
        except json.JSONDecodeError as e:
            console.print(f"[red]⚠ JSON parse error:[/red] {e}")
            console.print(f"[dim]{raw[:300]}...[/dim]")
            return []

        return [
            HotQuestion(
                question   = i.get("question", ""),
                answer     = i.get("answer", ""),
                topic      = i.get("topic", ""),
                marks      = i.get("marks", 1),
                likelihood = i.get("likelihood", "medium"),
                basis      = i.get("basis", ""),
                source     = i.get("source", "ai_predicted"),
            )
            for i in items
        ]

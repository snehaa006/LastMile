"""
pipeline/flashcard_gen.py
──────────────────────────
Generates exam-focused flashcards from NCERT content using RAG.

Flow:
  1. Queries the vector store for concept-dense chunks from the chapter
  2. Sends chunks to Claude with a structured prompt
  3. Parses the response into Flashcard objects

Usage:
    gen = FlashcardGenerator()
    cards = gen.generate(
        collection_name="class10_science_ch03",
        subject="science",
        chapter=3,
        class_num=10,
    )
"""

import json
import re
from dataclasses import dataclass, field
from typing import Literal, Optional

from rich.console import Console

from config import FLASHCARDS_PER_CHAPTER, TOP_K_CHUNKS
from pipeline.llm_client import LLMClient
from pipeline.vector_store import VectorStore

console = Console()

DifficultyLevel = Literal["easy", "medium", "hard"]

SEED_QUERIES = [
    "definition and key concepts",
    "important processes and mechanisms",
    "examples and applications",
    "differences and comparisons",
    "causes and effects",
    "formulas and numerical problems",
    "diagrams and structures",
]


@dataclass
class Flashcard:
    """A single flashcard with question, answer, and metadata."""
    card_id:    str
    question:   str
    answer:     str
    topic:      str
    difficulty: DifficultyLevel
    source:     Literal["ncert", "pyq"]
    subject:    str
    chapter:    int
    class_num:  int
    tags:       list[str] = field(default_factory=list)
    starred:    bool = False


class FlashcardGenerator:

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
        n: int = FLASHCARDS_PER_CHAPTER,
    ) -> list[Flashcard]:
        """
        Generates n flashcards from the indexed chapter.

        Args:
            collection_name : ChromaDB collection e.g. "class10_science_ch03"
            subject         : e.g. "science"
            chapter         : chapter number
            class_num       : 6–12
            n               : number of flashcards to generate

        Returns:
            List[Flashcard]
        """
        console.print(f"[blue]🃏 Generating flashcards:[/blue] {collection_name}")

        # 1. Retrieve diverse chunks using multiple seed queries
        chunks = self._retrieve_diverse_chunks(collection_name)
        if not chunks:
            raise RuntimeError(f"No chunks found in collection '{collection_name}'.")

        context = self._build_context(chunks)

        # 2. Call LLM
        raw = self._call_llm(context, subject, chapter, class_num, n)

        # 3. Parse into Flashcard objects
        cards = self._parse_flashcards(raw, subject, chapter, class_num)
        console.print(f"[green]✓ Generated:[/green] {len(cards)} flashcards")
        return cards

    def generate_from_pyqs(
        self,
        pyqs: list[dict],
        subject: str,
        class_num: int,
    ) -> list[Flashcard]:
        """
        Wraps PYQ dicts into Flashcard objects (no LLM needed).
        PYQs are already question-answer pairs.
        """
        cards = []
        for i, pyq in enumerate(pyqs):
            cards.append(
                Flashcard(
                    card_id    = f"pyq_{subject}_{pyq.get('year', 0)}_{i}",
                    question   = pyq["question"],
                    answer     = pyq.get("answer", "See solution."),
                    topic      = pyq.get("chapter", ""),
                    difficulty = self._marks_to_difficulty(pyq.get("marks", 1)),
                    source     = "pyq",
                    subject    = subject,
                    chapter    = int(pyq.get("chapter", 0)),
                    class_num  = class_num,
                    tags       = [str(pyq.get("year", "")), f"{pyq.get('marks', 1)} marks"],
                )
            )
        return cards

    # ──────────────────────────────────────────────────────────────────────────
    # Internal Helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _retrieve_diverse_chunks(self, collection_name: str) -> list[str]:
        """
        Uses multiple seed queries to retrieve topically diverse chunks,
        then deduplicates by chunk_id.
        """
        seen, chunks = set(), []
        for query in SEED_QUERIES:
            results = self._store.query(query, collection_name, top_k=3)
            for r in results:
                if r.chunk_id not in seen:
                    seen.add(r.chunk_id)
                    chunks.append(r.text)
        return chunks

    def _build_context(self, chunks: list[str]) -> str:
        return "\n\n---\n\n".join(chunks)

    def _call_llm(
        self,
        context: str,
        subject: str,
        chapter: int,
        class_num: int,
        n: int,
    ) -> str:
        prompt = f"""You are an expert CBSE teacher for Class {class_num} {subject.title()}.

Below is content from Chapter {chapter} of the NCERT textbook.

Your task: Generate exactly {n} high-quality flashcards for CBSE exam revision.

Rules:
- Cover definitions, processes, formulas, comparisons, and application questions
- Mix difficulty: ~40% easy, 40% medium, 20% hard
- Answers must be concise (2–4 sentences max)
- Focus on what commonly appears in CBSE board exams
- Each flashcard must have a clear topic tag

Return ONLY a JSON array. No explanation, no markdown fences.

Format:
[
  {{
    "question": "...",
    "answer": "...",
    "topic": "...",
    "difficulty": "easy|medium|hard",
    "tags": ["tag1", "tag2"]
  }}
]

NCERT Content:
{context}
"""

        return self._client.complete(prompt, max_tokens=4096)

    def _parse_flashcards(
        self,
        raw: str,
        subject: str,
        chapter: int,
        class_num: int,
    ) -> list[Flashcard]:
        """Parses LLM JSON output into Flashcard objects."""
        # Strip any accidental markdown fences
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        try:
            items = json.loads(clean)
        except json.JSONDecodeError as e:
            console.print(f"[red]⚠ JSON parse error:[/red] {e}")
            console.print(f"[dim]{raw[:300]}...[/dim]")
            return []

        cards = []
        for i, item in enumerate(items):
            cards.append(
                Flashcard(
                    card_id    = f"ncert_{subject}_ch{str(chapter).zfill(2)}_{i}",
                    question   = item.get("question", ""),
                    answer     = item.get("answer", ""),
                    topic      = item.get("topic", subject),
                    difficulty = item.get("difficulty", "medium"),
                    source     = "ncert",
                    subject    = subject,
                    chapter    = chapter,
                    class_num  = class_num,
                    tags       = item.get("tags", []),
                )
            )
        return cards

    def _marks_to_difficulty(self, marks: int) -> DifficultyLevel:
        if marks <= 1:
            return "easy"
        elif marks <= 3:
            return "medium"
        return "hard"

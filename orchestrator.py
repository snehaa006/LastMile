"""
agents/orchestrator.py
───────────────────────
The EduMind orchestrator — ties all pipeline components into a single
callable interface.

This is what your backend API or CLI calls. It decides which tools to
run and in what order based on the request type.

Usage:
    agent = EduMindAgent()

    # Full pipeline run (fetch → parse → embed → flashcards → highlights)
    result = agent.process_chapter(class_num=10, subject="science", chapter=3)

    # Generate a personalized test for a student
    test = agent.generate_test(student_id="stu_001", ...)

    # Just get flashcards from an already-indexed chapter
    cards = agent.get_flashcards(collection_name="class10_science_ch03")
"""

from dataclasses import dataclass, field
from typing import Optional

from rich.console import Console

from pipeline.flashcard_gen  import Flashcard, FlashcardGenerator
from pipeline.highlight_tagger import HighlightTagger, TaggedChunk
from pipeline.ncert_fetcher   import NCERTFetcher
from pipeline.pdf_parser      import PDFParser
from pipeline.test_builder    import PersonalizedTest, StudentProfile, TestBuilder
from pipeline.vector_store    import VectorStore

console = Console()


@dataclass
class ChapterResult:
    """Complete output from processing a chapter."""
    collection_name: str
    class_num:       int
    subject:         str
    chapter:         int
    flashcards:      list[Flashcard]       = field(default_factory=list)
    tagged_chunks:   list[TaggedChunk]     = field(default_factory=list)
    key_terms:       list[str]             = field(default_factory=list)
    pdf_path:        str                   = ""

    @property
    def high_importance_chunks(self) -> list[TaggedChunk]:
        return [c for c in self.tagged_chunks if c.importance == "HIGH"]

    def summary(self) -> str:
        return (
            f"Chapter {self.chapter} | {self.subject.title()} | Class {self.class_num}\n"
            f"  Flashcards      : {len(self.flashcards)}\n"
            f"  Chunks tagged   : {len(self.tagged_chunks)} "
            f"({len(self.high_importance_chunks)} HIGH)\n"
            f"  Key terms       : {len(self.key_terms)}\n"
            f"  PDF             : {self.pdf_path}"
        )


class EduMindAgent:

    def __init__(self):
        console.print("[bold blue]🚀 EduMind Agent initializing...[/bold blue]")
        self.fetcher    = NCERTFetcher()
        self.parser     = PDFParser()
        self.store      = VectorStore()
        self.flashcard  = FlashcardGenerator()
        self.tagger     = HighlightTagger()
        self.test       = TestBuilder()
        console.print("[green]✓ All pipeline components ready[/green]\n")

    # ──────────────────────────────────────────────────────────────────────────
    # Core Workflows
    # ──────────────────────────────────────────────────────────────────────────

    def process_chapter(
        self,
        class_num:      int,
        subject:        str,
        chapter:        int,
        generate_highlights: bool = True,
        num_flashcards: int  = 15,
    ) -> ChapterResult:
        """
        Full pipeline for a single NCERT chapter:
          1. Fetch PDF from ncert.nic.in (cached after first run)
          2. Parse and chunk the text
          3. Embed chunks → ChromaDB
          4. Generate flashcards via RAG
          5. Tag chunks with importance levels (optional)

        Args:
            class_num            : 6–12
            subject              : e.g. "science", "chemistry_1"
            chapter              : chapter number
            generate_highlights  : run highlight tagger (costs extra LLM calls)
            num_flashcards       : how many flashcards to generate

        Returns:
            ChapterResult with flashcards, tagged chunks, key terms
        """
        collection_name = f"class{class_num}_{subject}_ch{str(chapter).zfill(2)}"
        console.rule(f"[bold]Processing: Class {class_num} | {subject} | Ch.{chapter}[/bold]")

        # ── Step 1: Fetch NCERT PDF ───────────────────────────────────────────
        pdf_path = self.fetcher.fetch(class_num, subject, chapter)

        # ── Step 2: Parse + Chunk ─────────────────────────────────────────────
        chunks = self.parser.parse_and_chunk(pdf_path, class_num, subject, chapter)

        # ── Step 3: Embed → Vector DB ─────────────────────────────────────────
        self.store.add_chunks(chunks, collection_name)

        # ── Step 4: Generate Flashcards ───────────────────────────────────────
        flashcards = self.flashcard.generate(
            collection_name = collection_name,
            subject         = subject,
            chapter         = chapter,
            class_num       = class_num,
            n               = num_flashcards,
        )

        # ── Step 5: Highlight Tagging (optional) ──────────────────────────────
        tagged_chunks = []
        key_terms     = []
        if generate_highlights:
            tagged_chunks = self.tagger.tag(chunks)
            key_terms     = self.tagger.get_key_terms(tagged_chunks)

        result = ChapterResult(
            collection_name = collection_name,
            class_num       = class_num,
            subject         = subject,
            chapter         = chapter,
            flashcards      = flashcards,
            tagged_chunks   = tagged_chunks,
            key_terms       = key_terms,
            pdf_path        = pdf_path,
        )

        console.print(f"\n[bold green]✅ Done![/bold green]")
        console.print(result.summary())
        return result

    def generate_test(
        self,
        student_id:   str,
        class_num:    int,
        subject:      str,
        chapter:      int,
        weak_topics:  Optional[list[str]]  = None,
        strong_topics: Optional[list[str]] = None,
        avg_accuracy: float = 0.5,
        num_questions: int  = 10,
    ) -> PersonalizedTest:
        """
        Generates a personalized test for a student.
        The chapter must already be processed (indexed in vector store).

        Args:
            student_id    : unique student identifier
            weak_topics   : list of topic strings (from student progress DB)
            strong_topics : topics the student is confident in
            avg_accuracy  : 0.0–1.0, used to calibrate difficulty
            num_questions : number of questions in the test

        Returns:
            PersonalizedTest
        """
        collection_name = f"class{class_num}_{subject}_ch{str(chapter).zfill(2)}"

        if not self.store.collection_exists(collection_name):
            raise RuntimeError(
                f"Chapter not indexed yet. Run process_chapter() first.\n"
                f"Collection: {collection_name}"
            )

        profile = StudentProfile(
            student_id    = student_id,
            class_num     = class_num,
            subject       = subject,
            weak_topics   = weak_topics  or ["general concepts"],
            strong_topics = strong_topics or [],
            avg_accuracy  = avg_accuracy,
        )

        return self.test.build(
            student_profile  = profile,
            collection_name  = collection_name,
            chapter          = chapter,
            num_questions    = num_questions,
        )

    def get_flashcards(
        self,
        class_num: int,
        subject:   str,
        chapter:   int,
        n:         int = 15,
    ) -> list[Flashcard]:
        """
        Re-generates flashcards for an already-indexed chapter.
        Useful for refreshing the deck without re-downloading the PDF.
        """
        collection_name = f"class{class_num}_{subject}_ch{str(chapter).zfill(2)}"
        return self.flashcard.generate(
            collection_name = collection_name,
            subject         = subject,
            chapter         = chapter,
            class_num       = class_num,
            n               = n,
        )

    def search_chapter(
        self,
        query:     str,
        class_num: int,
        subject:   str,
        chapter:   int,
        top_k:     int = 5,
    ) -> list[str]:
        """
        Free-text semantic search within an indexed chapter.
        Useful for the student to ask questions about specific concepts.
        """
        collection_name = f"class{class_num}_{subject}_ch{str(chapter).zfill(2)}"
        results = self.store.query(query, collection_name, top_k=top_k)
        return [r.text for r in results]

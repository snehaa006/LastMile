"""
agents/orchestrator.py
───────────────────────
The EduMind orchestrator — ties all pipeline components into a single
callable interface.

This is what your backend API or CLI calls. It decides which tools to
run and in what order based on the request type.

Usage:
    agent = EduMindAgent()

    # Ingest once (fetch → parse → embed, no LLM calls)
    agent.ingest_chapter(class_num=10, subject="science", chapter=3)

    # Then run any feature independently, in any order
    cards = agent.get_flashcards(class_num=10, subject="science", chapter=3)
    tagged, key_terms = agent.generate_highlights(class_num=10, subject="science", chapter=3)
    test = agent.generate_test(student_id="stu_001", class_num=10, subject="science", chapter=3)

    # Or run the old one-shot bundle (ingest + flashcards + highlights)
    result = agent.process_chapter(class_num=10, subject="science", chapter=3)
"""

from dataclasses import dataclass, field
from typing import Optional

from rich.console import Console

from pipeline.flashcard_gen     import Flashcard, FlashcardGenerator
from pipeline.formula_sheet_gen import FormulaSheet, FormulaSheetGenerator
from pipeline.highlight_tagger  import HighlightTagger, TaggedChunk
from pipeline.hot_questions_gen import HotQuestion, HotQuestionsGenerator
from pipeline.ncert_fetcher     import NCERTFetcher
from pipeline.notes_gen         import ChapterNotes, NotesGenerator
from pipeline.pdf_parser        import PDFParser
from pipeline.test_builder      import PersonalizedTest, StudentProfile, TestBuilder
from pipeline.vector_store      import VectorStore

console = Console()


@dataclass
class IngestResult:
    """Result of fetching + parsing + embedding a chapter. No LLM calls."""
    collection_name: str
    class_num:       int
    subject:         str
    chapter:         int
    pdf_path:        str
    num_chunks:      int


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
        self.fetcher       = NCERTFetcher()
        self.parser        = PDFParser()
        # One shared VectorStore (and its embedding model) reused by every
        # generator below — avoids loading the embedding model 6 times.
        self.store         = VectorStore()
        self.flashcard     = FlashcardGenerator(store=self.store)
        self.tagger        = HighlightTagger()
        self.test          = TestBuilder(store=self.store)
        self.formula_sheet = FormulaSheetGenerator(store=self.store)
        self.notes         = NotesGenerator(store=self.store)
        self.hot_questions = HotQuestionsGenerator(store=self.store)
        console.print("[green]✓ All pipeline components ready[/green]\n")

    # ──────────────────────────────────────────────────────────────────────────
    # Core Workflows
    # ──────────────────────────────────────────────────────────────────────────

    def ingest_chapter(
        self,
        class_num: int,
        subject:   str,
        chapter:   int,
    ) -> IngestResult:
        """
        Fetch → parse → embed a chapter. No LLM calls — this step is free
        and fast (PDF fetch and parsing are cached/local after the first
        run). Every content-generation feature (flashcards, highlights,
        notes, hot questions, formula sheet, tests) requires this to have
        run first for the chapter.
        """
        collection_name = f"class{class_num}_{subject}_ch{str(chapter).zfill(2)}"
        console.rule(f"[bold]Ingesting: Class {class_num} | {subject} | Ch.{chapter}[/bold]")

        pdf_path = self.fetcher.fetch(class_num, subject, chapter)
        chunks   = self.parser.parse_and_chunk(pdf_path, class_num, subject, chapter)
        self.store.add_chunks(chunks, collection_name)

        return IngestResult(
            collection_name = collection_name,
            class_num       = class_num,
            subject         = subject,
            chapter         = chapter,
            pdf_path        = pdf_path,
            num_chunks      = len(chunks),
        )

    def is_ingested(self, class_num: int, subject: str, chapter: int) -> bool:
        """Whether a chapter has already been fetched, parsed, and embedded."""
        collection_name = f"class{class_num}_{subject}_ch{str(chapter).zfill(2)}"
        return self.store.collection_exists(collection_name)

    def process_chapter(
        self,
        class_num:      int,
        subject:        str,
        chapter:        int,
        generate_highlights: bool = True,
        num_flashcards: int  = 15,
    ) -> ChapterResult:
        """
        Convenience wrapper for a one-shot run: ingest + flashcards (+
        highlights). Kept for the CLI demo — for a UI where each feature
        runs independently, call ingest_chapter() once, then any of the
        generate_*() methods on demand.
        """
        ingest = self.ingest_chapter(class_num, subject, chapter)

        flashcards = self.flashcard.generate(
            collection_name = ingest.collection_name,
            subject         = subject,
            chapter         = chapter,
            class_num       = class_num,
            n               = num_flashcards,
        )

        tagged_chunks: list[TaggedChunk] = []
        key_terms:     list[str]         = []
        if generate_highlights:
            tagged_chunks, key_terms = self._tag_chapter(class_num, subject, chapter)

        result = ChapterResult(
            collection_name = ingest.collection_name,
            class_num       = class_num,
            subject         = subject,
            chapter         = chapter,
            flashcards      = flashcards,
            tagged_chunks   = tagged_chunks,
            key_terms       = key_terms,
            pdf_path        = ingest.pdf_path,
        )

        console.print(f"\n[bold green]✅ Done![/bold green]")
        console.print(result.summary())
        return result

    def generate_highlights(
        self,
        class_num: int,
        subject:   str,
        chapter:   int,
    ) -> tuple[list[TaggedChunk], list[str]]:
        """
        Tags an already-ingested chapter's chunks with importance levels
        and extracts key terms. Can be run independently of flashcards.
        """
        self._require_indexed(class_num, subject, chapter)
        return self._tag_chapter(class_num, subject, chapter)

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
        collection_name = self._require_indexed(class_num, subject, chapter)

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

    def generate_formula_sheet(
        self,
        class_num: int,
        subject:   str,
        chapter:   int,
    ) -> FormulaSheet:
        """
        Builds a consolidated formula sheet for an already-indexed chapter.
        Most useful for mathematics and formula-heavy science chapters;
        returns an empty sheet for purely descriptive chapters.
        """
        collection_name = self._require_indexed(class_num, subject, chapter)
        return self.formula_sheet.generate(
            collection_name = collection_name,
            subject         = subject,
            chapter         = chapter,
            class_num       = class_num,
        )

    def generate_notes(
        self,
        class_num: int,
        subject:   str,
        chapter:   int,
    ) -> ChapterNotes:
        """Builds condensed revision notes for an already-indexed chapter."""
        collection_name = self._require_indexed(class_num, subject, chapter)
        return self.notes.generate(
            collection_name = collection_name,
            subject         = subject,
            chapter         = chapter,
            class_num       = class_num,
        )

    def get_hot_questions(
        self,
        class_num: int,
        subject:   str,
        chapter:   int,
        n:         int = 10,
    ) -> list[HotQuestion]:
        """
        Returns the ranked list of questions most likely to appear in the
        exam for an already-indexed chapter, blending PYQ repetition with
        LLM-predicted high-yield questions.
        """
        collection_name = self._require_indexed(class_num, subject, chapter)
        return self.hot_questions.generate(
            collection_name = collection_name,
            subject         = subject,
            chapter         = chapter,
            class_num       = class_num,
            n               = n,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Internal Helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _require_indexed(self, class_num: int, subject: str, chapter: int) -> str:
        collection_name = f"class{class_num}_{subject}_ch{str(chapter).zfill(2)}"
        if not self.store.collection_exists(collection_name):
            raise RuntimeError(
                f"Chapter not indexed yet. Run ingest_chapter() first.\n"
                f"Collection: {collection_name}"
            )
        return collection_name

    def _tag_chapter(
        self, class_num: int, subject: str, chapter: int
    ) -> tuple[list[TaggedChunk], list[str]]:
        """Re-parses the cached PDF (fast, no network) and tags its chunks."""
        pdf_path      = self.fetcher.fetch(class_num, subject, chapter)
        chunks        = self.parser.parse_and_chunk(pdf_path, class_num, subject, chapter)
        tagged_chunks = self.tagger.tag(chunks)
        key_terms     = self.tagger.get_key_terms(tagged_chunks)
        return tagged_chunks, key_terms

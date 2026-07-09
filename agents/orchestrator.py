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

    # ...or run flashcards/highlights/notes/hot-questions/formula-sheet all
    # at once, concurrently
    results = agent.generate_all(class_num=10, subject="science", chapter=3)

    # Or run the old one-shot bundle (ingest + flashcards + highlights)
    result = agent.process_chapter(class_num=10, subject="science", chapter=3)
"""

import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from rich.console import Console

from config import UPLOAD_PDF_PATH
from pipeline.flashcard_gen     import Flashcard, FlashcardGenerator
from pipeline.formula_sheet_gen import FormulaSheet, FormulaSheetGenerator
from pipeline.highlight_tagger  import HighlightTagger, TaggedChunk
from pipeline.homework_gen      import HomeworkGenerator, HomeworkPaper, PaperBlock
from pipeline.homework_grader   import HomeworkGrader, PaperGrade
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
        self.homework      = HomeworkGenerator(store=self.store)
        self.grader        = HomeworkGrader()
        # In-memory record of where each ingested collection's source PDF
        # lives (and what page range, for uploaded books) — lets
        # _tag_chapter() re-parse without guessing the source. Lost on
        # process restart, same as the vector store's in-memory session.
        self._ingested: dict[str, dict] = {}
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
        collection_name = self._collection_name(class_num, subject, chapter)
        console.rule(f"[bold]Ingesting: Class {class_num} | {subject} | Ch.{chapter}[/bold]")

        pdf_path = self.fetcher.fetch(class_num, subject, chapter)
        chunks   = self.parser.parse_and_chunk(pdf_path, class_num, subject, chapter)
        self.store.add_chunks(chunks, collection_name)
        self._ingested[collection_name] = {"pdf_path": pdf_path, "page_range": None}

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
        return self.store.collection_exists(self._collection_name(class_num, subject, chapter))

    def collection_name(self, class_num: int, subject: str, chapter: int) -> str:
        """Public accessor for the collection-name format — lets a caller
        (e.g. the dashboard) predict the identifier before calling any
        generate_*() method, including for an upload where subject is
        derived from slugify(book_title)."""
        return self._collection_name(class_num, subject, chapter)

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
        collection_name = self._collection_name(class_num, subject, chapter)
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
        collection_name = self._collection_name(class_num, subject, chapter)
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

    def list_sections(
        self,
        class_num: int,
        subject:   str,
        chapter:   int,
    ) -> list[str]:
        """
        Returns the indexed chapter's section/topic headings, so the Homework
        composition builder can offer section-level scoping ("generate only
        from these parts of the chapter"). Empty if the chapter isn't indexed
        or no headings were extracted.
        """
        collection_name = self._collection_name(class_num, subject, chapter)
        return self.store.get_sections(collection_name)

    def generate_homework(
        self,
        class_num:  int,
        subject:    str,
        chapter:    int,
        blocks:     list[PaperBlock],
        difficulty: str = "medium",
        sections:   Optional[list[str]] = None,
    ) -> HomeworkPaper:
        """
        Generates an exam-style homework paper for an already-indexed chapter
        from a composition of question blocks (each a count + style + optional
        difficulty), optionally scoped to specific chapter sections. This is
        the AI-generation half of the Homework feature; grade the student's
        answers with grade_homework().
        """
        collection_name = self._require_indexed(class_num, subject, chapter)
        return self.homework.generate(
            collection_name = collection_name,
            subject         = subject,
            chapter         = chapter,
            class_num       = class_num,
            blocks          = blocks,
            difficulty      = difficulty,
            sections        = sections,
        )

    def grade_homework(
        self,
        paper:   HomeworkPaper,
        answers: dict,
    ) -> PaperGrade:
        """
        Grades a student's answers to a generated homework paper — objective
        questions (MCQ/fill-blank) locally, subjective questions via a single
        batched LLM call against each question's mark scheme.
        """
        return self.grader.grade(paper, answers)

    def generate_all(
        self,
        class_num: int,
        subject:   str,
        chapter:   int,
        num_flashcards:    int = 15,
        num_hot_questions: int = 10,
    ) -> dict:
        """
        Runs every chapter-level generator — flashcards, highlights, notes,
        hot questions, formula sheet — concurrently instead of one after
        another. The chapter must already be ingested. Personalized tests
        and semantic search need extra input from the user (student
        profile, a query), so they aren't part of this bundle.

        Returns a dict keyed by feature name ("flashcards", "highlights",
        "notes", "hot_questions", "formula_sheet"). If a generator raises,
        its value is the exception instead of a result — one slow or
        failed feature doesn't block the others from completing.
        """
        collection_name = self._require_indexed(class_num, subject, chapter)

        jobs = {
            "flashcards": lambda: self.flashcard.generate(
                collection_name = collection_name,
                subject         = subject,
                chapter         = chapter,
                class_num       = class_num,
                n               = num_flashcards,
            ),
            "highlights": lambda: self._tag_chapter(class_num, subject, chapter),
            "notes": lambda: self.notes.generate(
                collection_name = collection_name,
                subject         = subject,
                chapter         = chapter,
                class_num       = class_num,
            ),
            "hot_questions": lambda: self.hot_questions.generate(
                collection_name = collection_name,
                subject         = subject,
                chapter         = chapter,
                class_num       = class_num,
                n               = num_hot_questions,
            ),
            "formula_sheet": lambda: self.formula_sheet.generate(
                collection_name = collection_name,
                subject         = subject,
                chapter         = chapter,
                class_num       = class_num,
            ),
        }

        console.rule(f"[bold]Generating everything (parallel): {collection_name}[/bold]")
        results: dict = {}
        with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
            future_to_name = {pool.submit(fn): name for name, fn in jobs.items()}
            for future in as_completed(future_to_name):
                name = future_to_name[future]
                try:
                    results[name] = future.result()
                    console.print(f"[green]✓ {name} done[/green]")
                except Exception as e:
                    console.print(f"[red]✗ {name} failed:[/red] {e}")
                    results[name] = e

        return results

    # ──────────────────────────────────────────────────────────────────────────
    # Internal Helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _collection_name(self, class_num: int, subject: str, chapter: int) -> str:
        return f"class{class_num}_{subject}_ch{str(chapter).zfill(2)}"

    def _require_indexed(self, class_num: int, subject: str, chapter: int) -> str:
        collection_name = self._collection_name(class_num, subject, chapter)
        if not self.store.collection_exists(collection_name):
            raise RuntimeError(
                f"Chapter not indexed yet. Run ingest_chapter() first.\n"
                f"Collection: {collection_name}"
            )
        return collection_name

    def _tag_chapter(
        self, class_num: int, subject: str, chapter: int
    ) -> tuple[list[TaggedChunk], list[str]]:
        """Re-parses the source PDF (fast, no network for a cached/uploaded
        file) and tags its chunks. Looks up where the PDF lives — and, for
        uploaded books, which page range is this "chapter" — from the
        in-memory ingest registry rather than assuming NCERT."""
        collection_name = self._collection_name(class_num, subject, chapter)
        meta = self._ingested.get(collection_name)

        if meta is not None:
            pdf_path, page_range = meta["pdf_path"], meta["page_range"]
        elif class_num != 0:
            # NCERT chapter ingested in an earlier process — fetch() is a
            # cache hit if the PDF is already on disk.
            pdf_path, page_range = self.fetcher.fetch(class_num, subject, chapter), None
        else:
            raise RuntimeError(
                "This uploaded book's source PDF isn't available in this "
                "session (the app may have restarted). Please re-upload and "
                "re-ingest it."
            )

        chunks = self.parser.parse_and_chunk(
            pdf_path, class_num, subject, chapter, page_range=page_range
        )
        tagged_chunks = self.tagger.tag(chunks)
        key_terms     = self.tagger.get_key_terms(tagged_chunks)
        return tagged_chunks, key_terms

    def slugify(self, text: str, max_len: int = 40) -> str:
        slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
        return slug[:max_len] or "untitled"

    # ──────────────────────────────────────────────────────────────────────────
    # Custom uploads (books that aren't in the NCERT catalog)
    # ──────────────────────────────────────────────────────────────────────────

    def save_uploaded_pdf(self, file_bytes: bytes, filename: str) -> str:
        """Saves an uploaded PDF to local disk and returns its path."""
        Path(UPLOAD_PDF_PATH).mkdir(parents=True, exist_ok=True)
        safe_name = self.slugify(Path(filename).stem) + ".pdf"
        path = os.path.join(UPLOAD_PDF_PATH, safe_name)
        with open(path, "wb") as f:
            f.write(file_bytes)
        return path

    def detect_chapters(self, pdf_path: str) -> dict:
        """
        Returns {"toc": [...], "page_count": N} for an uploaded PDF, so the
        caller can offer a chapter picker. "toc" (the PDF's embedded
        bookmarks/outline) is empty for PDFs that don't have one — common
        for scanned books or ones assembled by hand — in which case the
        caller should fall back to a manual page-range picker using
        "page_count".
        """
        return self.parser.get_toc_with_page_count(pdf_path)

    def ingest_uploaded_pdf(
        self,
        pdf_path:   str,
        book_title: str,
        start_page: int,
        end_page:   int,
    ) -> IngestResult:
        """
        Ingests one page range of an arbitrary uploaded PDF as a "chapter".
        Not from the NCERT catalog, so class_num/subject are synthesized
        from the book title (class_num=0 is the "custom upload" sentinel;
        subject is a slug of the title). Every other feature (flashcards,
        notes, hot questions, ...) then works on it exactly like an NCERT
        chapter — they only need class_num/subject/chapter.
        """
        class_num = 0
        subject   = self.slugify(book_title)
        chapter   = start_page  # not a real chapter number — a unique slot per page range
        collection_name = self._collection_name(class_num, subject, chapter)

        console.rule(f"[bold]Ingesting upload: {book_title} (pages {start_page}-{end_page})[/bold]")

        chunks = self.parser.parse_and_chunk(
            pdf_path, class_num, subject, chapter, page_range=(start_page, end_page)
        )
        self.store.add_chunks(chunks, collection_name)
        self._ingested[collection_name] = {
            "pdf_path": pdf_path,
            "page_range": (start_page, end_page),
        }

        return IngestResult(
            collection_name = collection_name,
            class_num       = class_num,
            subject         = subject,
            chapter         = chapter,
            pdf_path        = pdf_path,
            num_chunks      = len(chunks),
        )

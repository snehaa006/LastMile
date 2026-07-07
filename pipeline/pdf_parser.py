"""
pipeline/pdf_parser.py
───────────────────────
Parses PDFs using PyMuPDF and splits the text into overlapping chunks with
metadata (page number, chapter, subject) for downstream embedding. Works
on NCERT chapter PDFs (whole-document) and on arbitrary uploaded PDFs
restricted to a page range (get_toc_with_page_count() + page_range=).

Usage:
    parser = PDFParser()
    chunks = parser.parse_and_chunk("./data/ncert_pdfs/class10_science_ch03.pdf",
                                     class_num=10, subject="science", chapter=3)
    # → List[Chunk]

    # Or restricted to a page range (e.g. one chapter of an uploaded book)
    chunks = parser.parse_and_chunk(pdf_path, class_num=0, subject="my_book",
                                     chapter=1, page_range=(12, 34))
"""

import re
from dataclasses import dataclass, field
from typing import Optional

import fitz  # PyMuPDF
from rich.console import Console

from config import CHUNK_SIZE, CHUNK_OVERLAP

console = Console()


@dataclass
class Chunk:
    """A single text chunk with metadata."""
    text:       str
    chunk_id:   str                # "class10_science_ch03_chunk_007"
    class_num:  int
    subject:    str
    chapter:    int
    page:       int
    topics:     list[str] = field(default_factory=list)   # extracted headings


class PDFParser:

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def parse_and_chunk(
        self,
        pdf_path: str,
        class_num: int,
        subject: str,
        chapter: int,
        page_range: Optional[tuple[int, int]] = None,
    ) -> list[Chunk]:
        """
        Extracts text from a PDF and splits it into overlapping chunks.

        Args:
            page_range : optional (start_page, end_page), 1-indexed inclusive.
                         Restricts parsing to that range — used for uploaded
                         books where only part of the PDF is the "chapter"
                         (e.g. a chapter selected from the PDF's table of
                         contents, or a manually picked page span). None
                         parses the whole document, as for NCERT chapters
                         which are already single-chapter PDFs.

        Returns:
            List[Chunk] — ordered list of text chunks with metadata
        """
        console.print(f"[blue]📄 Parsing:[/blue] {pdf_path}")

        pages = self._extract_pages(pdf_path, page_range)
        full_text, page_map = self._build_text_with_page_map(pages)
        raw_chunks = self._split_into_chunks(full_text)
        topics = self._extract_headings(full_text)

        chunks = []
        for idx, raw in enumerate(raw_chunks):
            page_num = self._find_page(raw["start"], page_map)
            chunk = Chunk(
                text      = raw["text"],
                chunk_id  = f"class{class_num}_{subject}_ch{str(chapter).zfill(2)}_chunk_{str(idx).zfill(3)}",
                class_num = class_num,
                subject   = subject,
                chapter   = chapter,
                page      = page_num,
                topics    = topics,
            )
            chunks.append(chunk)

        console.print(
            f"[green]✓ Parsed:[/green] {len(pages)} pages → {len(chunks)} chunks"
        )
        return chunks

    # ──────────────────────────────────────────────────────────────────────────
    # Internal Helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _extract_pages(
        self, pdf_path: str, page_range: Optional[tuple[int, int]] = None
    ) -> list[dict]:
        """Extracts text from each page using PyMuPDF, optionally restricted
        to a 1-indexed inclusive page_range."""
        doc = fitz.open(pdf_path)
        pages = []
        for page_num, page in enumerate(doc, start=1):
            if page_range and not (page_range[0] <= page_num <= page_range[1]):
                continue
            text = page.get_text("text")
            text = self._clean_text(text)
            if text.strip():
                pages.append({"page": page_num, "text": text})
        doc.close()
        return pages

    def get_toc_with_page_count(self, pdf_path: str) -> dict:
        """
        Returns the PDF's embedded table of contents (bookmarks/outline)
        plus its total page count — used to let a user pick a "chapter"
        out of an arbitrary uploaded PDF.

        Returns:
            {
                "toc": [{"title": str, "level": int, "start_page": int,
                          "end_page": int}, ...],   # 1-indexed, inclusive
                "page_count": int,
            }
            "toc" is empty if the PDF has no embedded outline — common for
            scanned or manually-assembled PDFs. Callers should fall back
            to a manual page-range picker in that case.
        """
        doc = fitz.open(pdf_path)
        page_count = doc.page_count
        raw_toc = doc.get_toc()  # [[level, title, start_page], ...], 1-indexed
        doc.close()

        if not raw_toc:
            return {"toc": [], "page_count": page_count}

        entries = []
        for i, (level, title, start) in enumerate(raw_toc):
            end = page_count
            for next_level, _, next_start in raw_toc[i + 1:]:
                if next_level <= level:
                    end = next_start - 1
                    break
            entries.append({
                "title":      title.strip(),
                "level":      level,
                "start_page": start,
                "end_page":   max(end, start),
            })
        return {"toc": entries, "page_count": page_count}

    def _build_text_with_page_map(
        self, pages: list[dict]
    ) -> tuple[str, list[dict]]:
        """
        Concatenates all pages into a single string.
        Builds a map of character offset → page number for later lookup.
        """
        full_text = ""
        page_map  = []   # [{"start": 0, "end": 1200, "page": 1}, ...]

        for p in pages:
            start = len(full_text)
            full_text += p["text"] + "\n\n"
            end = len(full_text)
            page_map.append({"start": start, "end": end, "page": p["page"]})

        return full_text, page_map

    def _split_into_chunks(self, text: str) -> list[dict]:
        """
        Splits text into overlapping chunks of CHUNK_SIZE characters.
        Tries to break at sentence boundaries (". ") rather than mid-word.
        """
        chunks = []
        start  = 0
        total  = len(text)

        while start < total:
            end = min(start + CHUNK_SIZE, total)

            # Try to end at a sentence boundary
            if end < total:
                boundary = text.rfind(". ", start, end)
                if boundary != -1 and boundary > start + CHUNK_SIZE // 2:
                    end = boundary + 1   # include the period

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append({"text": chunk_text, "start": start})

            # Reached the end of the text — stop, otherwise `start` recomputes
            # to the same value forever (CHUNK_SIZE > CHUNK_OVERLAP guarantees
            # end - CHUNK_OVERLAP + CHUNK_SIZE >= total once end == total).
            if end >= total:
                break

            # Advance with overlap
            start = end - CHUNK_OVERLAP

        return chunks

    def _extract_headings(self, text: str) -> list[str]:
        """
        Extracts likely section headings (short all-caps or title-case lines).
        Used to tag chunks with topic context.
        """
        headings = []
        for line in text.split("\n"):
            line = line.strip()
            # Heuristic: short line, starts with capital, no period at end
            if (
                5 < len(line) < 80
                and line[0].isupper()
                and not line.endswith(".")
                and len(line.split()) < 10
            ):
                headings.append(line)
        return list(dict.fromkeys(headings))[:20]   # deduplicate, cap at 20

    def _find_page(self, char_offset: int, page_map: list[dict]) -> int:
        """Returns the page number for a given character offset."""
        for entry in page_map:
            if entry["start"] <= char_offset < entry["end"]:
                return entry["page"]
        return page_map[-1]["page"] if page_map else 1

    def _clean_text(self, text: str) -> str:
        """Removes noisy artifacts common in NCERT PDFs."""
        # Remove headers/footers (standalone page numbers, chapter codes)
        text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
        # Collapse excessive whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" {2,}", " ", text)
        # Remove ligature artifacts
        text = text.replace("\ufb01", "fi").replace("\ufb02", "fl")
        return text.strip()

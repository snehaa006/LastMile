"""
pipeline/ncert_fetcher.py
─────────────────────────
Downloads NCERT chapter PDFs from ncert.nic.in.

Usage:
    fetcher = NCERTFetcher()
    path = fetcher.fetch(class_num=10, subject="science", chapter=3)
    # → "./data/ncert_pdfs/class10_science_ch03.pdf"
"""

import os
import re
import time

import requests
from pathlib import Path
from rich.console import Console

from config import NCERT_CODES, NCERT_BASE_URL, NCERT_PDF_PATH

console = Console()

MAX_RETRIES     = 3
RETRY_BACKOFF_S = 2   # doubles each retry: 2s, 4s, 8s


class NCERTFetcher:

    def __init__(self):
        Path(NCERT_PDF_PATH).mkdir(parents=True, exist_ok=True)

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def fetch(self, class_num: int, subject: str, chapter: int) -> str:
        """
        Downloads the NCERT PDF for the given class/subject/chapter.
        Returns the local file path. Uses cache if already downloaded.

        Args:
            class_num : int  → 6 to 12
            subject   : str  → e.g. "science", "chemistry_1", "biology"
            chapter   : int  → chapter number (1-indexed)

        Returns:
            str → absolute path to the downloaded PDF
        """
        subject = subject.lower().strip().replace(" ", "_")

        # Check cache first
        local_path = self._local_path(class_num, subject, chapter)
        if os.path.exists(local_path):
            console.print(f"[green]✓ Cache hit:[/green] {local_path}")
            return local_path

        # Build download URL
        url = self._build_url(class_num, subject, chapter)
        console.print(f"[blue]↓ Fetching:[/blue] {url}")

        response = self._download(url, class_num, subject, chapter)
        with open(local_path, "wb") as f:
            f.write(response.content)

        console.print(f"[green]✓ Saved:[/green] {local_path}")
        return local_path

    def list_available(self, class_num: int) -> list[str]:
        """Returns list of subject keys available for a given class."""
        return list(NCERT_CODES.get(class_num, {}).keys())

    # ──────────────────────────────────────────────────────────────────────────
    # Internal Helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _build_url(self, class_num: int, subject: str, chapter: int) -> str:
        class_map = NCERT_CODES.get(class_num)
        if not class_map:
            raise ValueError(f"Class {class_num} not supported. Choose 6–12.")

        code = class_map.get(subject)
        if not code:
            available = list(class_map.keys())
            raise ValueError(
                f"Subject '{subject}' not found for class {class_num}. "
                f"Available: {available}"
            )

        chapter_str = str(chapter).zfill(2)  # "3" → "03"
        return f"{NCERT_BASE_URL}/{code}{chapter_str}.pdf"

    def _local_path(self, class_num: int, subject: str, chapter: int) -> str:
        filename = f"class{class_num}_{subject}_ch{str(chapter).zfill(2)}.pdf"
        return os.path.join(NCERT_PDF_PATH, filename)

    def _sibling_part_hint(self, class_num: int, subject: str) -> str:
        """
        NCERT splits some subjects into two books (Part 1 / Part 2) with
        non-overlapping chapter numbers — e.g. Class 11 Physics Part 1 might
        only go up to chapter 8, with 9+ living in Part 2. A 404 on a
        "_1"/"_2" subject is very often just the wrong part for that chapter
        number, so suggest the sibling if one exists in the catalog.
        """
        match = re.match(r"^(.+)_([12])$", subject)
        if not match:
            return ""
        base, part = match.group(1), match.group(2)
        sibling = f"{base}_{'2' if part == '1' else '1'}"
        if sibling in NCERT_CODES.get(class_num, {}):
            return (
                f" This subject is split into Part 1/Part 2 books with "
                f"non-overlapping chapter numbers — try subject '{sibling}' instead."
            )
        return ""

    def _download(
        self, url: str, class_num: int, subject: str, chapter: int
    ) -> requests.Response:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "application/pdf,application/octet-stream,*/*",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://ncert.nic.in/textbook.php",
        }

        last_error: Exception = RuntimeError("unreachable")
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = requests.get(url, headers=headers, timeout=30)
                resp.raise_for_status()
                if "application/pdf" not in resp.headers.get("Content-Type", ""):
                    raise ValueError(
                        f"URL did not return a PDF. Got: {resp.headers.get('Content-Type')}"
                    )
                return resp
            except requests.exceptions.HTTPError as e:
                # A real HTTP error (404, etc.) means the URL/chapter combo is
                # wrong — retrying won't help, fail fast with a clear message.
                hint = self._sibling_part_hint(class_num, subject)
                raise RuntimeError(
                    f"Failed to download NCERT PDF.\n"
                    f"URL: {url}\n"
                    f"Error: {e}\n"
                    f"Tip: Chapter {chapter} doesn't exist for class {class_num} "
                    f"'{subject}'.{hint} You can check the real chapter list at "
                    f"https://ncert.nic.in/textbook.php."
                ) from e
            except requests.exceptions.RequestException as e:
                # Connection reset / timeout / DNS blip — can be transient,
                # so retry with backoff before giving up.
                last_error = e
                if attempt < MAX_RETRIES:
                    wait = RETRY_BACKOFF_S * (2 ** (attempt - 1))
                    console.print(
                        f"[yellow]⚠ Connection issue (attempt {attempt}/{MAX_RETRIES}):[/yellow] "
                        f"{e}\nRetrying in {wait}s..."
                    )
                    time.sleep(wait)

        raise RuntimeError(
            f"Could not reach NCERT after {MAX_RETRIES} attempts.\n"
            f"URL: {url}\n"
            f"Last error: {last_error}\n"
            f"This usually means ncert.nic.in is refusing connections from this "
            f"server's network (common for cloud/datacenter IPs hitting Indian "
            f"government sites) rather than a real outage. If this keeps "
            f"happening only when deployed (and works locally), the fix isn't "
            f"retrying harder — it's pre-fetching PDFs somewhere this block "
            f"doesn't apply and shipping them with the app instead of fetching "
            f"live on every deploy."
        ) from last_error

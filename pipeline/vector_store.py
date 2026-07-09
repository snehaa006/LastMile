"""
pipeline/vector_store.py
─────────────────────────
Manages ChromaDB collections for NCERT chunks and PYQs.
Uses sentence-transformers for local embeddings (no API cost).

Usage:
    store = VectorStore()
    store.add_chunks(chunks, collection_name="class10_science_ch03")
    results = store.query("what is photosynthesis", collection_name="class10_science_ch03")
"""

from dataclasses import dataclass
from typing import Optional

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from rich.console import Console

from config import CHROMA_DB_PATH, EMBEDDING_MODEL, TOP_K_CHUNKS
from pipeline.pdf_parser import Chunk

console = Console()


@dataclass
class QueryResult:
    """A single retrieved chunk with its similarity score."""
    text:     str
    chunk_id: str
    page:     int
    subject:  str
    chapter:  int
    score:    float   # distance (lower = more similar)


class VectorStore:

    def __init__(self):
        self._client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        self._ef = SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
        console.print(f"[blue]🗄  VectorStore:[/blue] {CHROMA_DB_PATH}")

    # ──────────────────────────────────────────────────────────────────────────
    # Ingestion
    # ──────────────────────────────────────────────────────────────────────────

    def add_chunks(self, chunks: list[Chunk], collection_name: str) -> None:
        """
        Embeds and stores a list of Chunks into a ChromaDB collection.
        If the collection already exists, it is reused (idempotent).

        Args:
            chunks          : list of Chunk objects from PDFParser
            collection_name : e.g. "class10_science_ch03"
        """
        collection = self._get_or_create(collection_name)

        # Check for existing IDs to skip re-embedding (idempotent)
        existing = set(collection.get()["ids"])
        new_chunks = [c for c in chunks if c.chunk_id not in existing]

        if not new_chunks:
            console.print(
                f"[yellow]⚡ Already indexed:[/yellow] {collection_name} ({len(chunks)} chunks)"
            )
            return

        collection.add(
            ids        = [c.chunk_id for c in new_chunks],
            documents  = [c.text for c in new_chunks],
            metadatas  = [self._chunk_to_meta(c) for c in new_chunks],
        )
        console.print(
            f"[green]✓ Indexed:[/green] {len(new_chunks)} chunks → '{collection_name}'"
        )

    def add_pyqs(self, pyqs: list[dict], collection_name: str = "pyq_bank") -> None:
        """
        Stores PYQ questions in a dedicated collection.

        Each PYQ dict should have:
            - question (str)
            - answer   (str)
            - subject  (str)
            - chapter  (str or int)
            - year     (int)
            - marks    (int)
            - class_num (int)
        """
        collection = self._get_or_create(collection_name)
        existing   = set(collection.get()["ids"])

        ids, docs, metas = [], [], []
        for i, pyq in enumerate(pyqs):
            pyq_id = f"pyq_{pyq['subject']}_{pyq['year']}_{i}"
            if pyq_id in existing:
                continue
            ids.append(pyq_id)
            docs.append(pyq["question"] + " " + pyq.get("answer", ""))
            metas.append({
                "question":  pyq["question"],
                "answer":    pyq.get("answer", ""),
                "subject":   pyq["subject"],
                "chapter":   str(pyq.get("chapter", "")),
                "year":      pyq.get("year", 0),
                "marks":     pyq.get("marks", 1),
                "class_num": pyq.get("class_num", 10),
                "type":      "pyq",
            })

        if ids:
            collection.add(ids=ids, documents=docs, metadatas=metas)
            console.print(f"[green]✓ PYQs indexed:[/green] {len(ids)} questions")

    # ──────────────────────────────────────────────────────────────────────────
    # Retrieval
    # ──────────────────────────────────────────────────────────────────────────

    def query(
        self,
        query_text: str,
        collection_name: str,
        top_k: int = TOP_K_CHUNKS,
        where: Optional[dict] = None,
    ) -> list[QueryResult]:
        """
        Retrieves the top-k most relevant chunks for a query.

        Args:
            query_text      : natural language query
            collection_name : ChromaDB collection to search
            top_k           : number of results to return
            where           : optional metadata filter, e.g. {"chapter": "3"}

        Returns:
            List[QueryResult] sorted by relevance (best first)
        """
        collection = self._client.get_collection(
            name=collection_name, embedding_function=self._ef
        )

        kwargs = dict(query_texts=[query_text], n_results=top_k, include=["documents", "metadatas", "distances"])
        if where:
            kwargs["where"] = where

        res = collection.query(**kwargs)

        results = []
        for doc, meta, dist in zip(
            res["documents"][0],
            res["metadatas"][0],
            res["distances"][0],
        ):
            results.append(
                QueryResult(
                    text     = doc,
                    chunk_id = meta.get("chunk_id", ""),
                    page     = meta.get("page", 0),
                    subject  = meta.get("subject", ""),
                    chapter  = meta.get("chapter", 0),
                    score    = round(dist, 4),
                )
            )
        return results

    def query_pyqs(
        self,
        topic: str,
        subject: str,
        class_num: int,
        top_k: int = 5,
    ) -> list[dict]:
        """
        Retrieves PYQs relevant to a topic for a given subject and class.
        """
        collection = self._client.get_collection(
            name="pyq_bank", embedding_function=self._ef
        )
        res = collection.query(
            query_texts=[topic],
            n_results=top_k,
            where={"subject": subject, "class_num": class_num},
            include=["metadatas", "distances"],
        )
        return res["metadatas"][0] if res["metadatas"] else []

    def collection_exists(self, collection_name: str) -> bool:
        try:
            self._client.get_collection(collection_name)
            return True
        except Exception:
            return False

    def get_sections(self, collection_name: str) -> list[str]:
        """
        Returns the chapter's extracted section/topic headings for an indexed
        collection, so a caller can offer section-level scoping (e.g. the
        Homework composition builder). PDFParser stores the chapter's headings
        as a comma-joined "topics" string on every chunk's metadata; this
        splits and de-duplicates them across the collection, preserving order.
        """
        try:
            collection = self._client.get_collection(collection_name)
        except Exception:
            return []
        metas = collection.get(include=["metadatas"]).get("metadatas") or []
        sections: list[str] = []
        seen: set[str] = set()
        for meta in metas:
            for topic in (meta.get("topics", "") or "").split(","):
                topic = topic.strip()
                if topic and topic.lower() not in seen:
                    seen.add(topic.lower())
                    sections.append(topic)
        return sections

    # ──────────────────────────────────────────────────────────────────────────
    # Internal Helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _get_or_create(self, name: str):
        return self._client.get_or_create_collection(
            name=name, embedding_function=self._ef
        )

    def _chunk_to_meta(self, chunk: Chunk) -> dict:
        return {
            "chunk_id":  chunk.chunk_id,
            "class_num": chunk.class_num,
            "subject":   chunk.subject,
            "chapter":   chunk.chapter,
            "page":      chunk.page,
            "topics":    ", ".join(chunk.topics[:5]),
        }

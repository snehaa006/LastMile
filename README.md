# EduMind — AI Study Companion

An LLM + RAG pipeline that turns a school textbook chapter into everything a
student needs to revise for it: flashcards, hot/likely exam questions, PYQ-based
tests, a formula sheet, highlighted key points, and condensed notes.

## The idea

Give it a class, subject, and chapter. It:

1. Downloads the textbook chapter (currently NCERT, CBSE curriculum)
2. Parses and chunks the text
3. Embeds it into a vector store for retrieval
4. Runs a set of Claude-powered generators over the retrieved content to produce:
   - **Flashcards** — Q&A pairs for spaced revision
   - **Hot questions** — the questions most likely to appear in the exam,
     blending repeated PYQ topics with LLM-predicted high-yield questions
   - **PYQ-based personalized tests** — calibrated to a student's weak topics
     and accuracy history
   - **Formula sheet** — every formula/equation/theorem in the chapter,
     organized and exam-ready (maths & formula-heavy science)
   - **Highlights** — HIGH/MEDIUM/LOW importance tagging per paragraph, plus
     key terms worth a popup definition, for an "e-textbook" reading view
   - **Notes** — condensed revision notes (TL;DR + bulleted sections)

This is **Phase 1**: a working core pipeline, single content source, local
storage, CLI-driven. It is meant to be reliable and cleanly separated so it
can be scaled into a real product — see [Roadmap](#roadmap).

## Architecture (current)

```
Student Input (class, subject, chapter)
        │
        ▼
┌─────────────────┐
│  NCERTFetcher    │  downloads chapter PDF from ncert.nic.in (cached to disk)
└────────┬─────────┘
         ▼
┌─────────────────┐
│  PDFParser       │  PyMuPDF extraction → cleaned text → overlapping chunks
└────────┬─────────┘  + heading/topic extraction, page-offset mapping
         ▼
┌─────────────────┐
│  VectorStore     │  sentence-transformers embeddings → ChromaDB
└────────┬─────────┘  (one collection per chapter, idempotent writes)
         │
         ├──────────────┬──────────────┬───────────────┬───────────────┬──────────────┐
         ▼              ▼              ▼               ▼               ▼              ▼
   FlashcardGen   HighlightTagger  TestBuilder   FormulaSheetGen   NotesGen   HotQuestionsGen
   (RAG → Claude) (HIGH/MED/LOW)  (PYQ + AI mix) (math/science)   (summary)  (PYQ + AI mix)
         │              │              │               │               │              │
         └──────────────┴──────────────┴───────────────┴───────────────┴──────────────┘
                                        ▼
                          ┌───────────────────────────┐
                          │  EduMindAgent (orchestrator)│  single facade over all of the above
                          └───────────────────────────┘
                                        ▼
                              main.py (CLI demo)
```

Every generator follows the same shape: **retrieve** relevant chunks from the
vector store with topic-diverse seed queries → **prompt** Claude with a
strict JSON-only response format → **parse** into typed dataclasses, with a
graceful fallback (not a crash) if the model output doesn't parse. This
keeps the generators independent, individually testable, and cheap to add to
— which is how `formula_sheet_gen.py`, `notes_gen.py`, and
`hot_questions_gen.py` were bolted on without touching the fetch/parse/embed
layer.

### What's already reliable

- Idempotent ingestion — re-running `process_chapter()` skips already-embedded
  chunks and cached PDFs instead of redoing work
- Clean separation between ingestion (fetch/parse/embed), generation
  (Claude-backed), and orchestration — each layer can be swapped independently
- Local embeddings (sentence-transformers) — no per-query API cost or latency
  for retrieval, only generation calls hit the LLM

### Known gaps (why this is "Phase 1," not production)

- **No API layer** — everything is driven through a `rich` CLI demo; nothing
  a frontend can call yet
- **No persistence for students** — `StudentProfile` and `record_attempt()`
  are in-memory mocks; weak/strong topics must be passed in by hand
- **Synchronous, unretried LLM calls** — a single failed Claude call or a
  malformed JSON response silently degrades (empty list / MEDIUM default)
  instead of retrying or repairing
- **Local, single-process vector store** — ChromaDB's `PersistentClient` on
  local disk won't survive multi-instance deployment
- **Single content source** — hardcoded to `ncert.nic.in` URL patterns; no
  abstraction for other publishers/boards yet
- **No auth, no rate limiting, no observability** beyond console logging

## Project Structure

```
.
├── main.py                    ← entry point / demo
├── config.py                  ← NCERT URL maps, model settings, paths
├── requirements.txt
├── .env.example                ← copy to .env and add your API key
│
├── pipeline/
│   ├── ncert_fetcher.py       ← downloads NCERT PDFs from ncert.nic.in
│   ├── pdf_parser.py          ← extracts + chunks PDF text (PyMuPDF)
│   ├── vector_store.py        ← ChromaDB embed/retrieve (sentence-transformers)
│   ├── flashcard_gen.py       ← RAG flashcard generation via Claude
│   ├── highlight_tagger.py    ← importance classification for e-NCERT
│   ├── test_builder.py        ← personalized test generation
│   ├── formula_sheet_gen.py   ← formula sheet extraction (maths/science)
│   ├── notes_gen.py           ← condensed revision notes
│   └── hot_questions_gen.py   ← likely-exam-question prediction
│
├── agents/
│   └── orchestrator.py        ← ties all pipeline components together
│
└── data/
    ├── ncert_pdfs/            ← cached NCERT chapter PDFs (gitignored)
    ├── pyqs/                  ← PYQ JSON files (tracked)
    └── chroma_db/             ← ChromaDB persistent storage (gitignored)
```

## Setup

```bash
# 1. Clone and install dependencies
pip install -r requirements.txt

# 2. Set up your API key
cp .env.example .env
# Edit .env and add: ANTHROPIC_API_KEY=your_key

# 3. Run the demo
python main.py
```

## Supported Classes & Subjects

| Class | Subjects |
|-------|---------|
| 6–8   | science, mathematics, social_science, english, hindi |
| 9–10  | science, mathematics, social_science, english |
| 11    | physics_1/2, chemistry_1/2, biology, mathematics, economics, accountancy_1/2, business, history, political_sci |
| 12    | physics_1/2, chemistry_1/2, biology, mathematics_1/2, economics, accountancy_1/2, business, history_1, political_sci |

## Quick Usage

```python
from agents.orchestrator import EduMindAgent

agent = EduMindAgent()

# Process a chapter (fetch → parse → embed → flashcards → highlights)
result = agent.process_chapter(class_num=10, subject="science", chapter=1)

# Generate a personalized test
test = agent.generate_test(
    student_id   = "stu_001",
    class_num    = 10,
    subject      = "science",
    chapter      = 1,
    weak_topics  = ["photosynthesis", "respiration"],
    avg_accuracy = 0.45,
)

# Hot / likely exam questions (PYQ repetition + LLM prediction)
hot = agent.get_hot_questions(class_num=10, subject="science", chapter=1)

# Condensed revision notes
notes = agent.generate_notes(class_num=10, subject="science", chapter=1)

# Formula sheet (best for maths / formula-heavy science chapters)
sheet = agent.generate_formula_sheet(class_num=10, subject="mathematics", chapter=6)

# Semantic search inside a chapter
results = agent.search_chapter(
    query     = "what is osmosis",
    class_num = 10,
    subject   = "science",
    chapter   = 1,
)
```

## Adding PYQs

Put PYQ JSON files in `data/pyqs/`. Format:
```json
[
  {
    "question": "Define photosynthesis.",
    "answer": "Photosynthesis is the process by which...",
    "subject": "science",
    "chapter": 1,
    "year": 2023,
    "marks": 2,
    "class_num": 10
  }
]
```

Then load them:
```python
import json
from pipeline.vector_store import VectorStore

store = VectorStore()
with open("data/pyqs/science_class10.json") as f:
    pyqs = json.load(f)
store.add_pyqs(pyqs)
```

## Roadmap

**Phase 1 — Core pipeline (this repo, current state)**
Fetch → parse → embed → flashcards, highlights, PYQ tests, formula sheets,
notes, hot questions. Single process, local storage, CLI-driven. Goal:
prove the retrieval + generation quality is good before building around it.

**Phase 2 — Reliability & API**
Turn the orchestrator into a real service instead of a CLI demo:
- Wrap `EduMindAgent` in a FastAPI service (one endpoint per generator)
- Structured LLM output validation with retry-and-repair on malformed JSON
  (not silent fallback to empty/MEDIUM)
- Async task queue for chapter processing (Celery/RQ, or async Claude calls
  with backoff) — chapter ingestion + all-generator runs are slow enough to
  need background jobs, not a blocking request
- Postgres for student profiles, attempt history, and weak-topic tracking
  (currently mocked in `test_builder.py`)
- Swap local ChromaDB for a hosted/networked vector store so the API can run
  as more than one process
- Containerize (Docker) + basic structured logging/metrics

**Phase 3 — Multi-source content**
- Generalize `NCERTFetcher` into a pluggable `BookSource` interface
- Add other boards/publishers (ICSE, state boards, JEE/NEET reference books)
- OCR fallback for scanned/image-based PDFs
- Admin tooling to manage the book catalog instead of hardcoded URL maps

**Phase 4 — Real personalization & scale**
- Weak/strong topics computed from actual `record_attempt()` history instead
  of passed in manually
- Spaced-repetition scheduling for flashcards
- Adaptive test difficulty (IRT-style) instead of static accuracy buckets
- Multi-user auth, Redis caching for hot content, horizontal scaling of
  embedding/generation workers

**Phase 5 — Frontend & delivery**
- Student-facing web/app UI, e-textbook reader with inline highlights and
  key-term popups, teacher/parent dashboards, notifications

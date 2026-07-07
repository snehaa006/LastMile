# EduMind — AI Pipeline

CBSE study companion powered by RAG + LLM agents.

## Project Structure

```
edumind/
├── main.py                    ← entry point / demo
├── config.py                  ← NCERT URL maps, model settings, paths
├── requirements.txt
├── .env.example               ← copy to .env and add your API key
│
├── pipeline/
│   ├── ncert_fetcher.py       ← downloads NCERT PDFs from ncert.nic.in
│   ├── pdf_parser.py          ← extracts + chunks PDF text (PyMuPDF)
│   ├── vector_store.py        ← ChromaDB embed/retrieve (sentence-transformers)
│   ├── flashcard_gen.py       ← RAG flashcard generation via Claude
│   ├── highlight_tagger.py    ← importance classification for e-NCERT
│   └── test_builder.py        ← personalized test generation
│
├── agents/
│   └── orchestrator.py        ← ties all pipeline components together
│
└── data/
    ├── ncert_pdfs/            ← cached NCERT chapter PDFs
    ├── pyqs/                  ← PYQ JSON files
    └── chroma_db/             ← ChromaDB persistent storage
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

## Pipeline Flow

```
Student Input (class, subject, chapter)
        ↓
NCERTFetcher     → downloads PDF from ncert.nic.in (cached)
        ↓
PDFParser        → extracts text, splits into overlapping chunks
        ↓
VectorStore      → embeds chunks → ChromaDB (sentence-transformers)
        ↓
FlashcardGen     → RAG: retrieves chunks → Claude → Q&A flashcards
        ↓
HighlightTagger  → Claude classifies each chunk: HIGH / MEDIUM / LOW
        ↓
TestBuilder      → pulls weak-topic PYQs + generates AI questions
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

import os
from dotenv import load_dotenv

load_dotenv()

# ── API Keys ──────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GEMINI_API_KEY    = os.getenv("GEMINI_API_KEY", "")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
CHROMA_DB_PATH  = os.path.join(BASE_DIR, "data", "chroma_db")
NCERT_PDF_PATH  = os.path.join(BASE_DIR, "data", "ncert_pdfs")
PYQ_PATH        = os.path.join(BASE_DIR, "data", "pyqs")

# ── Model Settings ────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # local, no API needed

# LLM_PROVIDER selects which LLM backend pipeline/llm_client.py talks to.
#   "gemini"    → Google Gemini API (has a free tier — good default for dev)
#   "anthropic" → Claude API (higher quality, no free tier, pay-per-token)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")

GEMINI_MODEL = "gemini-2.5-flash"      # free-tier eligible
LLM_MODEL    = "claude-sonnet-4-6"     # used only when LLM_PROVIDER="anthropic"

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE    = 600   # characters per chunk
CHUNK_OVERLAP = 80    # overlap between consecutive chunks

# ── Flashcard Generation ──────────────────────────────────────────────────────
FLASHCARDS_PER_CHAPTER = 15   # target flashcards per chapter
TOP_K_CHUNKS           = 5    # chunks retrieved per flashcard query

# ── Highlight Tagger ──────────────────────────────────────────────────────────
# Labels returned by the LLM for each paragraph
IMPORTANCE_LABELS = ["HIGH", "MEDIUM", "LOW"]

# ── NCERT URL Mapping ─────────────────────────────────────────────────────────
# Pattern: https://ncert.nic.in/textbook/pdf/{code}{chapter_zfill2}.pdf
# Code format: [class_prefix][e][subject_code][part]
#
# class prefix:  f=6, g=7, h=8, i=9, j=10, k=11, l=12
# 'e' = English medium (fixed)
# part = 1 for most subjects (some have part 2 as well)

NCERT_CODES = {
    6: {
        "science":        "fesc1",
        "mathematics":    "femh1",
        "social_science": "fess1",
        "english":        "feen1",
        "hindi":          "fehi1",
    },
    7: {
        "science":        "gesc1",
        "mathematics":    "gemh1",
        "social_science": "gess1",
        "english":        "geen1",
    },
    8: {
        "science":        "hesc1",
        "mathematics":    "hemh1",
        "social_science": "hess1",
        "english":        "heen1",
    },
    9: {
        "science":        "iesc1",
        "mathematics":    "iemh1",
        "social_science": "iess1",
        "english":        "ieen1",
    },
    10: {
        "science":        "jesc1",
        "mathematics":    "jemh1",
        "social_science": "jess1",
        "english":        "jeen1",
    },
    11: {
        "physics_1":      "keph1",
        "physics_2":      "keph2",
        "chemistry_1":    "kech1",
        "chemistry_2":    "kech2",
        "biology":        "kebo1",
        "mathematics":    "kemh1",
        "economics":      "keec1",
        "accountancy_1":  "keac1",
        "accountancy_2":  "keac2",
        "business":       "kebs1",
        "history":        "kehi1",
        "political_sci":  "keps1",
    },
    12: {
        "physics_1":      "leph1",
        "physics_2":      "leph2",
        "chemistry_1":    "lech1",
        "chemistry_2":    "lech2",
        "biology":        "lebo1",
        "mathematics_1":  "lemh1",
        "mathematics_2":  "lemh2",
        "economics":      "leec1",
        "accountancy_1":  "leac1",
        "accountancy_2":  "leac2",
        "business":       "lebs1",
        "history_1":      "lehi1",
        "political_sci":  "leps1",
    },
}

NCERT_BASE_URL = "https://ncert.nic.in/textbook/pdf"

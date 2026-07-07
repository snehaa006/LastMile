"""
app.py — EduMind Dashboard
─────────────────────────────
Streamlit UI for the EduMind pipeline. Pick a class/subject/chapter, fetch
and index it, then run any feature independently — nothing runs unless you
click its button.

Run:
    streamlit run app.py
"""

import os

import streamlit as st

# On Streamlit Community Cloud, API keys are set as "secrets" (st.secrets),
# not real environment variables. Mirror them into os.environ *before*
# importing config.py, so its os.getenv() calls work unchanged whether
# you're running locally with .env or deployed with Streamlit secrets.
try:
    for _key in ("LLM_PROVIDER", "GEMINI_API_KEY", "ANTHROPIC_API_KEY"):
        if _key in st.secrets:
            os.environ.setdefault(_key, st.secrets[_key])
except Exception:
    pass  # no secrets.toml — normal for local dev, .env handles it there

from agents.orchestrator import EduMindAgent
from config import NCERT_CODES

st.set_page_config(page_title="EduMind Dashboard", page_icon="📚", layout="wide")


@st.cache_resource(show_spinner="Initializing EduMind agent (loading embedding model)...")
def get_agent() -> EduMindAgent:
    return EduMindAgent()


def result_key(feature: str, collection_name: str) -> str:
    return f"{feature}::{collection_name}"


st.title("📚 EduMind — CBSE Study Dashboard")

try:
    agent = get_agent()
except RuntimeError as e:
    st.error(
        f"**Couldn't start the agent:** {e}\n\n"
        "Copy `.env.example` to `.env` and set your API key, then restart the app."
    )
    st.stop()

# ── Chapter picker ────────────────────────────────────────────────────────
st.sidebar.header("1. Pick a chapter")
class_num = st.sidebar.selectbox("Class", sorted(NCERT_CODES.keys()), index=4)
subjects  = list(NCERT_CODES.get(class_num, {}).keys())
subject   = st.sidebar.selectbox("Subject", subjects)
chapter   = st.sidebar.number_input("Chapter", min_value=1, max_value=30, value=1, step=1)

collection_name  = f"class{class_num}_{subject}_ch{str(chapter).zfill(2)}"
already_ingested = agent.is_ingested(class_num, subject, chapter)

st.sidebar.markdown("---")
st.sidebar.header("2. Fetch & index")
if already_ingested:
    st.sidebar.success(f"Indexed: `{collection_name}`")
else:
    st.sidebar.warning("Not indexed yet")

if st.sidebar.button("📥 Fetch & Index Chapter", type="primary", use_container_width=True):
    with st.spinner("Downloading PDF, parsing, embedding — no LLM calls, this is free..."):
        try:
            ingest = agent.ingest_chapter(class_num, subject, chapter)
            st.sidebar.success(f"Indexed {ingest.num_chunks} chunks from {ingest.pdf_path}")
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Failed: {e}")

if not already_ingested:
    st.info("👈 Fetch & index a chapter first using the sidebar. This step is free — no LLM calls.")
    st.stop()

st.caption(f"Working on **Class {class_num} · {subject.title()} · Chapter {chapter}** (`{collection_name}`)")

# ── Feature tabs — each one is independent, nothing runs until you click it ─
tab_flash, tab_highlight, tab_notes, tab_hot, tab_formula, tab_test, tab_search = st.tabs(
    ["🃏 Flashcards", "🔍 Highlights", "📓 Notes", "🔥 Hot Questions", "Σ Formula Sheet", "📝 Test", "🔎 Search"]
)

# ── Flashcards ───────────────────────────────────────────────────────────
with tab_flash:
    n = st.slider("Number of flashcards", 5, 30, 15, key="flash_n")
    if st.button("Generate Flashcards", key="btn_flash"):
        with st.spinner("Calling LLM..."):
            try:
                cards = agent.get_flashcards(class_num, subject, chapter, n=n)
                st.session_state[result_key("flash", collection_name)] = cards
            except Exception as e:
                st.error(str(e))

    cards = st.session_state.get(result_key("flash", collection_name))
    if cards:
        st.dataframe(
            [{"Q": c.question, "A": c.answer, "Difficulty": c.difficulty, "Topic": c.topic} for c in cards],
            use_container_width=True,
        )

# ── Highlights ───────────────────────────────────────────────────────────
with tab_highlight:
    if st.button("Generate Highlights", key="btn_highlight"):
        with st.spinner("Calling LLM..."):
            try:
                tagged, key_terms = agent.generate_highlights(class_num, subject, chapter)
                st.session_state[result_key("highlight", collection_name)] = (tagged, key_terms)
            except Exception as e:
                st.error(str(e))

    cached = st.session_state.get(result_key("highlight", collection_name))
    if cached:
        tagged, key_terms = cached
        high = [c for c in tagged if c.importance == "HIGH"]
        st.write(f"**{len(high)} / {len(tagged)}** chunks marked HIGH importance")
        if key_terms:
            st.write("**Key terms:** " + ", ".join(key_terms[:20]))
        for c in high:
            with st.expander(f"Page {c.page} — {c.reason[:80]}"):
                st.write(c.text)
                if c.key_terms:
                    st.caption("Key terms: " + ", ".join(c.key_terms))

# ── Notes ────────────────────────────────────────────────────────────────
with tab_notes:
    if st.button("Generate Notes", key="btn_notes"):
        with st.spinner("Calling LLM..."):
            try:
                notes = agent.generate_notes(class_num, subject, chapter)
                st.session_state[result_key("notes", collection_name)] = notes
            except Exception as e:
                st.error(str(e))

    notes = st.session_state.get(result_key("notes", collection_name))
    if notes:
        if notes.tldr:
            st.info(notes.tldr)
        for section in notes.sections:
            st.subheader(section.heading)
            for b in section.bullets:
                st.markdown(f"- {b}")

# ── Hot Questions ────────────────────────────────────────────────────────
with tab_hot:
    n_hot = st.slider("Number of hot questions", 5, 20, 10, key="hot_n")
    if st.button("Generate Hot Questions", key="btn_hot"):
        with st.spinner("Calling LLM..."):
            try:
                hot = agent.get_hot_questions(class_num, subject, chapter, n=n_hot)
                st.session_state[result_key("hot", collection_name)] = hot
            except Exception as e:
                st.error(str(e))

    hot = st.session_state.get(result_key("hot", collection_name))
    if hot:
        st.dataframe(
            [
                {
                    "Question": q.question,
                    "Likelihood": q.likelihood,
                    "Basis": q.basis,
                    "Source": q.source,
                    "Marks": q.marks,
                }
                for q in hot
            ],
            use_container_width=True,
        )

# ── Formula Sheet ────────────────────────────────────────────────────────
with tab_formula:
    if st.button("Generate Formula Sheet", key="btn_formula"):
        with st.spinner("Calling LLM..."):
            try:
                sheet = agent.generate_formula_sheet(class_num, subject, chapter)
                st.session_state[result_key("formula", collection_name)] = sheet
            except Exception as e:
                st.error(str(e))

    sheet = st.session_state.get(result_key("formula", collection_name))
    if sheet:
        if not sheet.entries:
            st.write("No formulas found for this chapter.")
        else:
            st.dataframe(
                [
                    {
                        "Name": f.name,
                        "Expression": f.expression,
                        "Variables": f.variables,
                        "When to use": f.when_to_use,
                        "Topic": f.topic,
                    }
                    for f in sheet.entries
                ],
                use_container_width=True,
            )

# ── Personalized Test ────────────────────────────────────────────────────
with tab_test:
    col1, col2 = st.columns(2)
    with col1:
        student_id = st.text_input("Student ID", value="stu_001", key="test_student")
        weak_raw   = st.text_input("Weak topics (comma-separated)", key="test_weak")
        strong_raw = st.text_input("Strong topics (comma-separated)", key="test_strong")
    with col2:
        avg_accuracy  = st.slider("Average accuracy", 0.0, 1.0, 0.5, key="test_accuracy")
        num_questions = st.slider("Number of questions", 5, 20, 10, key="test_n")

    if st.button("Generate Test", key="btn_test"):
        with st.spinner("Calling LLM..."):
            try:
                weak_topics   = [t.strip() for t in weak_raw.split(",") if t.strip()] or None
                strong_topics = [t.strip() for t in strong_raw.split(",") if t.strip()] or None
                test = agent.generate_test(
                    student_id    = student_id,
                    class_num     = class_num,
                    subject       = subject,
                    chapter       = chapter,
                    weak_topics   = weak_topics,
                    strong_topics = strong_topics,
                    avg_accuracy  = avg_accuracy,
                    num_questions = num_questions,
                )
                st.session_state[result_key("test", collection_name)] = test
            except Exception as e:
                st.error(str(e))

    test = st.session_state.get(result_key("test", collection_name))
    if test:
        st.write(
            f"**{len(test.questions)} questions, {test.total_marks} marks** — "
            f"weak focus: {', '.join(test.weak_focus) or 'none'}"
        )
        for i, q in enumerate(test.questions, 1):
            with st.expander(f"Q{i} [{q.marks}m | {q.difficulty}] {q.source}"):
                st.write(q.question)
                st.caption(f"Answer: {q.answer}")
                if q.hint:
                    st.caption(f"Hint: {q.hint}")

# ── Semantic Search (free — no LLM calls) ────────────────────────────────
with tab_search:
    st.caption("Free-text search over the indexed chapter. No LLM calls — pure vector retrieval.")
    query = st.text_input("Ask a question about this chapter", key="search_query")
    top_k = st.slider("Results", 1, 10, 5, key="search_k")
    if st.button("Search", key="btn_search") and query:
        results = agent.search_chapter(
            query=query, class_num=class_num, subject=subject, chapter=chapter, top_k=top_k
        )
        st.session_state[result_key("search", collection_name)] = results

    results = st.session_state.get(result_key("search", collection_name))
    if results:
        for i, r in enumerate(results, 1):
            st.markdown(f"**Result {i}**")
            st.write(r)

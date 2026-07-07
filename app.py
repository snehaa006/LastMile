"""
app.py — EduMind Dashboard
─────────────────────────────
Streamlit UI for the EduMind pipeline. Pick a class/subject/chapter, then
one click fetches, indexes, and generates flashcards, highlights, notes,
hot questions, and a formula sheet — all in parallel. Each tab also has
its own regenerate button if you want to refresh just one feature.

Run:
    streamlit run app.py
"""

import html
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

THEME_CSS = """
<style>
.block-container { padding-top: 2rem; max-width: 1200px; }

.em-card {
    background: #FFFFFF;
    border: 1px solid #E8DFD3;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: .8rem;
}
.em-card h4 { margin: 0 0 .4rem 0; font-size: 1rem; }
.em-status-done { color: #1A7A3C; font-weight: 600; font-size: .9rem; }
.em-status-pending { color: #9A9284; font-size: .9rem; }

.em-callout {
    border-radius: 8px;
    padding: .85rem 1.1rem;
    margin: .7rem 0;
}
.em-label {
    font-size: .7rem;
    letter-spacing: .06em;
    text-transform: uppercase;
    font-weight: 700;
    margin-bottom: .25rem;
    display: block;
}
.em-tldr      { background: #FFF4E3; border-left: 4px solid #D98E36; }
.em-tldr .em-label      { color: #B06A1B; }
.em-bigidea   { background: #FFF4E3; border-left: 4px solid #D98E36; }
.em-bigidea .em-label   { color: #B06A1B; }
.em-definition{ background: #F5F0E8; border-left: 4px solid #C1622F; }
.em-definition .em-label{ color: #C1622F; }
.em-formula   { background: #FDF8EF; border: 1px solid #E8DFD3; border-left: 4px solid #2B2521;
                font-family: "SFMono-Regular", Consolas, monospace; }
.em-formula .em-label   { font-family: sans-serif; color: #2B2521; }
.em-example   { background: #FBEDE3; border-left: 4px solid #C1622F; }
.em-example .em-label   { color: #C1622F; }

.em-table { width: 100%; border-collapse: collapse; margin: .5rem 0 1.1rem 0; font-size: .92rem; }
.em-table th { background: #2B2521; color: #FFFFFF; text-align: left; padding: .5rem .7rem; }
.em-table td { padding: .5rem .7rem; border-bottom: 1px solid #E8DFD3; }
.em-table tr:nth-child(even) td { background: #FBF7F0; }
.em-table-title { font-weight: 600; margin: .3rem 0 .2rem 0; }

.em-section-heading { margin-top: 1.6rem; margin-bottom: .2rem; }
</style>
"""
st.markdown(THEME_CSS, unsafe_allow_html=True)


def esc(text: str) -> str:
    return html.escape(text or "")


def render_callout(css_class: str, label: str, content: str) -> None:
    st.markdown(
        f'<div class="em-callout {css_class}">'
        f'<span class="em-label">{esc(label)}</span>{esc(content)}'
        f"</div>",
        unsafe_allow_html=True,
    )


def render_block(block) -> None:
    if block.type == "definition":
        render_callout("em-definition", f"Definition — {block.title}", block.content)
    elif block.type == "formula":
        render_callout("em-formula", f"Formula — {block.title}", block.content)
    elif block.type == "example":
        render_callout("em-example", f"Example — {block.title}", block.content)
    elif block.type == "bullets" and block.items:
        items_html = "".join(f"<li>{esc(i)}</li>" for i in block.items)
        st.markdown(f"<ul>{items_html}</ul>", unsafe_allow_html=True)
    elif block.type == "table" and block.table_rows:
        if block.title:
            st.markdown(f'<div class="em-table-title">{esc(block.title)}</div>', unsafe_allow_html=True)
        header_html = "".join(f"<th>{esc(h)}</th>" for h in block.table_headers)
        rows_html = "".join(
            "<tr>" + "".join(f"<td>{esc(str(c))}</td>" for c in row) + "</tr>"
            for row in block.table_rows
        )
        st.markdown(
            f"<table class='em-table'><thead><tr>{header_html}</tr></thead>"
            f"<tbody>{rows_html}</tbody></table>",
            unsafe_allow_html=True,
        )


@st.cache_resource(show_spinner="Initializing EduMind agent (loading embedding model)...")
def get_agent() -> EduMindAgent:
    return EduMindAgent()


def result_key(feature: str, collection_name: str) -> str:
    return f"{feature}::{collection_name}"


# Maps orchestrator's generate_all() result keys to this file's per-tab cache keys
ORCH_TO_CACHE_KEY = {
    "flashcards": "flash",
    "highlights": "highlight",
    "notes": "notes",
    "hot_questions": "hot",
    "formula_sheet": "formula",
}

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
st.sidebar.header("2. Generate")
st.sidebar.caption(
    "Fetches the chapter (if needed) and runs flashcards, highlights, "
    "notes, hot questions, and a formula sheet — all in parallel."
)

if st.sidebar.button("🚀 Fetch, Index & Generate Everything", type="primary", use_container_width=True):
    with st.spinner("Fetching + indexing..."):
        try:
            agent.ingest_chapter(class_num, subject, chapter)
        except Exception as e:
            st.sidebar.error(f"Ingest failed: {e}")
            st.stop()
    with st.spinner("Running all generators in parallel — this calls the LLM 5 times at once..."):
        results = agent.generate_all(class_num, subject, chapter)
        failed = []
        for orch_key, value in results.items():
            cache_key = ORCH_TO_CACHE_KEY[orch_key]
            if isinstance(value, Exception):
                failed.append((orch_key, value))
                continue
            st.session_state[result_key(cache_key, collection_name)] = value
        if failed:
            for name, err in failed:
                st.sidebar.error(f"{name} failed: {err}")
        st.sidebar.success(f"Done — {len(results) - len(failed)}/{len(results)} features generated.")
        st.rerun()

if already_ingested:
    st.sidebar.success(f"Indexed: `{collection_name}`")
else:
    st.sidebar.warning("Not indexed yet — click the button above.")

if not already_ingested:
    st.info("👈 Pick a chapter and click **Fetch, Index & Generate Everything** in the sidebar.")
    st.stop()

st.caption(f"Working on **Class {class_num} · {subject.title()} · Chapter {chapter}** (`{collection_name}`)")

# ── Chapter overview — status cards ─────────────────────────────────────────
overview_features = [
    ("flash", "🃏 Flashcards"),
    ("highlight", "🔍 Highlights"),
    ("notes", "📓 Notes"),
    ("hot", "🔥 Hot Questions"),
    ("formula", "Σ Formula Sheet"),
]
cols = st.columns(len(overview_features))
for col, (key, label) in zip(cols, overview_features):
    done = result_key(key, collection_name) in st.session_state
    status_html = (
        '<span class="em-status-done">✓ Generated</span>'
        if done
        else '<span class="em-status-pending">Not yet</span>'
    )
    col.markdown(f'<div class="em-card"><h4>{label}</h4>{status_html}</div>', unsafe_allow_html=True)

# ── Feature tabs — each also has its own regenerate button ─────────────────
tab_flash, tab_highlight, tab_notes, tab_hot, tab_formula, tab_test, tab_search = st.tabs(
    ["🃏 Flashcards", "🔍 Highlights", "📓 Notes", "🔥 Hot Questions", "Σ Formula Sheet", "📝 Test", "🔎 Search"]
)

# ── Flashcards ───────────────────────────────────────────────────────────
with tab_flash:
    n = st.slider("Number of flashcards", 5, 30, 15, key="flash_n")
    if st.button("Regenerate Flashcards", key="btn_flash"):
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
    else:
        st.caption("Not generated yet.")

# ── Highlights ───────────────────────────────────────────────────────────
with tab_highlight:
    if st.button("Regenerate Highlights", key="btn_highlight"):
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
    else:
        st.caption("Not generated yet.")

# ── Notes ────────────────────────────────────────────────────────────────
with tab_notes:
    if st.button("Regenerate Notes", key="btn_notes"):
        with st.spinner("Calling LLM..."):
            try:
                notes = agent.generate_notes(class_num, subject, chapter)
                st.session_state[result_key("notes", collection_name)] = notes
            except Exception as e:
                st.error(str(e))

    notes = st.session_state.get(result_key("notes", collection_name))
    if notes:
        if notes.tldr:
            render_callout("em-tldr", "TL;DR", notes.tldr)
        for section in notes.sections:
            st.markdown(f'<h4 class="em-section-heading">{esc(section.heading)}</h4>', unsafe_allow_html=True)
            if section.big_idea:
                render_callout("em-bigidea", "Big Idea", section.big_idea)
            for block in section.blocks:
                render_block(block)
    else:
        st.caption("Not generated yet.")

# ── Hot Questions ────────────────────────────────────────────────────────
with tab_hot:
    n_hot = st.slider("Number of hot questions", 5, 20, 10, key="hot_n")
    if st.button("Regenerate Hot Questions", key="btn_hot"):
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
    else:
        st.caption("Not generated yet.")

# ── Formula Sheet ────────────────────────────────────────────────────────
with tab_formula:
    if st.button("Regenerate Formula Sheet", key="btn_formula"):
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
    else:
        st.caption("Not generated yet.")

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
    else:
        st.caption("Fill in a student profile above and click Generate Test.")

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

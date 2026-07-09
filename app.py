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

import pandas as pd
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
from pipeline.homework_gen import PaperBlock
from pipeline.homework_styles import STYLE_TEMPLATES, get_template

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


def render_answer_key(paper) -> None:
    """The teacher-facing answer sheet: correct options, expected answers, and
    mark schemes for every question in the paper."""
    for i, q in enumerate(paper.questions, 1):
        st.markdown(f"**Q{i}. [{q.marks}m]** {esc(q.prompt)}", unsafe_allow_html=True)
        if q.type == "mcq":
            for j, opt in enumerate(q.options):
                mark = "  ✅" if j == q.correct_index else ""
                st.write(f"{chr(65 + j)}. {opt}{mark}")
        elif q.type == "fill_blank":
            st.write(f"Answer: **{q.expected_answer}**")
            if q.acceptable_answers:
                st.caption("Also accepted: " + ", ".join(q.acceptable_answers))
        else:
            if q.model_answer:
                st.caption(f"Model answer: {q.model_answer}")
            for pp in q.mark_scheme:
                st.write(f"- ({pp.marks}m) {pp.point}")
        st.markdown("---")


def render_homework(agent, paper, grade_key: str) -> None:
    """Renders the attempt form for a generated paper, grades it on submit,
    and shows per-question marks + feedback."""
    if paper.errors:
        for e in paper.errors:
            st.warning(e)
    if not paper.questions:
        st.error("No questions were generated — adjust the composition and try again.")
        return

    st.success(f"Paper ready: **{paper.total_questions} questions · {paper.total_marks} marks**"
               + (f" · sections: {', '.join(paper.sections)}" if paper.sections else " · whole chapter"))

    with st.form(key=f"hw_form::{paper.paper_id}"):
        answers: dict = {}
        for i, q in enumerate(paper.questions, 1):
            tmpl = STYLE_TEMPLATES.get(q.style)
            label = tmpl.label if tmpl else q.type
            st.markdown(f"**Q{i}. [{q.marks}m · {label}]**")
            st.write(q.prompt)
            wkey = f"hw_w::{paper.paper_id}::{q.id}"
            if q.type == "mcq":
                answers[q.id] = st.radio(
                    "Choose one", options=list(range(len(q.options))),
                    format_func=lambda j, opts=q.options: opts[j],
                    index=None, key=wkey, label_visibility="collapsed",
                )
            elif q.type == "fill_blank":
                answers[q.id] = st.text_input("Your answer", key=wkey, label_visibility="collapsed")
            else:
                answers[q.id] = st.text_area("Your answer", key=wkey, label_visibility="collapsed", height=120)
            st.markdown("---")
        submitted = st.form_submit_button("✅ Submit for grading", type="primary")

    if submitted:
        with st.spinner("Grading with AI (objective auto-marked, subjective marked against the scheme)..."):
            try:
                st.session_state[grade_key] = agent.grade_homework(paper, answers)
            except Exception as e:
                st.error(f"Grading failed: {e}")

    grade = st.session_state.get(grade_key)
    if grade and grade.paper_id == paper.paper_id:
        st.markdown(f"### 🎯 Result: {grade.total_marks} / {grade.max_marks} ({grade.percentage}%)")
        st.progress(min(1.0, grade.percentage / 100))
        if grade.ai_incomplete:
            st.warning("Some subjective answers couldn't be auto-graded and are marked "
                       "*pending* — review them against the answer key below.")
        by_id = {g.question_id: g for g in grade.per_question}
        for i, q in enumerate(paper.questions, 1):
            g = by_id.get(q.id)
            if not g:
                continue
            icon = "✅" if g.awarded == g.max else ("🟡" if g.awarded > 0 else "❌")
            with st.expander(f"{icon} Q{i}: {g.awarded}/{g.max} marks · graded by {g.graded_by}"):
                st.write(q.prompt)
                st.caption(f"Feedback: {g.feedback}")
                for pp in g.per_point:
                    st.write(f"- ({pp.awarded}/{pp.max}) {pp.point}")

    with st.expander("🔑 Answer key & mark scheme"):
        render_answer_key(paper)


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

# ── Source picker: NCERT catalog or an uploaded PDF ─────────────────────────
st.sidebar.header("1. Pick a source")
source_mode = st.sidebar.radio("Source", ["NCERT Catalog", "Upload your own PDF"], key="source_mode")

class_num = subject = chapter = collection_name = book_title = None
upload_pdf_path = upload_start_page = upload_end_page = None
ready_to_generate = False

if source_mode == "NCERT Catalog":
    class_num = st.sidebar.selectbox("Class", sorted(NCERT_CODES.keys()), index=4)
    subjects  = list(NCERT_CODES.get(class_num, {}).keys())
    subject   = st.sidebar.selectbox("Subject", subjects)
    chapter   = st.sidebar.number_input("Chapter", min_value=1, max_value=30, value=1, step=1)
    collection_name   = agent.collection_name(class_num, subject, chapter)
    ready_to_generate = True

else:  # Upload your own PDF
    uploaded_file = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_file is None:
        st.sidebar.info("Upload a PDF to continue.")
    else:
        default_title = uploaded_file.name.rsplit(".", 1)[0]
        book_title = st.sidebar.text_input("Book title", value=default_title, key="upload_title")

        # Save to disk once per unique upload — not on every rerun
        save_cache_key = f"upload_path::{uploaded_file.name}::{uploaded_file.size}"
        if save_cache_key not in st.session_state:
            st.session_state[save_cache_key] = agent.save_uploaded_pdf(
                uploaded_file.getvalue(), uploaded_file.name
            )
        upload_pdf_path = st.session_state[save_cache_key]

        # Detect chapters once per file — parsing the whole PDF isn't free
        detect_cache_key = f"detected::{upload_pdf_path}"
        if detect_cache_key not in st.session_state:
            st.session_state[detect_cache_key] = agent.detect_chapters(upload_pdf_path)
        detected = st.session_state[detect_cache_key]
        toc, page_count = detected["toc"], detected["page_count"]

        if toc:
            st.sidebar.caption(f"Found {len(toc)} chapters in this PDF's table of contents.")
            option_labels = [f"{e['title']} (p.{e['start_page']}–{e['end_page']})" for e in toc]
            chosen_idx = st.sidebar.selectbox(
                "Chapter", range(len(toc)), format_func=lambda i: option_labels[i]
            )
            upload_start_page = toc[chosen_idx]["start_page"]
            upload_end_page   = toc[chosen_idx]["end_page"]
        else:
            st.sidebar.caption(
                f"No table of contents found in this PDF ({page_count} pages) — "
                f"pick the page range for this chapter manually."
            )
            upload_start_page = st.sidebar.number_input(
                "Start page", min_value=1, max_value=page_count, value=1
            )
            upload_end_page = st.sidebar.number_input(
                "End page", min_value=1, max_value=page_count, value=min(10, page_count)
            )
            if upload_end_page < upload_start_page:
                st.sidebar.error("End page must be ≥ start page.")

        if book_title.strip() and upload_end_page and upload_end_page >= upload_start_page:
            class_num = 0
            subject   = agent.slugify(book_title)
            chapter   = upload_start_page
            collection_name   = agent.collection_name(class_num, subject, chapter)
            ready_to_generate = True

def _ingest() -> bool:
    """Fetches/indexes the current chapter. Returns True on success."""
    try:
        if source_mode == "NCERT Catalog":
            agent.ingest_chapter(class_num, subject, chapter)
        else:
            agent.ingest_uploaded_pdf(upload_pdf_path, book_title, upload_start_page, upload_end_page)
        return True
    except Exception as e:
        st.sidebar.error(f"Ingest failed: {e}")
        return False


st.sidebar.markdown("---")
st.sidebar.header("2. Index the chapter")
st.sidebar.caption("Free — no LLM calls. Required before generating anything below.")

already_ingested = ready_to_generate and agent.is_ingested(class_num, subject, chapter)

if ready_to_generate:
    if st.sidebar.button("📥 Fetch & Index", use_container_width=True):
        with st.spinner("Fetching + indexing..."):
            if _ingest():
                st.rerun()

    if already_ingested:
        st.sidebar.success(f"Indexed: `{collection_name}`")
    else:
        st.sidebar.warning("Not indexed yet — click the button above.")

    st.sidebar.markdown("---")
    st.sidebar.header("3. Generate")
    st.sidebar.caption(
        "Either go tab by tab below (one feature at a time — the safer choice "
        "on the Gemini free tier's 5-requests/minute limit), or generate "
        "everything at once here."
    )
    if st.sidebar.button("🚀 Fetch, Index & Generate Everything", type="primary", use_container_width=True):
        with st.spinner("Fetching + indexing..."):
            if not _ingest():
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

if not already_ingested:
    st.info("👈 Pick a chapter (or upload a PDF), then click **Fetch & Index** in the sidebar to get started.")
    st.stop()

if source_mode == "NCERT Catalog":
    st.caption(f"Working on **Class {class_num} · {subject.title()} · Chapter {chapter}** (`{collection_name}`)")
else:
    st.caption(f"Working on **{book_title}** · pages {upload_start_page}–{upload_end_page} (`{collection_name}`)")

# ── Chapter overview — status cards ─────────────────────────────────────────
overview_features = [
    ("flash", "🃏 Flashcards"),
    ("highlight", "🔍 Highlights"),
    ("notes", "📓 Notes"),
    ("hot", "🔥 Hot Questions"),
    ("formula", "Σ Formula Sheet"),
    ("homework", "📝 Questions"),
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
(
    tab_flash, tab_highlight, tab_notes, tab_hot, tab_formula,
    tab_questions, tab_test, tab_search,
) = st.tabs(
    ["🃏 Flashcards", "🔍 Highlights", "📓 Notes", "🔥 Hot Questions",
     "Σ Formula Sheet", "📝 Questions", "🧪 Test", "🔎 Search"]
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

# ── Questions / Homework (compose → generate → attempt → AI grade) ─────────
with tab_questions:
    st.caption(
        "Compose an exam-style paper by mixing question blocks (by board & type), "
        "optionally scoped to specific sections of the chapter, then attempt it and "
        "get it graded by AI — objective questions matched automatically, subjective "
        "answers marked against a generated mark scheme."
    )

    hw_key    = result_key("homework", collection_name)
    grade_key = f"hw_grade::{collection_name}"

    # 1 ─ Section scoping
    sections = agent.list_sections(class_num, subject, chapter)
    if sections:
        chosen_sections = st.multiselect(
            "Scope to sections (optional — leave empty to use the whole chapter)",
            options=sections, key=f"hw_sections::{collection_name}",
        )
    else:
        chosen_sections = []
        st.caption("No sub-sections were detected for this chapter — questions will span the whole chapter.")

    # 2 ─ Composition builder (add/remove blocks; each = count × style × difficulty)
    st.markdown("**Paper composition** — one row per block of questions. Add or delete rows freely.")
    label_to_code = {t.label: code for code, t in STYLE_TEMPLATES.items()}
    default_blocks = pd.DataFrame([
        {"Questions": 5, "Style": STYLE_TEMPLATES["CBSE_MCQ"].label, "Difficulty": "(paper default)"},
        {"Questions": 3, "Style": STYLE_TEMPLATES["CBSE_SA"].label,  "Difficulty": "(paper default)"},
    ])
    edited = st.data_editor(
        default_blocks,
        num_rows="dynamic",
        use_container_width=True,
        key=f"hw_editor::{collection_name}",
        column_config={
            "Questions": st.column_config.NumberColumn(min_value=1, max_value=30, step=1),
            "Style": st.column_config.SelectboxColumn(options=list(label_to_code.keys()), width="large"),
            "Difficulty": st.column_config.SelectboxColumn(options=["(paper default)", "easy", "medium", "hard"]),
        },
    )

    col_a, col_b = st.columns([1, 1])
    paper_difficulty = col_a.selectbox("Paper difficulty", ["easy", "medium", "hard"], index=1, key=f"hw_diff::{collection_name}")
    total_q = int(pd.to_numeric(edited["Questions"], errors="coerce").fillna(0).sum()) if len(edited) else 0
    col_b.metric("Total questions", total_q)

    if st.button("🚀 Generate paper", type="primary", key=f"hw_gen::{collection_name}"):
        blocks = []
        for _, row in edited.iterrows():
            code  = label_to_code.get(row.get("Style"))
            count = int(row.get("Questions") or 0) if pd.notna(row.get("Questions")) else 0
            if not code or count < 1:
                continue
            diff = row.get("Difficulty")
            blocks.append(PaperBlock(
                count=count, style=code,
                difficulty="" if diff in (None, "(paper default)") else str(diff),
            ))
        if not blocks:
            st.error("Add at least one valid block (a positive question count and a style).")
        else:
            with st.spinner(f"Generating {total_q} questions across {len(blocks)} block(s) with AI..."):
                try:
                    paper = agent.generate_homework(
                        class_num, subject, chapter,
                        blocks=blocks, difficulty=paper_difficulty,
                        sections=chosen_sections or None,
                    )
                    st.session_state[hw_key] = paper
                    st.session_state.pop(grade_key, None)
                except Exception as e:
                    st.error(str(e))

    # 3 ─ Attempt + grade the generated paper
    paper = st.session_state.get(hw_key)
    if paper is not None:
        st.markdown("---")
        render_homework(agent, paper, grade_key)
    else:
        st.info("Set up your composition above and click **Generate paper** to begin.")

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

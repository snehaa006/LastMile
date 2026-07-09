"""
pipeline/homework_styles.py
────────────────────────────
Style-template library for the Homework / Questions feature.

This is EduMind's native port of the "style_templates" table from the
Ascend Now Homework Generator (see its
db/docs/HOMEWORK_GENERATOR_ARCHITECTURE.md, Phase 2). Instead of a Supabase
table it's a plain in-code registry — one entry per exam-format the user can
pick in the composition builder. Each entry carries:

  - board         : the exam board / curriculum it emulates ("" = generic)
  - question_type : a controlled value the generator and grader BOTH branch on
  - prompt_fragment: the instruction text handed to the LLM prompt builder
                     (format, command words, mark-scheme shape, length)

`question_type` is the single most important field: it decides how a
question is generated (which JSON schema) and how it's graded —

  - "mcq" / "fill_blank"           → auto-gradable (exact / fuzzy match)
  - "short_answer" / "structured"
    / "extended_response" / "essay" → LLM-graded against a mark scheme
                                       (a list of individually gradeable points)

The mark-scheme types deliberately instruct the model to emit the mark
scheme as a list of concrete {point, marks} rows, because the grader
(pipeline/homework_grader.py) depends on that shape being present.
"""

from dataclasses import dataclass

# ── Controlled question-type values ──────────────────────────────────────────
# Auto-gradable by exact / fuzzy match.
AUTO_GRADED_TYPES = {"mcq", "fill_blank"}
# Graded by the LLM against the question's mark scheme.
MARK_SCHEME_TYPES = {"short_answer", "structured", "extended_response", "essay"}

ALL_QUESTION_TYPES = AUTO_GRADED_TYPES | MARK_SCHEME_TYPES

# Command words / interrogatives used as a cheap "does this prompt actually
# ask something" check for mark-scheme questions (mirrors Ascend Now Phase 4's
# validateRawQuestion heuristic). A mark-scheme question that contains none of
# these and no "?" almost certainly failed to actually pose a question.
COMMAND_WORDS = (
    "calculate", "explain", "define", "describe", "predict", "derive",
    "evaluate", "compare", "contrast", "discuss", "analyse", "analyze",
    "state", "list", "outline", "justify", "prove", "show", "find",
    "determine", "identify", "name", "give", "suggest", "what", "why",
    "how", "when", "where", "which", "who",
)


@dataclass(frozen=True)
class StyleTemplate:
    """One selectable exam format for the composition builder."""
    code:            str
    board:           str   # "" for board-agnostic generic types
    question_type:   str
    label:           str   # human-facing name shown in the UI dropdown
    prompt_fragment: str


# ── The template library ─────────────────────────────────────────────────────
# CBSE is first-class here (EduMind is NCERT/CBSE-focused), and the same
# international boards Ascend Now ships are kept so the feature is a faithful
# replica that also works for IB/IGCSE/AS-A/AP practice.
_TEMPLATES: list[StyleTemplate] = [
    StyleTemplate(
        code="CBSE_MCQ",
        board="CBSE",
        question_type="mcq",
        label="CBSE — MCQ (1 mark)",
        prompt_fragment=(
            "Write CBSE board-style multiple-choice questions worth 1 mark each. "
            "Exactly four options (A–D), exactly one correct. Options must be "
            "plausible and mutually exclusive, with no 'all/none of the above'. "
            "Keep the stem a single clear sentence."
        ),
    ),
    StyleTemplate(
        code="CBSE_ASSERTION_REASON",
        board="CBSE",
        question_type="mcq",
        label="CBSE — Assertion & Reason (1 mark)",
        prompt_fragment=(
            "Write CBSE Assertion–Reason items worth 1 mark. The stem gives an "
            "Assertion (A) and a Reason (R); the four options are the standard "
            "CBSE set: (A) both A and R true and R explains A, (B) both true but "
            "R does not explain A, (C) A true R false, (D) A false. Put the "
            "Assertion and Reason text inside the prompt."
        ),
    ),
    StyleTemplate(
        code="CBSE_FILL_BLANK",
        board="CBSE",
        question_type="fill_blank",
        label="CBSE — Fill in the blank (1 mark)",
        prompt_fragment=(
            "Write fill-in-the-blank questions worth 1 mark each. Use a single "
            "underscore blank '_____' where the answer goes. The expected answer "
            "must be one short word or phrase; list acceptable synonyms/spellings."
        ),
    ),
    StyleTemplate(
        code="CBSE_VSA",
        board="CBSE",
        question_type="short_answer",
        label="CBSE — Very Short Answer (1–2 marks)",
        prompt_fragment=(
            "Write CBSE Very-Short-Answer questions worth 1–2 marks. One or two "
            "sentences of expected answer. Use a command word (Define / State / "
            "Name / Give). Provide a mark scheme as a list of {point, marks} rows "
            "that sum to the question's marks — each point a single checkable fact."
        ),
    ),
    StyleTemplate(
        code="CBSE_SA",
        board="CBSE",
        question_type="structured",
        label="CBSE — Short Answer (3 marks)",
        prompt_fragment=(
            "Write CBSE Short-Answer questions worth 3 marks. May use labelled "
            "sub-parts (a)/(b). Use command words (Explain / Describe / "
            "Distinguish / Calculate). Provide a mark scheme as a list of "
            "{point, marks} rows summing to 3, each point independently gradeable."
        ),
    ),
    StyleTemplate(
        code="CBSE_LA",
        board="CBSE",
        question_type="extended_response",
        label="CBSE — Long Answer (5 marks)",
        prompt_fragment=(
            "Write CBSE Long-Answer questions worth 5 marks, typically multi-part "
            "(a)/(b)/(c), testing explanation plus application/derivation. Provide "
            "a mark scheme as a list of {point, marks} rows summing to 5, each a "
            "concrete gradeable point (definition mark, working mark, etc.)."
        ),
    ),
    StyleTemplate(
        code="CBSE_CASE_STUDY",
        board="CBSE",
        question_type="structured",
        label="CBSE — Case/Source-based (4 marks)",
        prompt_fragment=(
            "Write a CBSE case-study / source-based question worth 4 marks: a "
            "short passage or scenario drawn from the chapter content, followed "
            "by 2–3 labelled sub-questions (a)/(b)/(c) referring to it. Provide a "
            "mark scheme as {point, marks} rows summing to 4."
        ),
    ),
    StyleTemplate(
        code="IGCSE_CORE",
        board="IGCSE",
        question_type="structured",
        label="IGCSE — Core structured",
        prompt_fragment=(
            "Write IGCSE Core-tier structured questions with clear command words "
            "and a marks-in-brackets convention per sub-part. Provide a mark "
            "scheme as a list of {point, marks} rows, one row per creditable point."
        ),
    ),
    StyleTemplate(
        code="IGCSE_EXTENDED",
        board="IGCSE",
        question_type="structured",
        label="IGCSE — Extended structured",
        prompt_fragment=(
            "Write IGCSE Extended-tier structured questions — more demanding "
            "application and analysis than Core. Labelled sub-parts with marks in "
            "brackets. Provide a mark scheme as {point, marks} rows."
        ),
    ),
    StyleTemplate(
        code="IBDP_SECTION_A",
        board="IBDP",
        question_type="structured",
        label="IBDP — Section A structured",
        prompt_fragment=(
            "Write IBDP Paper-2 Section-A style structured questions with data/"
            "scenario stems and command terms (State / Outline / Explain / "
            "Deduce). Provide a mark scheme as {point, marks} rows keyed to IB "
            "command-term expectations."
        ),
    ),
    StyleTemplate(
        code="IBDP_SECTION_B",
        board="IBDP",
        question_type="extended_response",
        label="IBDP — Section B extended response",
        prompt_fragment=(
            "Write an IBDP extended-response question using higher command terms "
            "(Discuss / Evaluate / To what extent). Provide a mark scheme as a "
            "list of {point, marks} rows covering the required arguments, worth "
            "8–15 marks in total."
        ),
    ),
    StyleTemplate(
        code="AS_A_LEVEL",
        board="AS & A Levels",
        question_type="structured",
        label="AS/A Level — structured",
        prompt_fragment=(
            "Write AS/A-Level structured questions with marks in brackets per "
            "part and precise command words. Provide a mark scheme as {point, "
            "marks} rows, each an examinable point."
        ),
    ),
    StyleTemplate(
        code="AP_MCQ",
        board="AP",
        question_type="mcq",
        label="AP — Multiple choice",
        prompt_fragment=(
            "Write AP-style multiple-choice questions: four options (A–D), one "
            "correct, application-oriented stems rather than pure recall."
        ),
    ),
    StyleTemplate(
        code="AP_FRQ",
        board="AP",
        question_type="extended_response",
        label="AP — Free response (FRQ)",
        prompt_fragment=(
            "Write an AP Free-Response Question with labelled parts (a)/(b)/(c). "
            "Provide a mark scheme as a list of {point, marks} rows, one row per "
            "scoring point in the AP rubric."
        ),
    ),
    # ── Generic, board-agnostic types ────────────────────────────────────────
    StyleTemplate(
        code="MCQ",
        board="",
        question_type="mcq",
        label="Generic — MCQ",
        prompt_fragment=(
            "Write clear multiple-choice questions: exactly four options, one "
            "correct, distractors plausible."
        ),
    ),
    StyleTemplate(
        code="FILL_IN_BLANK",
        board="",
        question_type="fill_blank",
        label="Generic — Fill in the blank",
        prompt_fragment=(
            "Write fill-in-the-blank questions with a single '_____' blank; the "
            "expected answer is one short word or phrase; list acceptable variants."
        ),
    ),
    StyleTemplate(
        code="SHORT_ANSWER",
        board="",
        question_type="short_answer",
        label="Generic — Short answer",
        prompt_fragment=(
            "Write short-answer questions answerable in 1–3 sentences, phrased "
            "with a command word. Provide a mark scheme as {point, marks} rows."
        ),
    ),
    StyleTemplate(
        code="SUBJECTIVE",
        board="",
        question_type="essay",
        label="Generic — Essay / subjective",
        prompt_fragment=(
            "Write an essay-style subjective question. Provide the mark scheme as "
            "a list of individually gradeable {point, marks} rows (thesis, each "
            "required argument, evaluation) — NOT as free text — because the "
            "grader checks the student's answer against each point separately."
        ),
    ),
]

# code → StyleTemplate, and preserved order for the UI.
STYLE_TEMPLATES: dict[str, StyleTemplate] = {t.code: t for t in _TEMPLATES}
STYLE_CODES: list[str] = [t.code for t in _TEMPLATES]


def get_template(code: str) -> StyleTemplate:
    """Look up a template by code, raising a clear error on an unknown code."""
    try:
        return STYLE_TEMPLATES[code]
    except KeyError as exc:
        raise KeyError(
            f"Unknown style template code {code!r}. "
            f"Known codes: {', '.join(STYLE_CODES)}"
        ) from exc


def default_marks_for(question_type: str) -> int:
    """A sensible default mark value per question type, used when the model
    omits `marks` on a generated question."""
    return {
        "mcq": 1,
        "fill_blank": 1,
        "short_answer": 2,
        "structured": 3,
        "extended_response": 5,
        "essay": 10,
    }.get(question_type, 1)

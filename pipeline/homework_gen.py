"""
pipeline/homework_gen.py
─────────────────────────
Homework paper generator — EduMind's native port of the Ascend Now Homework
Generator's Phase 4 generation service (see that repo's
db/docs/HOMEWORK_GENERATOR_ARCHITECTURE.md).

Unlike the personalized `test_builder` (which is student-profile driven and
PYQ-blended), this builds an exam-style **paper** from a *composition* the
user mixes together: a list of blocks, each `{count, style, difficulty}`,
optionally scoped to specific sections/topics of the chapter. It mirrors the
Ascend Now flow — pick content source (here: an indexed chapter) → pick paper
composition (mix-and-match question blocks by style/board) → generate — minus
the teacher/student plumbing.

Same generator shape as every other pipeline module:
    retrieve relevant chunks (section-scoped if asked) →
    prompt the LLM per block with a strict JSON-only format keyed to the
        block's question_type →
    validate count + required fields per question (retry once on failure) →
    parse into typed dataclasses.

Blocks are generated concurrently (one LLM call each), like
orchestrator.generate_all().

Usage:
    gen = HomeworkGenerator(store=shared_store)
    paper = gen.generate(
        collection_name="class10_science_ch01",
        subject="science", chapter=1, class_num=10,
        blocks=[PaperBlock(count=5, style="CBSE_MCQ"),
                PaperBlock(count=3, style="CBSE_SA")],
        difficulty="medium",
        sections=["Chemical Reactions", "Balancing Equations"],   # optional
    )
"""

import json
import re
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional

from rich.console import Console

from pipeline.homework_styles import (
    COMMAND_WORDS,
    MARK_SCHEME_TYPES,
    default_marks_for,
    get_template,
)
from pipeline.llm_client import LLMClient
from pipeline.vector_store import VectorStore

console = Console()

# Generic seed queries used to pull a spread of chapter content when no
# specific sections are chosen (same idea as hot_questions_gen's seeds).
_WHOLE_CHAPTER_SEEDS = [
    "key definition and concept",
    "important process or mechanism",
    "worked example or numerical",
    "diagram, structure or classification",
    "compare, contrast or application",
]


@dataclass
class MarkSchemePoint:
    """One creditable point in a subjective question's mark scheme."""
    point: str
    marks: int


@dataclass
class HomeworkQuestion:
    """
    A single generated question. Only the fields relevant to `type` are
    populated:
      - mcq        : options + correct_index
      - fill_blank : expected_answer + acceptable_answers
      - subjective : mark_scheme (+ model_answer as a reference)
    """
    id:                 str
    type:               str
    prompt:             str
    marks:              int
    style:              str = ""      # style-template code it came from
    topic:              str = ""
    # mcq
    options:            list[str] = field(default_factory=list)
    correct_index:      int = 0
    # fill_blank
    expected_answer:    str = ""
    acceptable_answers: list[str] = field(default_factory=list)
    # mark-scheme (subjective) types
    mark_scheme:        list[MarkSchemePoint] = field(default_factory=list)
    model_answer:       str = ""


@dataclass
class PaperBlock:
    """One row of the composition builder: N questions in a given style."""
    count:      int
    style:      str          # a StyleTemplate code, e.g. "CBSE_MCQ"
    difficulty: str = ""     # optional per-block override of paper difficulty


@dataclass
class HomeworkPaper:
    """A generated homework paper."""
    paper_id:        str
    collection_name: str
    subject:         str
    chapter:         int
    class_num:       int
    difficulty:      str
    sections:        list[str]           = field(default_factory=list)
    questions:       list[HomeworkQuestion] = field(default_factory=list)
    errors:          list[str]           = field(default_factory=list)

    @property
    def total_marks(self) -> int:
        return sum(q.marks for q in self.questions)

    @property
    def total_questions(self) -> int:
        return len(self.questions)


class HomeworkGenerator:

    # number of validate/generate attempts per block before giving up
    MAX_ATTEMPTS = 2

    def __init__(self, store: Optional[VectorStore] = None):
        self._client = LLMClient()
        self._store  = store or VectorStore()

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def generate(
        self,
        collection_name: str,
        subject:         str,
        chapter:         int,
        class_num:       int,
        blocks:          list[PaperBlock],
        difficulty:      str = "medium",
        sections:        Optional[list[str]] = None,
    ) -> HomeworkPaper:
        """
        Builds a homework paper from the given composition. The chapter must
        already be indexed. Each block is generated with its own LLM call;
        blocks run concurrently. A block that fails after retries contributes
        no questions and an entry in `paper.errors` rather than failing the
        whole paper (mirrors Ascend Now's per-block resilience).
        """
        if not blocks:
            raise ValueError("At least one composition block is required.")

        sections = [s for s in (sections or []) if s.strip()]
        console.print(
            f"[blue]📝 Generating homework:[/blue] {collection_name} "
            f"({len(blocks)} blocks, difficulty={difficulty}"
            + (f", sections={sections}" if sections else ", whole chapter")
            + ")"
        )

        # Resolve source content once — shared by every block. Section-scoped
        # if sections were chosen, otherwise a spread across the chapter.
        context = self._resolve_context(collection_name, sections)

        paper = HomeworkPaper(
            paper_id        = f"hw_{collection_name}_{uuid.uuid4().hex[:8]}",
            collection_name = collection_name,
            subject         = subject,
            chapter         = chapter,
            class_num       = class_num,
            difficulty      = difficulty,
            sections        = sections,
        )

        if not context.strip():
            paper.errors.append(
                "No indexed content found for this chapter/section selection."
            )
            return paper

        # Generate every block concurrently.
        results: dict[int, tuple[list[HomeworkQuestion], Optional[str]]] = {}
        with ThreadPoolExecutor(max_workers=max(1, len(blocks))) as pool:
            future_to_idx = {
                pool.submit(
                    self._generate_block,
                    block, context, subject, chapter, class_num,
                    block.difficulty or difficulty,
                    sections,
                ): idx
                for idx, block in enumerate(blocks)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = (future.result(), None)
                except Exception as e:  # noqa: BLE001 — one block never sinks the paper
                    console.print(f"[red]✗ block {idx} failed:[/red] {e}")
                    results[idx] = ([], str(e))

        # Reassemble in the user's block order and renumber ids sequentially.
        for idx, block in enumerate(blocks):
            questions, err = results.get(idx, ([], "block did not run"))
            if err:
                paper.errors.append(f"Block {idx + 1} ({block.style}): {err}")
            paper.questions.extend(questions)

        for n, q in enumerate(paper.questions, 1):
            q.id = f"q{n}"

        console.print(
            f"[green]✓ Homework paper:[/green] {paper.total_questions} questions, "
            f"{paper.total_marks} marks"
            + (f" ({len(paper.errors)} block error(s))" if paper.errors else "")
        )
        return paper

    # ──────────────────────────────────────────────────────────────────────────
    # Source resolution (RAG retrieval, section-scoped)
    # ──────────────────────────────────────────────────────────────────────────

    def _resolve_context(
        self, collection_name: str, sections: list[str]
    ) -> str:
        """
        Pulls a de-duplicated spread of chapter content to ground generation.
        When `sections` are given, each section title seeds its own retrieval
        so questions are drawn from just those parts of the chapter; otherwise
        a fixed set of generic seed queries pulls a broad spread.
        """
        seeds = sections if sections else _WHOLE_CHAPTER_SEEDS
        seen: set[str] = set()
        chunks: list[str] = []
        # More chunks per seed when the scope is narrow (few sections) so a
        # single-section paper still has enough grounding content.
        per_seed = 6 if sections and len(sections) <= 2 else 4
        for seed in seeds:
            try:
                for r in self._store.query(seed, collection_name, top_k=per_seed):
                    key = r.chunk_id or r.text[:60]
                    if key not in seen:
                        seen.add(key)
                        chunks.append(r.text)
            except Exception as e:  # noqa: BLE001
                console.print(f"[yellow]⚠ retrieval failed for {seed!r}:[/yellow] {e}")
        return "\n\n---\n\n".join(chunks)[:6000]

    # ──────────────────────────────────────────────────────────────────────────
    # Per-block generation
    # ──────────────────────────────────────────────────────────────────────────

    def _generate_block(
        self,
        block:      PaperBlock,
        context:    str,
        subject:    str,
        chapter:    int,
        class_num:  int,
        difficulty: str,
        sections:   list[str],
    ) -> list[HomeworkQuestion]:
        """Generates one block's questions, validating and retrying once."""
        template = get_template(block.style)
        feedback = ""    # validation failure fed back into the retry prompt

        for attempt in range(1, self.MAX_ATTEMPTS + 1):
            prompt = self._build_prompt(
                template, block.count, context, subject, chapter,
                class_num, difficulty, sections, feedback,
            )
            raw = self._client.complete(prompt, max_tokens=3072)
            items = self._parse_json_array(raw)
            questions, problem = self._materialize(items, template, block.count)

            if problem is None:
                return questions

            console.print(
                f"[yellow]⚠ {block.style} attempt {attempt} invalid:[/yellow] {problem}"
            )
            feedback = (
                f"\nYour previous response was rejected: {problem}. "
                f"Return exactly {block.count} valid question object(s) that fix this."
            )

        raise ValueError(
            f"could not produce {block.count} valid '{template.question_type}' "
            f"question(s) after {self.MAX_ATTEMPTS} attempts"
        )

    def _build_prompt(
        self,
        template,
        count:      int,
        context:    str,
        subject:    str,
        chapter:    int,
        class_num:  int,
        difficulty: str,
        sections:   list[str],
        feedback:   str,
    ) -> str:
        qtype = template.question_type
        board = f" in the {template.board} exam style" if template.board else ""
        scope = (
            f"Draw the questions ONLY from these sections/topics of the chapter: "
            f"{', '.join(sections)}. Ignore unrelated parts of the content.\n"
            if sections else ""
        )
        who = (
            f"Class {class_num} {subject.title()}" if class_num != 0
            else f"{subject.replace('_', ' ').title()}"
        )

        header = (
            f"You are an expert {who} examiner setting Chapter {chapter} questions{board}.\n"
            f"{template.prompt_fragment}\n\n"
            f"Generate exactly {count} question(s). Difficulty: {difficulty}.\n"
            f"{scope}"
            "Every question must be answerable purely from the source content "
            "below — do not invent facts not present in it.\n"
        )

        schema = self._schema_for(qtype)
        rules = self._rules_for(qtype)

        return (
            f"{header}\n{rules}\n"
            "Return ONLY a JSON array of question objects — no prose, no markdown "
            f"fences. Each object must have this shape:\n{schema}\n"
            f"{feedback}\n\n"
            f"Source content:\n{context}\n"
        )

    def _schema_for(self, qtype: str) -> str:
        if qtype == "mcq":
            return (
                '[{"prompt": "...", "options": ["A text", "B text", "C text", '
                '"D text"], "correct_index": 0, "marks": 1, "topic": "..."}]'
            )
        if qtype == "fill_blank":
            return (
                '[{"prompt": "... _____ ...", "expected_answer": "...", '
                '"acceptable_answers": ["...", "..."], "marks": 1, "topic": "..."}]'
            )
        # mark-scheme (subjective) types
        return (
            '[{"prompt": "...", "marks": 3, "topic": "...", '
            '"model_answer": "a concise ideal answer", '
            '"mark_scheme": [{"point": "specific creditable point", "marks": 1}, '
            '{"point": "...", "marks": 2}]}]'
        )

    def _rules_for(self, qtype: str) -> str:
        if qtype == "mcq":
            return (
                "Rules:\n"
                "- 'options' has exactly 4 entries; do NOT prefix them with A./B./etc.\n"
                "- 'correct_index' is the 0-based index of the single correct option.\n"
            )
        if qtype == "fill_blank":
            return (
                "Rules:\n"
                "- Put a single '_____' blank in the prompt where the answer goes.\n"
                "- 'expected_answer' is the canonical short answer; "
                "'acceptable_answers' lists equivalent spellings/synonyms.\n"
            )
        return (
            "Rules:\n"
            "- The prompt must actually ASK something (a question mark or a "
            "command word like Explain/Calculate/Describe); use labelled "
            "sub-parts (a)/(b)/(c) for multi-part questions.\n"
            "- Every mark-scheme point must correspond to something the prompt "
            "explicitly asks for. The 'marks' of the mark-scheme points must sum "
            "to the question's 'marks'.\n"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Parsing & validation
    # ──────────────────────────────────────────────────────────────────────────

    def _parse_json_array(self, raw: str) -> list:
        """Strips markdown fences and parses a JSON array, tolerating the model
        wrapping it in prose by grabbing the outermost [ ... ]."""
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        try:
            data = json.loads(clean)
        except json.JSONDecodeError:
            match = re.search(r"\[.*\]", clean, re.DOTALL)
            if not match:
                return []
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                return []
        return data if isinstance(data, list) else []

    def _materialize(
        self, items: list, template, expected_count: int
    ) -> tuple[list[HomeworkQuestion], Optional[str]]:
        """
        Turns raw dicts into HomeworkQuestion objects, validating per type.
        Returns (questions, None) on success or ([], reason) on a validation
        failure that should trigger a retry.
        """
        if not items:
            return [], "no parseable question objects were returned"
        if len(items) < expected_count:
            return [], f"expected {expected_count} questions, got {len(items)}"

        qtype = template.question_type
        questions: list[HomeworkQuestion] = []
        for raw in items[:expected_count]:
            if not isinstance(raw, dict):
                return [], "a question was not a JSON object"
            q, problem = self._one_question(raw, template, qtype)
            if problem is not None:
                return [], problem
            questions.append(q)
        return questions, None

    def _one_question(
        self, raw: dict, template, qtype: str
    ) -> tuple[Optional[HomeworkQuestion], Optional[str]]:
        prompt = str(raw.get("prompt", "")).strip()
        if not prompt:
            return None, "a question had an empty prompt"

        marks = self._as_int(raw.get("marks"), default_marks_for(qtype))
        topic = str(raw.get("topic", "")).strip()
        base = dict(id="", type=qtype, prompt=prompt, marks=max(1, marks),
                    style=template.code, topic=topic)

        if qtype == "mcq":
            options = [str(o).strip() for o in raw.get("options", []) if str(o).strip()]
            if len(options) != 4:
                return None, "an MCQ did not have exactly 4 options"
            ci = self._as_int(raw.get("correct_index"), 0)
            if not (0 <= ci < 4):
                return None, "an MCQ had an out-of-range correct_index"
            return HomeworkQuestion(options=options, correct_index=ci, **base), None

        if qtype == "fill_blank":
            expected = str(raw.get("expected_answer", "")).strip()
            if not expected:
                return None, "a fill-in-the-blank had no expected_answer"
            accept = [str(a).strip() for a in raw.get("acceptable_answers", []) if str(a).strip()]
            return HomeworkQuestion(
                expected_answer=expected, acceptable_answers=accept, **base
            ), None

        # mark-scheme (subjective) types
        if qtype in MARK_SCHEME_TYPES:
            if not self._prompt_actually_asks(prompt):
                return None, "a subjective prompt didn't actually ask a question"
            raw_scheme = raw.get("mark_scheme", [])
            scheme: list[MarkSchemePoint] = []
            for p in raw_scheme:
                if not isinstance(p, dict):
                    continue
                text = str(p.get("point", "")).strip()
                if not text:
                    continue
                scheme.append(MarkSchemePoint(point=text, marks=max(1, self._as_int(p.get("marks"), 1))))
            if not scheme:
                return None, "a subjective question had an empty mark scheme"
            # Trust the mark scheme's own sum as the question's marks so grading
            # can't exceed it.
            base["marks"] = sum(p.marks for p in scheme)
            return HomeworkQuestion(
                mark_scheme=scheme,
                model_answer=str(raw.get("model_answer", "")).strip(),
                **base,
            ), None

        return None, f"unsupported question_type {qtype!r}"

    def _prompt_actually_asks(self, prompt: str) -> bool:
        """Cheap presence check that a mark-scheme prompt actually poses a
        question (mirrors Ascend Now Phase 4's validateRawQuestion guard).
        Matches command words on word boundaries so e.g. 'statement' doesn't
        count as the command word 'state'."""
        if "?" in prompt:
            return True
        words = set(re.findall(r"[a-z]+", prompt.lower()))
        return any(w in words for w in COMMAND_WORDS)

    @staticmethod
    def _as_int(value, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

"""
pipeline/homework_grader.py
────────────────────────────
Homework grading service — EduMind's native port of the Ascend Now Homework
Generator's Phase 7 grading service (see that repo's
db/docs/HOMEWORK_GENERATOR_ARCHITECTURE.md).

Given a generated `HomeworkPaper` and a student's answers, it grades every
question:

  - MCQ         → exact option-index match           (graded_by = "auto")
  - fill_blank  → normalized string match against the
                  expected + acceptable answers        (graded_by = "auto")
  - subjective  → the LLM scores the answer against
    (short_answer/  each mark-scheme point, in ONE batched call for the whole
     structured/    paper (not one call per question, to stay inside the
     extended_/     Gemini free tier's per-minute cap — same fix as Ascend
     essay)         Now Phase 7)                        (graded_by = "ai")

Resilience mirrors Ascend Now: if the batched AI call fails or omits a
question, that subjective question comes back `awarded = 0` with a "pending
review" note and `ai_incomplete = True` on the result — objective marks are
still returned rather than failing the whole grade.

Usage:
    grader = HomeworkGrader()
    result = grader.grade(paper, answers={"q1": 2, "q2": "photosynthesis", ...})
    print(result.total_marks, "/", result.max_marks)
"""

import json
import re
from dataclasses import dataclass, field
from typing import Optional, Union

from rich.console import Console

from pipeline.homework_gen import HomeworkPaper, HomeworkQuestion
from pipeline.homework_styles import AUTO_GRADED_TYPES, MARK_SCHEME_TYPES
from pipeline.llm_client import LLMClient

console = Console()

# An answer is an option index (mcq) or a text string (everything else).
Answer = Union[int, str, None]


@dataclass
class PointGrade:
    """Awarded vs. max marks for one mark-scheme point."""
    point:   str
    max:     int
    awarded: int


@dataclass
class QuestionGrade:
    """Grade for a single question."""
    question_id: str
    type:        str
    awarded:     int
    max:         int
    graded_by:   str            # "auto" | "ai" | "pending"
    feedback:    str = ""
    per_point:   list[PointGrade] = field(default_factory=list)


@dataclass
class PaperGrade:
    """Complete grade for a submitted paper."""
    paper_id:      str
    per_question:  list[QuestionGrade] = field(default_factory=list)
    ai_incomplete: bool = False    # some subjective Q couldn't be AI-graded

    @property
    def total_marks(self) -> int:
        return sum(g.awarded for g in self.per_question)

    @property
    def max_marks(self) -> int:
        return sum(g.max for g in self.per_question)

    @property
    def percentage(self) -> float:
        return round(100 * self.total_marks / self.max_marks, 1) if self.max_marks else 0.0


class HomeworkGrader:

    def __init__(self):
        self._client = LLMClient()

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def grade(
        self, paper: HomeworkPaper, answers: dict[str, Answer]
    ) -> PaperGrade:
        """
        Grades `paper` against `answers` (keyed by question id). Objective
        questions are graded locally; all subjective questions are graded in a
        single LLM call. A missing answer scores 0 but is still graded.
        """
        console.print(f"[blue]🧮 Grading paper:[/blue] {paper.paper_id}")
        result = PaperGrade(paper_id=paper.paper_id)

        subjective: list[HomeworkQuestion] = []
        for q in paper.questions:
            ans = answers.get(q.id)
            if q.type in AUTO_GRADED_TYPES:
                result.per_question.append(self._grade_objective(q, ans))
            elif q.type in MARK_SCHEME_TYPES:
                subjective.append(q)
            else:
                result.per_question.append(
                    QuestionGrade(q.id, q.type, 0, q.marks, "pending",
                                  "Unsupported question type — needs manual review.")
                )

        if subjective:
            ai_grades, incomplete = self._grade_subjective_batch(subjective, answers)
            result.per_question.extend(ai_grades)
            result.ai_incomplete = incomplete

        # Preserve the paper's question order in the output.
        order = {q.id: i for i, q in enumerate(paper.questions)}
        result.per_question.sort(key=lambda g: order.get(g.question_id, 0))

        console.print(
            f"[green]✓ Graded:[/green] {result.total_marks}/{result.max_marks} "
            f"({result.percentage}%)"
            + (" [yellow](some AI grades pending review)[/yellow]" if result.ai_incomplete else "")
        )
        return result

    # ──────────────────────────────────────────────────────────────────────────
    # Objective grading (local, no LLM)
    # ──────────────────────────────────────────────────────────────────────────

    def _grade_objective(
        self, q: HomeworkQuestion, answer: Answer
    ) -> QuestionGrade:
        if q.type == "mcq":
            chosen = self._as_int(answer, -1)
            correct = chosen == q.correct_index
            picked = (
                q.options[chosen] if 0 <= chosen < len(q.options) else "— (no answer)"
            )
            right = q.options[q.correct_index] if 0 <= q.correct_index < len(q.options) else "?"
            fb = "Correct." if correct else f"Incorrect. You chose “{picked}”; correct answer: “{right}”."
            return QuestionGrade(q.id, q.type, q.marks if correct else 0, q.marks, "auto", fb)

        # fill_blank
        given = self._normalize(answer if isinstance(answer, str) else "")
        candidates = {self._normalize(q.expected_answer)} | {
            self._normalize(a) for a in q.acceptable_answers
        }
        candidates.discard("")
        correct = given != "" and given in candidates
        fb = "Correct." if correct else f"Incorrect. Expected: “{q.expected_answer}”."
        return QuestionGrade(q.id, q.type, q.marks if correct else 0, q.marks, "auto", fb)

    def _normalize(self, text: str) -> str:
        """trim / lowercase / collapse whitespace / strip trailing punctuation —
        same normalization Ascend Now's fill-blank grader uses."""
        t = (text or "").strip().lower()
        t = re.sub(r"\s+", " ", t)
        t = re.sub(r"[.!?,;:]+$", "", t)
        return t.strip()

    # ──────────────────────────────────────────────────────────────────────────
    # Subjective grading (one batched LLM call for the whole paper)
    # ──────────────────────────────────────────────────────────────────────────

    def _grade_subjective_batch(
        self, questions: list[HomeworkQuestion], answers: dict[str, Answer]
    ) -> tuple[list[QuestionGrade], bool]:
        """Grades every subjective question in a single call. On any failure,
        falls back to per-question 'pending review' zeros (incomplete=True)."""
        pending = [
            QuestionGrade(
                q.id, q.type, 0, q.marks, "pending",
                "Awaiting review — automatic grading was unavailable.",
                [PointGrade(p.point, p.marks, 0) for p in q.mark_scheme],
            )
            for q in questions
        ]

        prompt = self._build_grading_prompt(questions, answers)
        try:
            raw = self._client.complete(prompt, max_tokens=3072)
            parsed = self._parse_grades(raw)
        except Exception as e:  # noqa: BLE001
            console.print(f"[red]✗ subjective grading failed:[/red] {e}")
            return pending, True

        grades: list[QuestionGrade] = []
        incomplete = False
        for q in questions:
            entry = parsed.get(q.id)
            if entry is None:
                grades.append(next(p for p in pending if p.question_id == q.id))
                incomplete = True
                continue
            grades.append(self._assemble_grade(q, entry))
        return grades, incomplete

    def _build_grading_prompt(
        self, questions: list[HomeworkQuestion], answers: dict[str, Answer]
    ) -> str:
        blocks = []
        for q in questions:
            student = answers.get(q.id)
            student_text = student if isinstance(student, str) and student.strip() else "(no answer given)"
            scheme = "\n".join(
                f'    - point {i}: "{p.point}" (max {p.marks} marks)'
                for i, p in enumerate(q.mark_scheme)
            )
            blocks.append(
                f'Question id "{q.id}" (worth {q.marks} marks):\n'
                f"  Prompt: {q.prompt}\n"
                f"  Mark scheme:\n{scheme}\n"
                f"  Student answer: {student_text}\n"
            )
        joined = "\n".join(blocks)

        return (
            "You are a strict but fair examiner. Grade each student answer "
            "against ITS OWN mark scheme, point by point. Award each point's "
            "marks fully, partially (an integer), or zero based on whether the "
            "student's answer earns it. Never award more than a point's max, and "
            "the per-point awards must not exceed the question's total marks.\n\n"
            f"{joined}\n"
            "Return ONLY a JSON object keyed by question id — no prose, no "
            "markdown fences — in this shape:\n"
            '{\n'
            '  "q1": {"per_point": [1, 0, 2], "feedback": "short, specific feedback"},\n'
            '  "q2": {"per_point": [2], "feedback": "..."}\n'
            "}\n"
            "The 'per_point' array must have exactly one integer per mark-scheme "
            "point, in order."
        )

    def _assemble_grade(self, q: HomeworkQuestion, entry: dict) -> QuestionGrade:
        raw_points = entry.get("per_point", [])
        per_point: list[PointGrade] = []
        for i, p in enumerate(q.mark_scheme):
            awarded = self._as_int(raw_points[i], 0) if i < len(raw_points) else 0
            awarded = max(0, min(awarded, p.marks))     # clamp to [0, point max]
            per_point.append(PointGrade(p.point, p.marks, awarded))

        total = min(sum(pp.awarded for pp in per_point), q.marks)   # clamp to Q max
        feedback = str(entry.get("feedback", "")).strip() or "Graded against the mark scheme."
        return QuestionGrade(q.id, q.type, total, q.marks, "ai", feedback, per_point)

    def _parse_grades(self, raw: str) -> dict[str, dict]:
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        try:
            data = json.loads(clean)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", clean, re.DOTALL)
            if not match:
                return {}
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                return {}
        return data if isinstance(data, dict) else {}

    @staticmethod
    def _as_int(value, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

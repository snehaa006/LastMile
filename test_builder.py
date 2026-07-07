"""
pipeline/test_builder.py
─────────────────────────
Builds personalized tests for students by:
  1. Pulling PYQs related to their weak topics from the vector store
  2. Generating AI questions for gaps not covered by PYQs
  3. Mixing difficulty levels based on the student's performance profile

Usage:
    builder = TestBuilder()
    test = builder.build(
        student_profile=profile,
        collection_name="class10_science_ch03",
        num_questions=10,
    )
"""

import json
import re
from dataclasses import dataclass, field

import anthropic
from rich.console import Console

from config import ANTHROPIC_API_KEY, LLM_MODEL
from pipeline.vector_store import VectorStore

console = Console()


@dataclass
class StudentProfile:
    """
    Lightweight model of a student's performance.
    In production this is populated from a PostgreSQL DB.
    """
    student_id:   str
    class_num:    int
    subject:      str
    weak_topics:  list[str]   # topics with < 60% accuracy
    strong_topics: list[str]  # topics with > 80% accuracy
    avg_accuracy: float = 0.5 # 0.0 – 1.0


@dataclass
class TestQuestion:
    """A single question in a generated test."""
    question_id:   str
    question:      str
    answer:        str
    marks:         int
    difficulty:    str   # easy / medium / hard
    topic:         str
    source:        str   # "pyq_YYYY" or "ai_generated"
    hint:          str = ""


@dataclass
class PersonalizedTest:
    """A complete generated test."""
    test_id:     str
    student_id:  str
    subject:     str
    class_num:   int
    chapter:     int
    questions:   list[TestQuestion] = field(default_factory=list)
    total_marks: int = 0
    weak_focus:  list[str] = field(default_factory=list)


class TestBuilder:

    def __init__(self):
        self._client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        self._store  = VectorStore()

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def build(
        self,
        student_profile: StudentProfile,
        collection_name: str,
        chapter: int,
        num_questions: int = 10,
    ) -> PersonalizedTest:
        """
        Builds a personalized test for a student.

        Strategy:
        - 60% questions from weak topics
        - 30% questions from medium topics
        - 10% from strong topics (to maintain confidence)
        - Mix of PYQs and AI-generated questions
        - Difficulty calibrated to student's avg accuracy

        Args:
            student_profile  : StudentProfile object
            collection_name  : ChromaDB collection for this chapter
            chapter          : chapter number
            num_questions    : total questions in the test

        Returns:
            PersonalizedTest
        """
        console.print(
            f"[blue]📝 Building test:[/blue] "
            f"{student_profile.subject} Ch.{chapter} "
            f"for student {student_profile.student_id}"
        )

        questions = []

        # 1. PYQ-based questions for weak topics
        pyq_questions = self._get_pyq_questions(
            weak_topics = student_profile.weak_topics,
            subject     = student_profile.subject,
            class_num   = student_profile.class_num,
            n           = int(num_questions * 0.5),
        )
        questions.extend(pyq_questions)

        # 2. AI-generated questions to fill the remaining slots
        remaining = num_questions - len(questions)
        if remaining > 0:
            ai_questions = self._generate_ai_questions(
                student_profile  = student_profile,
                collection_name  = collection_name,
                chapter          = chapter,
                n                = remaining,
            )
            questions.extend(ai_questions)

        total_marks = sum(q.marks for q in questions)

        test = PersonalizedTest(
            test_id     = f"test_{student_profile.student_id}_{collection_name}",
            student_id  = student_profile.student_id,
            subject     = student_profile.subject,
            class_num   = student_profile.class_num,
            chapter     = chapter,
            questions   = questions[:num_questions],
            total_marks = total_marks,
            weak_focus  = student_profile.weak_topics[:5],
        )

        console.print(
            f"[green]✓ Test built:[/green] {len(test.questions)} questions, "
            f"{test.total_marks} marks | Weak focus: {', '.join(test.weak_focus[:3])}"
        )
        return test

    def record_attempt(
        self,
        student_id:  str,
        question_id: str,
        topic:       str,
        is_correct:  bool,
    ) -> dict:
        """
        Records a student's answer attempt.
        In production: writes to PostgreSQL and recalculates weak topics.
        Here: returns an updated mock profile entry.
        """
        return {
            "student_id":  student_id,
            "question_id": question_id,
            "topic":       topic,
            "is_correct":  is_correct,
            "message":     "Recorded. Profile will update after 5 attempts.",
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Internal Helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _get_pyq_questions(
        self,
        weak_topics: list[str],
        subject: str,
        class_num: int,
        n: int,
    ) -> list[TestQuestion]:
        """Retrieves PYQs from vector store matching weak topics."""
        pyq_metas = []
        seen = set()

        for topic in weak_topics:
            results = self._store.query_pyqs(topic, subject, class_num, top_k=3)
            for r in results:
                q = r.get("question", "")
                if q and q not in seen:
                    seen.add(q)
                    pyq_metas.append(r)
                    if len(pyq_metas) >= n:
                        break
            if len(pyq_metas) >= n:
                break

        questions = []
        for i, meta in enumerate(pyq_metas[:n]):
            year = meta.get("year", "")
            questions.append(
                TestQuestion(
                    question_id = f"pyq_{year}_{i}",
                    question    = meta.get("question", ""),
                    answer      = meta.get("answer", ""),
                    marks       = meta.get("marks", 1),
                    difficulty  = self._marks_to_difficulty(meta.get("marks", 1)),
                    topic       = meta.get("chapter", ""),
                    source      = f"pyq_{year}",
                )
            )
        return questions

    def _generate_ai_questions(
        self,
        student_profile: StudentProfile,
        collection_name: str,
        chapter: int,
        n: int,
    ) -> list[TestQuestion]:
        """Generates AI questions targeting the student's weak topics."""
        # Retrieve content for weak topics from vector store
        context_chunks = []
        for topic in student_profile.weak_topics[:3]:
            results = self._store.query(topic, collection_name, top_k=3)
            context_chunks.extend([r.text for r in results])

        if not context_chunks:
            return []

        context = "\n\n".join(dict.fromkeys(context_chunks))
        difficulty = self._accuracy_to_difficulty(student_profile.avg_accuracy)

        prompt = f"""You are a CBSE expert for Class {student_profile.class_num} {student_profile.subject.title()}.

A student is weak in these topics: {', '.join(student_profile.weak_topics)}.

Generate exactly {n} practice questions targeting these weak areas.
Difficulty: {difficulty} (calibrated to student's current performance).

Rules:
- Mix of 1-mark (MCQ/fill-in-the-blank), 2-mark, and 3-mark questions
- Include a brief answer for each
- Include a short hint to guide the student
- Questions must be directly answerable from the NCERT content below

Return ONLY a JSON array. No markdown fences.

Format:
[
  {{
    "question": "...",
    "answer": "...",
    "marks": 1,
    "difficulty": "easy|medium|hard",
    "topic": "...",
    "hint": "..."
  }}
]

NCERT Content:
{context[:3000]}
"""

        message = self._client.messages.create(
            model      = LLM_MODEL,
            max_tokens = 2048,
            messages   = [{"role": "user", "content": prompt}],
        )
        raw = message.content[0].text
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()

        try:
            items = json.loads(clean)
        except json.JSONDecodeError:
            return []

        questions = []
        for i, item in enumerate(items[:n]):
            questions.append(
                TestQuestion(
                    question_id = f"ai_ch{chapter}_{i}",
                    question    = item.get("question", ""),
                    answer      = item.get("answer", ""),
                    marks       = item.get("marks", 1),
                    difficulty  = item.get("difficulty", "medium"),
                    topic       = item.get("topic", ""),
                    source      = "ai_generated",
                    hint        = item.get("hint", ""),
                )
            )
        return questions

    def _marks_to_difficulty(self, marks: int) -> str:
        if marks <= 1: return "easy"
        if marks <= 3: return "medium"
        return "hard"

    def _accuracy_to_difficulty(self, accuracy: float) -> str:
        if accuracy < 0.4:  return "easy"
        if accuracy < 0.7:  return "medium"
        return "hard"

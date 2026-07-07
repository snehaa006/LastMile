"""
main.py — EduMind Pipeline Demo
─────────────────────────────────
Demonstrates the full EduMind pipeline end-to-end.

Run:
    python main.py

Make sure you have set ANTHROPIC_API_KEY in a .env file:
    echo "ANTHROPIC_API_KEY=your_key_here" > .env
"""

from rich.console import Console
from rich.panel   import Panel
from rich.table   import Table

from agents.orchestrator import EduMindAgent

console = Console()


def print_flashcards(flashcards, n=5):
    table = Table(title=f"Sample Flashcards (showing {n})", show_lines=True)
    table.add_column("Q", style="cyan", width=40)
    table.add_column("A", style="white", width=40)
    table.add_column("Difficulty", style="yellow", width=10)
    table.add_column("Topic", style="green", width=20)

    for card in flashcards[:n]:
        table.add_row(card.question, card.answer, card.difficulty, card.topic)

    console.print(table)


def print_test(test, n=5):
    console.print(
        Panel(
            f"[bold]Student:[/bold] {test.student_id}\n"
            f"[bold]Weak focus:[/bold] {', '.join(test.weak_focus)}\n"
            f"[bold]Total questions:[/bold] {len(test.questions)}\n"
            f"[bold]Total marks:[/bold] {test.total_marks}",
            title="Personalized Test Summary",
        )
    )
    for i, q in enumerate(test.questions[:n], 1):
        console.print(
            f"\n[bold cyan]Q{i}[/bold cyan] [{q.marks}m | {q.difficulty}] "
            f"[dim]{q.source}[/dim]\n"
            f"  {q.question}\n"
            f"  [dim]→ {q.answer[:100]}{'...' if len(q.answer) > 100 else ''}[/dim]"
        )
        if q.hint:
            console.print(f"  [yellow]Hint: {q.hint}[/yellow]")


def print_highlights(tagged_chunks, n=3):
    console.print("\n[bold]High Importance Chunks (e-NCERT highlights):[/bold]")
    high = [c for c in tagged_chunks if c.importance == "HIGH"]
    for chunk in high[:n]:
        console.print(
            Panel(
                f"{chunk.text[:300]}{'...' if len(chunk.text) > 300 else ''}\n\n"
                f"[yellow]Key terms: {', '.join(chunk.key_terms)}[/yellow]\n"
                f"[dim]Reason: {chunk.reason}[/dim]",
                title=f"[red]HIGH[/red] — Page {chunk.page}",
                border_style="red",
            )
        )


def main():
    console.print(
        Panel.fit(
            "[bold blue]EduMind — AI Study Companion[/bold blue]\n"
            "CBSE | NCERT-first | RAG pipeline",
            border_style="blue",
        )
    )

    # ── Configuration ─────────────────────────────────────────────────────────
    # Change these to test any class/subject/chapter
    CLASS_NUM   = 10
    SUBJECT     = "science"
    CHAPTER     = 1                  # Life Processes (Class 10 Bio)
    STUDENT_ID  = "stu_radhika_001"

    # ── Step 1: Full Chapter Pipeline ─────────────────────────────────────────
    console.rule("[bold]STEP 1: Process Chapter[/bold]")

    agent  = EduMindAgent()
    result = agent.process_chapter(
        class_num           = CLASS_NUM,
        subject             = SUBJECT,
        chapter             = CHAPTER,
        generate_highlights = True,
        num_flashcards      = 15,
    )

    # ── Step 2: Show Flashcards ───────────────────────────────────────────────
    console.rule("[bold]STEP 2: Flashcards[/bold]")
    print_flashcards(result.flashcards, n=5)

    # ── Step 3: Show e-NCERT Highlights ──────────────────────────────────────
    console.rule("[bold]STEP 3: e-NCERT Highlights[/bold]")
    print_highlights(result.tagged_chunks, n=3)
    console.print(f"\n[dim]Key terms for popups: {', '.join(result.key_terms[:10])}[/dim]")

    # ── Step 4: Personalized Test ─────────────────────────────────────────────
    console.rule("[bold]STEP 4: Personalized Test[/bold]")
    test = agent.generate_test(
        student_id    = STUDENT_ID,
        class_num     = CLASS_NUM,
        subject       = SUBJECT,
        chapter       = CHAPTER,
        weak_topics   = ["photosynthesis", "respiration", "nutrition in plants"],
        strong_topics = ["digestion"],
        avg_accuracy  = 0.45,    # student is struggling → easier questions
        num_questions = 10,
    )
    print_test(test, n=5)

    # ── Step 5: Semantic Search ───────────────────────────────────────────────
    console.rule("[bold]STEP 5: Semantic Search (Ask a concept)[/bold]")
    query = "What is the difference between autotrophic and heterotrophic nutrition?"
    console.print(f"[cyan]Query:[/cyan] {query}\n")
    results = agent.search_chapter(
        query     = query,
        class_num = CLASS_NUM,
        subject   = SUBJECT,
        chapter   = CHAPTER,
        top_k     = 2,
    )
    for i, r in enumerate(results, 1):
        console.print(Panel(r[:400], title=f"Result {i}", border_style="cyan"))

    console.print("\n[bold green]✅ Pipeline demo complete![/bold green]")


if __name__ == "__main__":
    main()

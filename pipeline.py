"""
Multi-agent research pipeline.

Orchestrates: Search → Reader → Writer ⇄ Critic (revision loop)
Saves the final report to an output file.
"""

import logging
import sys
import os
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table

from agents import (
    run_search_agent,
    run_reader_agent,
    run_writer_agent,
    run_critic_agent,
)
from models import PipelineState, CriticFeedback

# ═══════════════════════════════════════════════════════════════════════════
# Logging setup
# ═══════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")

console = Console()


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

MAX_REVISIONS = 3
QUALITY_THRESHOLD = 7  # Critic score 7+ = pass
MAX_SEARCH_QUERIES = 2
MAX_URLS_TO_SCRAPE = 5
OUTPUT_DIR = "output"


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline
# ═══════════════════════════════════════════════════════════════════════════

def run_research_pipeline(topic: str) -> PipelineState:
    """
    Run the full research pipeline:
      1. Search Agent → structured search results
      2. Reader Agent → scraped source evidence
      3. Writer Agent → draft report
      4. Critic Agent → evaluate + revision loop (up to MAX_REVISIONS)
      5. Save final report to file
    """
    state = PipelineState(
        topic=topic,
        max_revisions=MAX_REVISIONS,
        quality_threshold=QUALITY_THRESHOLD,
    )

    # ── Step 1: Search Agent ──────────────────────────────────────────────
    console.print(Panel(
        "[bold cyan]Step 1 — Search Agent[/bold cyan]\n"
        f"Finding information about: {topic}",
        border_style="cyan",
    ))

    try:
        state.search_results = run_search_agent(
            topic, max_queries=MAX_SEARCH_QUERIES
        )
    except Exception as e:
        logger.error("Search Agent failed: %s", e)
        state.warnings.append(f"Search Agent error: {e}")
        state.search_results = []

    if not state.search_results:
        console.print("[bold red]⚠ No search results found. "
                       "The report will be limited.[/bold red]")
        state.warnings.append("No search results were returned.")
    else:
        # Display search results table
        table = Table(title=f"Search Results ({len(state.search_results)} found)")
        table.add_column("#", style="dim", width=3)
        table.add_column("Title", style="green", max_width=50)
        table.add_column("Domain", style="cyan", max_width=25)
        table.add_column("URL", style="dim", max_width=50)
        for i, sr in enumerate(state.search_results, 1):
            table.add_row(str(i), sr.title[:50], sr.source_domain, sr.url[:50])
        console.print(table)

    # ── Step 2: Reader Agent ──────────────────────────────────────────────
    console.print(Panel(
        "[bold cyan]Step 2 — Reader Agent[/bold cyan]\n"
        f"Scraping top {MAX_URLS_TO_SCRAPE} URLs...",
        border_style="cyan",
    ))

    try:
        state.sources = run_reader_agent(
            state.search_results, max_urls=MAX_URLS_TO_SCRAPE
        )
    except Exception as e:
        logger.error("Reader Agent failed: %s", e)
        state.warnings.append(f"Reader Agent error: {e}")
        state.sources = []

    successful = sum(1 for s in state.sources if s.scrape_success)
    total = len(state.sources)
    console.print(
        f"  Scraped [green]{successful}[/green]/{total} URLs successfully."
    )

    if successful == 0:
        console.print("[bold red]⚠ No URLs could be scraped. "
                       "Report will rely on search snippets only.[/bold red]")
        state.warnings.append("No URLs could be scraped successfully.")

    # ── Step 3 & 4: Writer + Critic Revision Loop ─────────────────────────
    console.print(Panel(
        "[bold cyan]Step 3 — Writer + Critic Revision Loop[/bold cyan]\n"
        f"Max revisions: {MAX_REVISIONS} | Quality threshold: {QUALITY_THRESHOLD}/10",
        border_style="cyan",
    ))

    revision_instructions = ""
    best_report = ""
    best_score = 0

    for attempt in range(MAX_REVISIONS + 1):  # 0 = initial draft, 1..N = revisions
        is_revision = attempt > 0
        label = f"Revision {attempt}" if is_revision else "Initial Draft"
        console.print(f"\n  [bold yellow]📝 {label}[/bold yellow]")

        # Writer
        try:
            state.report_draft = run_writer_agent(
                topic=topic,
                sources=state.sources,
                report_date=state.report_date,
                revision_instructions=revision_instructions,
            )
        except Exception as e:
            logger.error("Writer Agent failed on %s: %s", label, e)
            state.warnings.append(f"Writer Agent error ({label}): {e}")
            break

        if not state.report_draft.strip():
            logger.warning("Writer produced empty report on %s.", label)
            state.warnings.append(f"Writer produced empty report on {label}.")
            break

        # Track best draft
        if best_report == "":
            best_report = state.report_draft

        # Critic
        console.print(f"  [bold yellow]🔍 Critic evaluating {label}...[/bold yellow]")
        try:
            feedback = run_critic_agent(
                report=state.report_draft,
                sources=state.sources,
            )
            state.critic_feedback = feedback
        except Exception as e:
            logger.error("Critic Agent failed on %s: %s", label, e)
            state.warnings.append(f"Critic Agent error ({label}): {e}")
            # Keep the current draft as best
            best_report = state.report_draft
            break

        # Display critic results
        score_color = "green" if feedback.score >= QUALITY_THRESHOLD else "red"
        console.print(
            f"  Score: [{score_color}]{feedback.score}/10[/{score_color}] "
            f"— Decision: [bold]{feedback.decision}[/bold]"
        )
        if feedback.issues:
            for issue in feedback.issues[:5]:
                console.print(f"    [red]• {issue}[/red]")
        if feedback.strengths:
            for strength in feedback.strengths[:3]:
                console.print(f"    [green]✓ {strength}[/green]")

        # Track best draft by score
        if feedback.score > best_score:
            best_score = feedback.score
            best_report = state.report_draft

        state.revision_count = attempt

        # Check pass condition
        if feedback.decision == "pass" and feedback.score >= QUALITY_THRESHOLD:
            console.print(
                f"\n  [bold green]✅ Report PASSED critic review "
                f"(score: {feedback.score}/10)[/bold green]"
            )
            break

        # Prepare for next revision (if we haven't hit the limit)
        if attempt < MAX_REVISIONS:
            corrections = "\n".join(
                f"- {c}" for c in feedback.required_corrections
            ) if feedback.required_corrections else "\n".join(
                f"- {issue}" for issue in feedback.issues
            )
            revision_instructions = (
                f"Previous draft score: {feedback.score}/10\n"
                f"Critic's issues:\n{corrections}\n"
                f"Critic's summary: {feedback.summary}\n"
            )
            console.print(
                f"  [yellow]↻ Sending back for revision "
                f"({attempt + 1}/{MAX_REVISIONS})...[/yellow]"
            )
        else:
            console.print(
                f"\n  [bold yellow]⚠ Maximum revisions ({MAX_REVISIONS}) reached. "
                f"Using best draft (score: {best_score}/10).[/bold yellow]"
            )
            state.warnings.append(
                f"Report did not pass after {MAX_REVISIONS} revisions. "
                f"Best score: {best_score}/10."
            )

    # Use the best draft
    state.report_draft = best_report

    # ── Step 5: Save and Display ──────────────────────────────────────────
    output_path = _save_report(topic, state)

    console.print("\n")
    console.print(Panel(
        Markdown(state.report_draft),
        title="[bold green]Final Report[/bold green]",
        border_style="green",
        padding=(1, 2),
    ))

    if state.warnings:
        console.print(Panel(
            "\n".join(f"⚠ {w}" for w in state.warnings),
            title="[bold yellow]Warnings[/bold yellow]",
            border_style="yellow",
        ))

    console.print(f"\n[dim]Report saved to: {output_path}[/dim]")

    return state


def _save_report(topic: str, state: PipelineState) -> str:
    """Save the final report to a timestamped output file."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Create a safe filename
    safe_topic = "".join(
        c if c.isalnum() or c in " -_" else "_" for c in topic
    )[:50].strip()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{safe_topic}_{timestamp}.md"
    filepath = os.path.join(OUTPUT_DIR, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(state.report_draft)

        # Append pipeline metadata
        f.write("\n\n---\n\n")
        f.write("## Pipeline Metadata\n\n")
        f.write(f"- **Topic:** {state.topic}\n")
        f.write(f"- **Report Date:** {state.report_date}\n")
        f.write(f"- **Sources Found:** {len(state.search_results)}\n")
        f.write(f"- **Sources Scraped:** "
                f"{sum(1 for s in state.sources if s.scrape_success)}"
                f"/{len(state.sources)}\n")
        f.write(f"- **Revisions:** {state.revision_count}\n")
        if state.critic_feedback:
            f.write(f"- **Final Critic Score:** "
                    f"{state.critic_feedback.score}/10\n")
            f.write(f"- **Critic Decision:** "
                    f"{state.critic_feedback.decision}\n")
        if state.warnings:
            f.write(f"\n### Warnings\n\n")
            for w in state.warnings:
                f.write(f"- {w}\n")

    logger.info("Report saved to %s", filepath)
    return filepath


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    try:
        topic = input("\n Enter a research topic: ")
        if not topic.strip():
            console.print("[red]No topic provided. Exiting.[/red]")
            sys.exit(1)
        run_research_pipeline(topic.strip())
    except KeyboardInterrupt:
        console.print("\n[yellow]Pipeline interrupted by user.[/yellow]")
        sys.exit(0)
    except Exception as e:
        logger.critical("Pipeline failed: %s", e, exc_info=True)
        console.print(f"[bold red]Pipeline failed: {e}[/bold red]")
        sys.exit(1)
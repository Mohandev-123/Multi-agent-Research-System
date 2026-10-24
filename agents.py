"""
Multi-agent research system — Agent definitions.

Agents:
  1. Search Agent  — calls Tavily search, returns structured SearchResult list.
  2. Reader Agent  — scrapes URLs from search results, returns SourceEvidence list.
  3. Writer Agent  — LLM chain that writes a cited research report.
  4. Critic Agent  — LLM chain that evaluates the report and returns CriticFeedback.
"""

import json
import logging
from datetime import datetime
from urllib.parse import urlparse

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

from tools import web_search, scrape_url
from models import (
    SearchResult,
    ScrapedDocument,
    SourceEvidence,
    CriticFeedback,
)

load_dotenv()
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# Model setup — shared LLM instance
# ═══════════════════════════════════════════════════════════════════════════

try:
    llm = ChatOpenAI(
        model="qwen3.5-2b",
        base_url="http://localhost:1235/v1",
        api_key="lm-studio",
        temperature=0,
    )
except Exception as e:
    logger.critical("Failed to initialise LLM: %s", e)
    raise RuntimeError(
        "Could not connect to LM Studio at http://localhost:1235/v1. "
        "Ensure LM Studio is running and the model 'qwen3.5-2b' is loaded."
    ) from e


# ═══════════════════════════════════════════════════════════════════════════
# Agent 1 — Search Agent
# ═══════════════════════════════════════════════════════════════════════════

def run_search_agent(topic: str, max_queries: int = 2) -> list[SearchResult]:
    """
    Run the search agent: generate search queries with the LLM, then call the
    Tavily search tool deterministically. Returns a deduplicated list of
    SearchResult objects.

    The LLM is used ONLY to generate good search queries — it never rewrites
    the search results themselves.
    """
    logger.info("Search Agent starting for topic: %s", topic)

    # Ask LLM to generate diverse search queries
    query_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a research assistant. Given a research topic, generate "
         f"exactly {max_queries} diverse, specific search queries that would "
         "find authoritative, recent information about the topic. "
         "Return ONLY a JSON array of query strings. No explanation."),
        ("human", "Topic: {topic}"),
    ])

    query_chain = query_prompt | llm | StrOutputParser()
    raw_queries = query_chain.invoke({"topic": topic})

    # Parse the queries
    queries = []
    try:
        parsed = json.loads(raw_queries.strip().strip("`").removeprefix("json").strip())
        if isinstance(parsed, list):
            queries = [str(q).strip() for q in parsed if str(q).strip()]
    except (json.JSONDecodeError, TypeError):
        logger.warning("Could not parse LLM query list, falling back to topic as query.")

    # Always include the original topic as a fallback query
    if not queries:
        queries = [topic]
    queries = queries[:max_queries]

    logger.info("Search queries: %s", queries)

    # Execute searches deterministically via the tool
    all_results: list[SearchResult] = []
    seen_urls: set[str] = set()

    for query in queries:
        raw_json = web_search.invoke(query)
        try:
            data = json.loads(raw_json)
        except json.JSONDecodeError:
            logger.warning("Failed to parse search results for query: %s", query)
            continue

        if data.get("error"):
            logger.warning("Search error for '%s': %s", query, data["error"])
            continue

        for item in data.get("results", []):
            url = item.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                all_results.append(SearchResult(**item))

    logger.info("Search Agent found %d unique results.", len(all_results))

    if not all_results:
        logger.warning("Search Agent returned no results for topic: %s", topic)

    return all_results


# ═══════════════════════════════════════════════════════════════════════════
# Agent 2 — Reader Agent
# ═══════════════════════════════════════════════════════════════════════════

def run_reader_agent(
    search_results: list[SearchResult],
    max_urls: int = 5,
) -> list[SourceEvidence]:
    """
    Scrape the top URLs from the search results, extracting clean text.
    Returns SourceEvidence objects that pair search metadata with extracted
    content. Failed URLs are logged and skipped, not retried endlessly.
    """
    logger.info("Reader Agent starting — %d URLs to process (max %d).",
                len(search_results), max_urls)

    sources: list[SourceEvidence] = []
    citation_index = 0

    for sr in search_results[:max_urls]:
        if not sr.url:
            continue

        citation_index += 1
        logger.info("  [%d] Scraping: %s", citation_index, sr.url)

        raw_json = scrape_url.invoke(sr.url)
        try:
            scraped = json.loads(raw_json)
        except json.JSONDecodeError:
            scraped = {"success": False, "error": "Could not parse scrape response."}

        doc = ScrapedDocument(**scraped)

        source = SourceEvidence(
            index=citation_index,
            title=doc.title if doc.title else sr.title,
            url=sr.url,
            source_domain=sr.source_domain or urlparse(sr.url).netloc,
            published_date=sr.published_date,
            snippet=sr.snippet,
            extracted_content=doc.content,
            scrape_success=doc.success,
            scrape_error=doc.error,
        )
        sources.append(source)

        if doc.success:
            logger.info("    ✓ Extracted %d chars from %s", len(doc.content), sr.url)
        else:
            logger.warning("    ✗ Failed to scrape %s: %s", sr.url, doc.error)

    successful = sum(1 for s in sources if s.scrape_success)
    logger.info("Reader Agent finished — %d/%d URLs scraped successfully.",
                successful, len(sources))

    return sources


# ═══════════════════════════════════════════════════════════════════════════
# Agent 3 — Writer Agent (LLM chain)
# ═══════════════════════════════════════════════════════════════════════════

def _format_evidence_for_writer(sources: list[SourceEvidence]) -> str:
    """Format the collected evidence into a structured text block for the writer LLM."""
    parts = []
    for s in sources:
        status = "SCRAPED" if s.scrape_success else "FAILED"
        content_preview = s.extracted_content[:2000] if s.extracted_content else s.snippet
        parts.append(
            f"[Source {s.index}] ({status})\n"
            f"  Title: {s.title}\n"
            f"  URL: {s.url}\n"
            f"  Domain: {s.source_domain}\n"
            f"  Date: {s.published_date or 'Unknown'}\n"
            f"  Content:\n{content_preview}\n"
        )
    return "\n---\n".join(parts)


def _build_references_block(sources: list[SourceEvidence]) -> str:
    """Build a ready-made references list for the writer to include."""
    lines = []
    for s in sources:
        date_str = f" ({s.published_date})" if s.published_date else ""
        lines.append(f"[{s.index}] {s.title}{date_str}. {s.url}")
    return "\n".join(lines)


writer_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an expert research report writer. You produce detailed, "
     "evidence-based, professionally structured reports. "
     "CRITICAL RULES:\n"
     "- Support every factual claim with a citation like [1], [2] etc.\n"
     "- Use ONLY the source evidence provided — do NOT invent facts, statistics, "
     "  URLs, authors, journal names, or publication dates.\n"
     "- If a claim cannot be verified from the sources, qualify it with phrases "
     "  like 'reportedly', 'according to unverified sources', or omit it.\n"
     "- Include the exact References list provided at the end of the report.\n"
     "- Use the exact Report Date provided.\n"
     "- Do NOT fabricate citations or reference entries.\n"
     "- Do NOT provide investment advice or financial recommendations."),
    ("human",
     "Write a detailed research report on the topic below.\n\n"
     "Topic: {topic}\n\n"
     "Report Date: {report_date}\n\n"
     "=== SOURCE EVIDENCE ===\n{evidence}\n\n"
     "=== REFERENCES (include exactly as-is at the end of your report) ===\n{references}\n\n"
     "{revision_instructions}"
     "Structure the report with these sections:\n"
     "1. Title\n"
     "2. Report date and metadata\n"
     "3. Executive Summary\n"
     "4. Research Question and Scope\n"
     "5. Key Findings (minimum 3 well-explained points with citations)\n"
     "6. Detailed Analysis\n"
     "7. Limitations and Conflicting Evidence\n"
     "8. Conclusion\n"
     "9. References\n\n"
     "Be detailed, factual, and professional. Cite sources with [N] notation."),
])

writer_chain = writer_prompt | llm | StrOutputParser()


def run_writer_agent(
    topic: str,
    sources: list[SourceEvidence],
    report_date: str | None = None,
    revision_instructions: str = "",
) -> str:
    """
    Generate a research report using the collected evidence.
    Optionally includes revision instructions from the critic.
    """
    if report_date is None:
        report_date = datetime.now().strftime("%B %d, %Y")

    evidence_text = _format_evidence_for_writer(sources)
    references_text = _build_references_block(sources)

    if revision_instructions:
        revision_instructions = (
            f"=== REVISION INSTRUCTIONS ===\n"
            f"A reviewer found issues with your previous draft. "
            f"Address these specific problems:\n{revision_instructions}\n\n"
        )

    logger.info("Writer Agent generating report (revision instructions: %s).",
                "yes" if revision_instructions else "no")

    report = writer_chain.invoke({
        "topic": topic,
        "report_date": report_date,
        "evidence": evidence_text,
        "references": references_text,
        "revision_instructions": revision_instructions,
    })

    logger.info("Writer Agent produced %d chars.", len(report))
    return report


# ═══════════════════════════════════════════════════════════════════════════
# Agent 4 — Critic Agent (LLM chain)
# ═══════════════════════════════════════════════════════════════════════════

critic_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a rigorous research report critic. You evaluate reports against "
     "the source evidence provided. You are honest and specific.\n\n"
     "CRITICAL RULES:\n"
     "- Do NOT approve a report just because it is well-written.\n"
     "- Check that every cited source [N] matches an entry in the references.\n"
     "- Check that statistics and claims are supported by the source evidence.\n"
     "- Flag any fabricated data, URLs, authors, or citation details.\n"
     "- Flag missing coverage of the research question.\n"
     "- Flag outdated information or incorrect dates.\n"
     "- Check for logical consistency and contradictions."),
    ("human",
     "Evaluate the research report below against the source evidence.\n\n"
     "=== REPORT ===\n{report}\n\n"
     "=== SOURCE EVIDENCE ===\n{evidence}\n\n"
     "=== AVAILABLE REFERENCES ===\n{references}\n\n"
     "Respond in EXACTLY this JSON format (no other text):\n"
     '{{\n'
     '  "score": <0-10>,\n'
     '  "decision": "<pass or revise>",\n'
     '  "strengths": ["..."],\n'
     '  "issues": ["..."],\n'
     '  "required_corrections": ["..."],\n'
     '  "summary": "one-line verdict"\n'
     '}}\n\n'
     "Rules for scoring:\n"
     "- Score 7+ and decision 'pass': report is good enough.\n"
     "- Score below 7 or any fabricated content: decision must be 'revise'.\n"
     "- Be specific in issues and required_corrections."),
])

critic_chain = critic_prompt | llm | StrOutputParser()


def _parse_critic_response(raw: str) -> CriticFeedback:
    """Parse the critic's JSON response into a CriticFeedback model."""
    # Try to find JSON in the response
    raw = raw.strip()

    # Strip markdown code fences if present
    if raw.startswith("```"):
        lines = raw.split("\n")
        # Remove first and last line (fences)
        lines = [l for l in lines if not l.strip().startswith("```")]
        raw = "\n".join(lines).strip()

    try:
        data = json.loads(raw)
        return CriticFeedback(
            score=int(data.get("score", 0)),
            decision=str(data.get("decision", "revise")).lower().strip(),
            strengths=data.get("strengths", []),
            issues=data.get("issues", []),
            required_corrections=data.get("required_corrections", []),
            summary=data.get("summary", ""),
        )
    except (json.JSONDecodeError, TypeError, ValueError) as e:
        logger.warning("Could not parse critic JSON (%s). Treating as 'revise'.", e)
        return CriticFeedback(
            score=3,
            decision="revise",
            issues=["Critic response could not be parsed as structured JSON."],
            required_corrections=["Re-run critic evaluation."],
            summary=raw[:200],
        )


def run_critic_agent(
    report: str,
    sources: list[SourceEvidence],
) -> CriticFeedback:
    """
    Evaluate the report against the collected source evidence.
    Returns structured CriticFeedback.
    """
    evidence_text = _format_evidence_for_writer(sources)
    references_text = _build_references_block(sources)

    logger.info("Critic Agent evaluating report (%d chars).", len(report))

    raw = critic_chain.invoke({
        "report": report,
        "evidence": evidence_text,
        "references": references_text,
    })

    feedback = _parse_critic_response(raw)
    logger.info("Critic score: %d/10 — Decision: %s", feedback.score, feedback.decision)

    return feedback
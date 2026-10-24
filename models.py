"""
Pydantic models for structured data exchange between agents.
Provides type-safe contracts for search results, scraped content,
critic feedback, and the overall pipeline state.
"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class SearchResult(BaseModel):
    """A single search result from the web search tool."""
    title: str = ""
    url: str = ""
    snippet: str = ""
    source_domain: str = ""
    published_date: Optional[str] = None


class ScrapedDocument(BaseModel):
    """Content extracted from a single URL."""
    url: str = ""
    title: str = ""
    content: str = ""
    success: bool = False
    error: Optional[str] = None
    source_domain: str = ""


class SourceEvidence(BaseModel):
    """A source with its search metadata and scraped content combined."""
    index: int = 0           # 1-based citation index
    title: str = ""
    url: str = ""
    source_domain: str = ""
    published_date: Optional[str] = None
    snippet: str = ""        # from search result
    extracted_content: str = ""  # from scraping
    scrape_success: bool = False
    scrape_error: Optional[str] = None


class CriticFeedback(BaseModel):
    """Structured feedback from the critic agent."""
    score: int = Field(default=0, ge=0, le=10)
    decision: str = "revise"  # "pass" or "revise"
    issues: list[str] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    required_corrections: list[str] = Field(default_factory=list)
    summary: str = ""


class PipelineState(BaseModel):
    """Overall state of the research pipeline."""
    topic: str = ""
    search_results: list[SearchResult] = Field(default_factory=list)
    sources: list[SourceEvidence] = Field(default_factory=list)
    report_draft: str = ""
    critic_feedback: Optional[CriticFeedback] = None
    revision_count: int = 0
    max_revisions: int = 3
    quality_threshold: int = 7  # score out of 10
    report_date: str = Field(default_factory=lambda: datetime.now().strftime("%B %d, %Y"))
    warnings: list[str] = Field(default_factory=list)

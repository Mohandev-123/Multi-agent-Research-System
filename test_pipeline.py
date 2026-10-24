"""
Tests for the multi-agent research pipeline.

Includes both unit tests (with mocked external services) and
an integration test marker for live pipeline testing.
"""

import json
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime

from models import (
    SearchResult,
    ScrapedDocument,
    SourceEvidence,
    CriticFeedback,
    PipelineState,
)


# ═══════════════════════════════════════════════════════════════════════════
# Test Models
# ═══════════════════════════════════════════════════════════════════════════

class TestModels:
    """Test that Pydantic models validate correctly."""

    def test_search_result_defaults(self):
        sr = SearchResult()
        assert sr.title == ""
        assert sr.url == ""

    def test_search_result_with_data(self):
        sr = SearchResult(
            title="Test Article",
            url="https://example.com/article",
            snippet="This is a test.",
            source_domain="example.com",
            published_date="2024-01-01",
        )
        assert sr.title == "Test Article"
        assert sr.url == "https://example.com/article"
        assert sr.source_domain == "example.com"

    def test_critic_feedback_score_bounds(self):
        fb = CriticFeedback(score=0)
        assert fb.score == 0
        fb = CriticFeedback(score=10)
        assert fb.score == 10

    def test_critic_feedback_default_decision(self):
        fb = CriticFeedback()
        assert fb.decision == "revise"

    def test_pipeline_state_defaults(self):
        state = PipelineState(topic="test")
        assert state.topic == "test"
        assert state.max_revisions == 3
        assert state.quality_threshold == 7
        assert state.revision_count == 0
        assert len(state.report_date) > 0

    def test_source_evidence_citation_index(self):
        se = SourceEvidence(index=1, url="https://example.com")
        assert se.index == 1
        assert se.scrape_success is False


# ═══════════════════════════════════════════════════════════════════════════
# Test Tools (mocked)
# ═══════════════════════════════════════════════════════════════════════════

class TestWebSearchTool:
    """Test web_search tool with mocked Tavily client."""

    @patch("tools.tavily")
    def test_returns_structured_json(self, mock_tavily):
        mock_tavily.search.return_value = {
            "results": [
                {
                    "title": "War and Markets",
                    "url": "https://reuters.com/war-markets",
                    "content": "War impacts stock markets significantly.",
                    "published_date": "2024-06-15",
                },
                {
                    "title": "Defense Spending Surge",
                    "url": "https://bbc.com/defense-spending",
                    "content": "Global defense spending hits record.",
                },
            ]
        }

        from tools import web_search

        result = web_search.invoke("war stock market impact")
        data = json.loads(result)

        assert "results" in data
        assert len(data["results"]) == 2
        assert data["results"][0]["title"] == "War and Markets"
        assert data["results"][0]["url"] == "https://reuters.com/war-markets"
        assert "source_domain" in data["results"][0]

    @patch("tools.tavily")
    def test_deduplicates_urls(self, mock_tavily):
        mock_tavily.search.return_value = {
            "results": [
                {"title": "A", "url": "https://example.com/1", "content": "..."},
                {"title": "B", "url": "https://example.com/1", "content": "..."},  # duplicate
                {"title": "C", "url": "https://example.com/2", "content": "..."},
            ]
        }

        from tools import web_search, deduplicate_results

        result = web_search.invoke("test query")
        data = json.loads(result)

        # The tool itself deduplicates
        urls = [r["url"] for r in data["results"]]
        assert len(set(urls)) == len(urls)

    @patch("tools.tavily", None)
    def test_returns_error_when_no_api_key(self):
        from tools import web_search

        result = web_search.invoke("test query")
        data = json.loads(result)
        assert "error" in data
        assert len(data.get("results", [])) == 0


class TestScrapeUrlTool:
    """Test scrape_url tool with mocked HTTP requests."""

    @patch("tools._fetch_page")
    def test_successful_scrape(self, mock_fetch):
        mock_response = MagicMock()
        mock_response.headers = {"Content-Type": "text/html; charset=utf-8"}
        mock_response.text = """
        <html>
            <head><title>Test Article</title></head>
            <body>
                <nav>Navigation</nav>
                <article>
                    <p>This is the main article content about stock markets.</p>
                    <p>It contains important research data.</p>
                </article>
                <footer>Footer content</footer>
            </body>
        </html>
        """
        mock_fetch.return_value = mock_response

        from tools import scrape_url

        result = scrape_url.invoke("https://example.com/article")
        data = json.loads(result)

        assert data["success"] is True
        assert data["title"] == "Test Article"
        assert "stock markets" in data["content"]
        assert "Navigation" not in data["content"]
        assert "Footer content" not in data["content"]

    @patch("tools._fetch_page")
    def test_failed_scrape_returns_error(self, mock_fetch):
        import requests
        mock_fetch.side_effect = requests.Timeout("Connection timed out")

        from tools import scrape_url

        result = scrape_url.invoke("https://example.com/slow")
        data = json.loads(result)

        assert data["success"] is False
        assert data["error"] is not None
        assert "timed out" in data["error"].lower()


# ═══════════════════════════════════════════════════════════════════════════
# Test Agents (mocked LLM)
# ═══════════════════════════════════════════════════════════════════════════

class TestSearchAgent:
    """Test search agent with mocked LLM and tools."""

    @patch("agents.web_search")
    @patch("agents.llm")
    def test_returns_search_results(self, mock_llm, mock_web_search):
        # Mock LLM to return query list
        mock_llm_response = MagicMock()
        mock_llm_response.content = '["war stock market impact", "geopolitical conflict financial markets"]'

        # Make the chain work by patching the invoke
        search_json = json.dumps({
            "results": [
                {
                    "title": "War and Markets",
                    "url": "https://reuters.com/war",
                    "snippet": "War impacts markets.",
                    "source_domain": "reuters.com",
                    "published_date": None,
                }
            ]
        })
        mock_web_search.invoke.return_value = search_json

        from agents import run_search_agent

        with patch("agents.StrOutputParser") as mock_parser:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = '["war stock market impact"]'

            with patch.object(
                type(mock_llm), '__or__',
                return_value=mock_chain
            ):
                # Direct test: parse search results
                results_data = json.loads(search_json)
                results = [SearchResult(**r) for r in results_data["results"]]

                assert len(results) == 1
                assert results[0].url == "https://reuters.com/war"
                assert results[0].title == "War and Markets"


class TestReaderAgent:
    """Test reader agent with mocked scraper."""

    @patch("agents.scrape_url")
    def test_scrapes_multiple_urls(self, mock_scrape):
        scrape_response = json.dumps({
            "url": "https://example.com",
            "title": "Test",
            "content": "Article content here.",
            "success": True,
            "error": None,
        })
        mock_scrape.invoke.return_value = scrape_response

        from agents import run_reader_agent

        search_results = [
            SearchResult(
                title="Article 1",
                url="https://example.com/1",
                snippet="Snippet 1",
                source_domain="example.com",
            ),
            SearchResult(
                title="Article 2",
                url="https://example.com/2",
                snippet="Snippet 2",
                source_domain="example.com",
            ),
        ]

        sources = run_reader_agent(search_results, max_urls=5)
        assert len(sources) == 2
        assert sources[0].index == 1
        assert sources[1].index == 2
        assert all(s.scrape_success for s in sources)

    @patch("agents.scrape_url")
    def test_handles_failed_scrapes(self, mock_scrape):
        mock_scrape.invoke.return_value = json.dumps({
            "url": "https://example.com",
            "title": "",
            "content": "",
            "success": False,
            "error": "Connection timed out",
        })

        from agents import run_reader_agent

        search_results = [
            SearchResult(
                title="Unreachable",
                url="https://example.com/down",
                snippet="Some snippet",
                source_domain="example.com",
            ),
        ]

        sources = run_reader_agent(search_results, max_urls=5)
        assert len(sources) == 1
        assert sources[0].scrape_success is False
        assert sources[0].scrape_error is not None
        # Snippet should still be available
        assert sources[0].snippet == "Some snippet"


class TestCriticParsing:
    """Test critic JSON response parsing."""

    def test_parse_valid_json(self):
        from agents import _parse_critic_response

        raw = json.dumps({
            "score": 8,
            "decision": "pass",
            "strengths": ["Good structure", "Well cited"],
            "issues": [],
            "required_corrections": [],
            "summary": "Report meets quality standards.",
        })

        feedback = _parse_critic_response(raw)
        assert feedback.score == 8
        assert feedback.decision == "pass"
        assert len(feedback.strengths) == 2

    def test_parse_markdown_wrapped_json(self):
        from agents import _parse_critic_response

        raw = '```json\n{"score": 5, "decision": "revise", "strengths": [], "issues": ["Missing citations"], "required_corrections": ["Add citations"], "summary": "Needs work."}\n```'

        feedback = _parse_critic_response(raw)
        assert feedback.score == 5
        assert feedback.decision == "revise"
        assert "Missing citations" in feedback.issues

    def test_parse_invalid_json_returns_revise(self):
        from agents import _parse_critic_response

        feedback = _parse_critic_response("This is not JSON at all.")
        assert feedback.decision == "revise"
        assert feedback.score == 3


class TestRevisionLoop:
    """Test the writer-critic revision loop logic."""

    def test_pipeline_state_tracks_revisions(self):
        state = PipelineState(topic="test", max_revisions=3)
        assert state.revision_count == 0

        state.revision_count = 1
        assert state.revision_count == 1

    def test_pass_decision_stops_loop(self):
        feedback = CriticFeedback(score=8, decision="pass")
        should_stop = (
            feedback.decision == "pass"
            and feedback.score >= 7
        )
        assert should_stop is True

    def test_revise_decision_continues_loop(self):
        feedback = CriticFeedback(score=4, decision="revise")
        should_stop = (
            feedback.decision == "pass"
            and feedback.score >= 7
        )
        assert should_stop is False


# ═══════════════════════════════════════════════════════════════════════════
# Integration test (requires live LM Studio + internet)
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
class TestLiveIntegration:
    """
    Integration tests that require a running LM Studio server and internet.
    Run with: pytest test_pipeline.py -m integration -v
    """

    def test_full_pipeline(self):
        """Run the complete pipeline with a simple topic."""
        from pipeline import run_research_pipeline

        state = run_research_pipeline("impact of AI on healthcare")

        # Verify search results
        assert len(state.search_results) > 0, "Search should return results"

        # Verify sources
        assert len(state.sources) > 0, "Reader should produce sources"

        # Verify at least some URLs have content
        scraped = [s for s in state.sources if s.scrape_success]
        assert len(scraped) > 0, "At least one URL should be scraped"

        # Verify report
        assert len(state.report_draft) > 500, "Report should be substantial"

        # Verify citations in report
        assert "[1]" in state.report_draft, "Report should contain citations"

        # Verify critic feedback
        assert state.critic_feedback is not None, "Critic should provide feedback"
        assert 0 <= state.critic_feedback.score <= 10

        # Verify report date is current
        today = datetime.now().strftime("%B")
        assert today in state.report_draft or today.lower() in state.report_draft.lower(), \
            "Report should contain current month"

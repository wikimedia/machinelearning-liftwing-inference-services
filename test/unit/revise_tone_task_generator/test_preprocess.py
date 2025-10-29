import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.models.revise_tone_task_generator.model_server.model import (
    ReviseToneTaskGenerator,
)


@pytest.fixture
def mock_model():
    """Create a ReviseToneTaskGenerator instance with mocked model loading."""
    with patch.object(ReviseToneTaskGenerator, "load", return_value=MagicMock()):
        model = ReviseToneTaskGenerator(name="revise-tone-task-generator")
        return model


@pytest.fixture
def sample_page_content_change_event():
    """Load sample mediawiki.page_content_change.v1 event from JSON file."""
    sample_file = Path(__file__).parent / "sample_payload.json"
    with open(sample_file) as f:
        return json.load(f)


@pytest.fixture
def mock_article_topics():
    """Mock response from the article topic model for Anees (musician) (page_id=70793851)."""
    return {
        "prediction": {
            "article": "https://en.wikipedia.org/wiki?curid=70793851",
            "results": [
                {"topic": "Culture.Biography.Biography*", "score": 0.9748311638832092},
                {"topic": "Culture.Media.Music", "score": 0.9099169969558716},
                {"topic": "Culture.Media.Media*", "score": 0.8634016513824463},
                {
                    "topic": "Geography.Regions.Americas.North_America",
                    "score": 0.6224693655967712,
                },
            ],
        }
    }


@pytest.mark.asyncio
async def test_preprocess_extracts_paragraphs(
    mock_model, sample_page_content_change_event, mock_article_topics
):
    """Test that preprocess method extracts and parses paragraphs from the event."""
    # Mock the get_article_topics method to avoid external API calls
    with patch.object(
        mock_model,
        "get_article_topics",
        new_callable=AsyncMock,
        return_value=mock_article_topics,
    ):
        result = await mock_model.preprocess(sample_page_content_change_event)

    # Check that paragraphs were extracted
    assert "paragraphs" in result
    assert result["paragraphs"] is not None
    assert len(result["paragraphs"]) > 0

    # Check paragraph structure (list of tuples)
    assert isinstance(result["paragraphs"], list)
    for paragraph in result["paragraphs"]:
        assert isinstance(paragraph, tuple)
        assert len(paragraph) == 2
        section_name, text = paragraph
        assert isinstance(section_name, str)
        assert isinstance(text, str)
        assert len(text) > 30  # Should be > 30 chars

    # Check that it contains expected content
    all_text = " ".join([p[1] for p in result["paragraphs"]])
    assert "Anees Mokhiber" in all_text
    assert "Palestinian-American" in all_text


@pytest.mark.asyncio
async def test_preprocess_extracts_metadata(
    mock_model, sample_page_content_change_event, mock_article_topics
):
    """Test that preprocess method extracts necessary metadata from the event."""
    # Mock the get_article_topics method to avoid external API calls
    with patch.object(
        mock_model,
        "get_article_topics",
        new_callable=AsyncMock,
        return_value=mock_article_topics,
    ):
        result = await mock_model.preprocess(sample_page_content_change_event)

    # Check that metadata was extracted
    assert "page_id" in result
    assert result["page_id"] == 70793851

    assert "page_title" in result
    assert result["page_title"] == "Anees_(musician)"

    assert "wiki_id" in result
    assert result["wiki_id"] == "enwiki"

    assert "lang" in result
    assert result["lang"] == "en"


@pytest.mark.asyncio
async def test_preprocess_filters_sections(
    mock_model, sample_page_content_change_event, mock_article_topics
):
    """Test that preprocess filters out unwanted sections like References."""
    # Add a References section to the content
    sample_page_content_change_event["revision"]["content_slots"]["main"][
        "content_body"
    ] += """

== References ==
Some reference content here.

== External links ==
Some external links here.
"""

    # Mock the get_article_topics method to avoid external API calls
    with patch.object(
        mock_model,
        "get_article_topics",
        new_callable=AsyncMock,
        return_value=mock_article_topics,
    ):
        result = await mock_model.preprocess(sample_page_content_change_event)

    # Check that References and External links sections are not in the paragraphs
    section_names = [p[0] for p in result["paragraphs"]]
    assert "References" not in section_names
    assert "External links" not in section_names


@pytest.mark.asyncio
async def test_preprocess_includes_article_topics(
    mock_model, sample_page_content_change_event, mock_article_topics
):
    """Test that preprocess includes article topics from the outlink model."""
    # Mock the get_article_topics method
    with patch.object(
        mock_model,
        "get_article_topics",
        new_callable=AsyncMock,
        return_value=mock_article_topics,
    ):
        result = await mock_model.preprocess(sample_page_content_change_event)

    # Check that article_topics is included in the result
    assert "article_topics" in result
    assert result["article_topics"] == mock_article_topics
    assert "prediction" in result["article_topics"]
    assert "results" in result["article_topics"]["prediction"]
    assert len(result["article_topics"]["prediction"]["results"]) == 4


def test_should_process_article_with_matching_topic(mock_model):
    """Test that articles with matching topics are marked for processing."""
    # Article with Biography topic (matches allowed topics)
    article_topics = {
        "prediction": {
            "article": "https://en.wikipedia.org/wiki?curid=70793851",
            "results": [
                {"topic": "Culture.Biography.Biography*", "score": 0.9748},
                {"topic": "Culture.Media.Music", "score": 0.9099},
            ],
        }
    }

    assert mock_model.should_process_article(article_topics) is True


def test_should_process_article_without_matching_topic(mock_model):
    """Test that articles without matching topics are not marked for processing."""
    # Article without any matching topics
    article_topics = {
        "prediction": {
            "article": "https://en.wikipedia.org/wiki?curid=12345",
            "results": [
                {"topic": "Science.Technology", "score": 0.8},
                {"topic": "Geography.Regions", "score": 0.7},
            ],
        }
    }

    assert mock_model.should_process_article(article_topics) is False


def test_should_process_article_with_empty_topics(mock_model):
    """Test that articles with empty topics are not marked for processing."""
    # Empty article topics
    assert mock_model.should_process_article({}) is False
    assert mock_model.should_process_article({"prediction": {}}) is False
    assert mock_model.should_process_article({"prediction": {"results": []}}) is False


def test_should_process_article_with_women_topic(mock_model):
    """Test that articles with Culture.Biography.Women topic are marked for processing."""
    article_topics = {
        "prediction": {
            "article": "https://en.wikipedia.org/wiki?curid=99999",
            "results": [
                {"topic": "Culture.Biography.Women", "score": 0.95},
            ],
        }
    }

    assert mock_model.should_process_article(article_topics) is True


@pytest.mark.asyncio
async def test_preprocess_sets_should_process_flag(
    mock_model, sample_page_content_change_event, mock_article_topics
):
    """Test that preprocess sets the should_process flag correctly."""
    # Mock with matching topics
    with patch.object(
        mock_model,
        "get_article_topics",
        new_callable=AsyncMock,
        return_value=mock_article_topics,
    ):
        result = await mock_model.preprocess(sample_page_content_change_event)

    # Check that should_process is set to True (mock has Biography topic)
    assert "should_process" in result
    assert result["should_process"] is True


def test_extract_paragraphs_basic(mock_model):
    """Test the extract_paragraphs method with basic wikitext."""
    wikitext = """
'''Test Article''' is an article for testing.

== Section 1 ==
This is the first paragraph in section 1. It has enough text to pass the length filter.

This is the second paragraph in section 1. Also long enough.

== Section 2 ==
This is a paragraph in section 2. It contains sufficient text.
"""

    paragraphs = mock_model.extract_paragraphs(wikitext, "en")

    assert len(paragraphs) >= 3
    section_names = [p[0] for p in paragraphs]
    assert "LEAD_SECTION" in section_names
    assert "Section 1" in section_names
    assert "Section 2" in section_names

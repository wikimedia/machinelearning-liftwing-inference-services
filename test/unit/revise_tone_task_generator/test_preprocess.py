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
        model = ReviseToneTaskGenerator(
            name="revise-tone-task-generator", use_cache=False
        )
        return model


@pytest.fixture
def sample_page_change_event():
    """Load sample mediawiki.page_change.v1 event from JSON file."""
    sample_file = Path(__file__).parent / "sample_payload.json"
    with open(sample_file) as f:
        return json.load(f)


@pytest.fixture
def sample_page_content():
    """Sample wikitext content for Anees (musician) page."""
    return """{{Short description|American singer, rapper & songwriter (born 1992)}}
{{About|the musician|other uses|Anees}}
{{Use American English|date=July 2022}}
{{Use mdy dates|date=July 2022}}
{{Infobox person
| name               = anees
| image              =Anees headshot.jpg
| alt                =
| caption            = anees in 2022.
| alias              = anees
| birth_name         = Anees Mokhiber
| birth_date         = {{Birth date and age|1992|7|30}}
| birth_place        = [[Washington D.C.]], U.S.
| occupation         = {{hlist|Singer|rapper|songwriter}}
| television         =
| parents            =
| awards             =
| website            = {{URL|aneesofficial.com}}
| module             = {{Infobox musical artist
| embed           = yes
| background      = solo_singer
| genre           = {{hlist|[[Pop music|Pop]]|[[R&B]]|[[hip hop music|hip hop]]|[[pop rap]]}}
| instrument      = {{hlist|Vocals|guitar}}
| years_active    = 2017–present
| label           =
| associated_acts = {{hlist|[[Justin Bieber]]|[[Ex Battalion|JRoa]]}}
}}
| signature          =
}}

'''Anees Mokhiber''' (born July 30, 1992), known mononymously as '''anees''' (stylized in small caps), is a [[Palestinian Americans|Palestinian-American]] singer, rapper, and songwriter known for his hit song, "[[Sun and Moon (Anees song)|Sun and Moon]]", becoming the first international artist to top ''[[Billboard charts|Billboard]]'' [[Philippines Songs]] chart.

== Life and career ==

=== 1992–2024: Early life ===
Anees Mokhiber was born on July 30, 1992, in the suburbs of [[Washington, D.C.|Washington D.C.]]<ref name=":2">{{Cite web |last=Lee |first=Derrick |date=2022-01-20 |title=Anees Performs For The First Time In Los Angeles For WFNM |url=https://blurredculture.com/anees-performs-for-the-first-time-in-los-angeles-for-wfnm/ |access-date=2022-05-15 |website=Blurred Culture |language=en}}</ref> He was raised and has resided in [[Northern Virginia]].<ref name=":0" /> Anees is an Arab American with [[Palestinian]] and [[Lebanese people|Lebanese]] ancestry.<ref name=":6">{{Cite web |last=Ihmoud |first=Nader |date=2021-09-28 |title=Palestine in America — Blog — A Palestinian you should know: Anees Mokhiber |url=https://www.palestineinamerica.com/blog/a-palestinian-you-should-know-anees-mokhiber |access-date=2022-05-15 |website=Palestine in America |language=en-US}}</ref>

== Artistry and influences ==
Anees described his sound as genre-defying, ranging from [[Hip hop music|hip-hop]], [[Pop music|pop]], [[Rock music|rock]], [[Rhythm and blues|R&B]], and [[Soul music|soul]].

== Discography ==

=== Singles ===
{| class="wikitable plainrowheaders" style="text-align:center;"
|+List of singles
! Title
! Year
|-
! scope="row" | "Sun and Moon"
| 2022
|}

== References ==
{{Reflist}}

{{DEFAULTSORT:Anees}}
[[Category:Living people]]
[[Category:1992 births]]
"""


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
    mock_model, sample_page_change_event, sample_page_content, mock_article_topics
):
    """Test that preprocess method extracts and parses paragraphs from the event."""
    # Mock get_page_content and get_article_topics to avoid external API calls
    with (
        patch.object(
            mock_model,
            "get_page_content",
            new_callable=AsyncMock,
            return_value=sample_page_content,
        ),
        patch.object(
            mock_model,
            "get_article_topics",
            new_callable=AsyncMock,
            return_value=mock_article_topics,
        ),
    ):
        result = await mock_model.preprocess(sample_page_change_event)

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
        assert len(text) > 100  # Should be > 100 chars
        assert len(text) <= 500  # Should be <= 500 chars

    # Check that it contains expected content
    all_text = " ".join([p[1] for p in result["paragraphs"]])
    assert "Anees Mokhiber" in all_text
    assert "Palestinian-American" in all_text


@pytest.mark.asyncio
async def test_preprocess_extracts_metadata(
    mock_model, sample_page_change_event, sample_page_content, mock_article_topics
):
    """Test that preprocess method extracts necessary metadata from the event."""
    # Mock get_page_content and get_article_topics to avoid external API calls
    with (
        patch.object(
            mock_model,
            "get_page_content",
            new_callable=AsyncMock,
            return_value=sample_page_content,
        ),
        patch.object(
            mock_model,
            "get_article_topics",
            new_callable=AsyncMock,
            return_value=mock_article_topics,
        ),
    ):
        result = await mock_model.preprocess(sample_page_change_event)

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
    mock_model, sample_page_change_event, sample_page_content, mock_article_topics
):
    """Test that preprocess filters out unwanted sections like References."""
    # The sample_page_content already includes References section
    # Mock get_page_content and get_article_topics to avoid external API calls
    with (
        patch.object(
            mock_model,
            "get_page_content",
            new_callable=AsyncMock,
            return_value=sample_page_content,
        ),
        patch.object(
            mock_model,
            "get_article_topics",
            new_callable=AsyncMock,
            return_value=mock_article_topics,
        ),
    ):
        result = await mock_model.preprocess(sample_page_change_event)

    # Check that References and External links sections are not in the paragraphs
    section_names = [p[0] for p in result["paragraphs"]]
    assert "References" not in section_names
    assert "External links" not in section_names


@pytest.mark.asyncio
async def test_preprocess_includes_article_topics(
    mock_model, sample_page_change_event, sample_page_content, mock_article_topics
):
    """Test that preprocess includes article topics from the outlink model."""
    # Mock get_page_content and get_article_topics
    with (
        patch.object(
            mock_model,
            "get_page_content",
            new_callable=AsyncMock,
            return_value=sample_page_content,
        ),
        patch.object(
            mock_model,
            "get_article_topics",
            new_callable=AsyncMock,
            return_value=mock_article_topics,
        ),
    ):
        result = await mock_model.preprocess(sample_page_change_event)

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


@pytest.mark.skip(reason="Topic filtering temporarily disabled")
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


@pytest.mark.skip(reason="Topic filtering temporarily disabled")
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
    mock_model, sample_page_change_event, sample_page_content, mock_article_topics
):
    """Test that preprocess sets the should_process flag correctly."""
    # Mock get_page_content and get_article_topics with matching topics
    with (
        patch.object(
            mock_model,
            "get_page_content",
            new_callable=AsyncMock,
            return_value=sample_page_content,
        ),
        patch.object(
            mock_model,
            "get_article_topics",
            new_callable=AsyncMock,
            return_value=mock_article_topics,
        ),
    ):
        result = await mock_model.preprocess(sample_page_change_event)

    # Check that should_process is set to True (mock has Biography topic)
    assert "should_process" in result
    assert result["should_process"] is True


def test_extract_paragraphs_basic(mock_model):
    """Test the extract_paragraphs method with basic wikitext."""
    wikitext = """
'''Test Article''' is an article for testing. Too short.

== Section 1 ==
This is the first paragraph in section 1. It has enough text to pass the length filter. It has enough text to pass the length filter. It has enough text to pass the length filter.

This is the second paragraph in section 1. Also long enough. Also long enough. Also long enough. Also long enough. Also long enough.

== Section 2 ==
This is a paragraph in section 2. It contains sufficient text. It contains sufficient text. It contains sufficient text. It contains sufficient text.

== Section 3 ==
This is a paragraph in section 3. It has more than 500 characters. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nunc posuere, nunc vitae ultricies cursus, velit augue pulvinar mauris, id malesuada sapien erat quis augue. Suspendisse feugiat sem vitae risus malesuada, eget dictum eros porta. Sed vehicula fermentum felis, sit amet volutpat augue tristique ac. Curabitur accumsan velit leo, sed consectetur odio placerat sit amet. Pellentesque viverra rhoncus ex, eu dignissim odio dignissim nec. Aliquam erat volutpat. Vestibulum sit amet congue orci. Integer varius libero augue, sit amet aliquam nisi facilisis ac. Nam ut risus vitae justo blandit dignissim sit amet non libero. Phasellus vel iaculis nunc, nec ultricies nibh. In hac habitasse platea dictumst. Quisque ut pretium erat, varius luctus justo. Nullam faucibus bibendum metus, eget luctus lorem aliquet at. Donec interdum, ipsum porttitor luctus fermentum, magna nisl euismod risus, ac facilisis dui velit sed mi.

"""

    paragraphs = mock_model.extract_paragraphs(wikitext, "en")

    assert len(paragraphs) == 3
    section_names = [p[0] for p in paragraphs]
    assert "LEAD_SECTION" not in section_names
    assert "Section 1" in section_names
    assert "Section 2" in section_names
    assert "Section 3" not in section_names

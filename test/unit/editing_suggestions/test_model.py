import csv
from unittest.mock import patch

import pytest
from kserve.errors import InvalidInput

from src.models.editing_suggestions.model_server.model import (
    EditingSuggestionsModel,
    load_from_csv,
)

FIELDNAMES = [
    "revision_id",
    "page_title",
    "page_id",
    "suggestion_type",
    "description",
    "target",
    "wiki_id",
    "suggestion_id",
    "static_description",
    "title",
]


@pytest.fixture
def sample_rows():
    return [
        {
            "revision_id": "1",
            "page_title": "Test_Page",
            "page_id": "100",
            "suggestion_type": "MOS:GEO",
            "description": "Fix geography",
            "target": "Some target text",
            "wiki_id": "enwiki",
            "suggestion_id": "abc-123",
            "static_description": "MOS:GEO guide text",
            "title": "MOS:GEO",
        },
        {
            "revision_id": "2",
            "page_title": "Test_Page",
            "page_id": "100",
            "suggestion_type": "NPOV",
            "description": "Neutral tone",
            "target": "Other target text",
            "wiki_id": "enwiki",
            "suggestion_id": "def-456",
            "static_description": "NPOV guide text",
            "title": "NPOV",
        },
        {
            "revision_id": "3",
            "page_title": "Other_Page",
            "page_id": "200",
            "suggestion_type": "simplify_language",
            "description": "Shorter sentences",
            "target": "Long sentence here.",
            "wiki_id": "frwiki",
            "suggestion_id": "ghi-789",
            "static_description": "Simplify language guide text",
            "title": "Simplify language",
        },
    ]


@pytest.fixture
def suggestions_csv(tmp_path, sample_rows):
    path = tmp_path / "suggestions.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(sample_rows)
    return path


def test_load_from_csv(suggestions_csv):
    result = load_from_csv(str(suggestions_csv))

    assert set(result.keys()) == {"enwiki", "frwiki"}
    assert set(result["enwiki"].keys()) == {100}
    assert len(result["enwiki"][100]) == 2
    assert result["enwiki"][100][0]["suggestion_id"] == "abc-123"
    assert result["enwiki"][100][0]["page_id"] == 100
    assert result["enwiki"][100][0]["revision_id"] == 1
    assert result["frwiki"][200][0]["wiki_id"] == "frwiki"


def test_load_reads_csv_from_model_path(suggestions_csv):
    model = EditingSuggestionsModel("editing-suggestions", str(suggestions_csv))

    assert model.ready is True
    result = model.predict({"wiki_id": "enwiki", "page_id": 100})
    assert len(result["suggestions"]) == 2
    suggestion_ids = {s["suggestion_id"] for s in result["suggestions"]}
    assert suggestion_ids == {"abc-123", "def-456"}
    assert all(s["page_id"] == 100 for s in result["suggestions"])
    assert all(s["page_title"] == "Test_Page" for s in result["suggestions"])
    assert all(s["wiki_id"] == "enwiki" for s in result["suggestions"])
    assert all("suggestion_type" in s for s in result["suggestions"])
    assert all("static_description" in s for s in result["suggestions"])
    assert all("title" in s for s in result["suggestions"])


@pytest.fixture
def model(suggestions_csv):
    with patch.object(EditingSuggestionsModel, "load", return_value=None):
        m = EditingSuggestionsModel("editing-suggestions", "/unused")
        m.editing_suggestions = load_from_csv(str(suggestions_csv))
        m.ready = True
        return m


def test_predict_returns_matching_suggestions(model):
    result = model.predict({"wiki_id": "enwiki", "page_id": 100})

    assert len(result["suggestions"]) == 2
    assert all(suggestion["page_id"] == 100 for suggestion in result["suggestions"])
    assert all(
        suggestion["page_title"] == "Test_Page" for suggestion in result["suggestions"]
    )
    assert all(
        suggestion["wiki_id"] == "enwiki" for suggestion in result["suggestions"]
    )


def test_predict_returns_empty_list_for_unknown_page(model):
    result = model.predict({"wiki_id": "enwiki", "page_id": 999})

    assert result == {"suggestions": []}


def test_predict_returns_empty_list_for_unknown_wiki(model):
    result = model.predict({"wiki_id": "xxwiki", "page_id": 100})

    assert result == {"suggestions": []}


@pytest.mark.parametrize(
    "inputs,missing_param",
    [
        ({"page_id": 100}, "wiki_id"),
        ({"wiki_id": "enwiki"}, "page_id"),
    ],
)
def test_preprocess_raises_for_missing_params(model, inputs, missing_param):
    with pytest.raises(InvalidInput, match=missing_param):
        model.preprocess(inputs)


def test_preprocess_raises_for_non_string_wiki_id(model):
    with pytest.raises(InvalidInput, match="wiki_id"):
        model.preprocess({"wiki_id": 123, "page_id": 100})


def test_preprocess_raises_for_non_int_page_id(model):
    with pytest.raises(InvalidInput, match="page_id"):
        model.preprocess({"wiki_id": "enwiki", "page_id": "testingpageid"})


def test_preprocess_accepts_string_page_id(model):
    result = model.preprocess({"wiki_id": "enwiki", "page_id": "100"})

    assert result == {"wiki_id": "enwiki", "page_id": 100}


def test_preprocess_returns_inputs(model):
    result = model.preprocess({"wiki_id": "enwiki", "page_id": 100})

    assert result == {
        "wiki_id": "enwiki",
        "page_id": 100,
    }

from unittest.mock import patch

import pandas as pd
import pytest
from kserve.errors import InvalidInput

from src.models.editing_suggestions.model_server.model import (
    EditingSuggestionsModel,
    to_dict,
)


@pytest.fixture
def sample_dataframe():
    return pd.DataFrame(
        [
            {
                "revision_id": "1",
                "page_title": "Test_Page",
                "title": "MOS:GEO",
                "description": "Fix geography",
                "target": "Some target text",
                "suggestion_id": "abc-123",
                "wiki_id": "enwiki",
            },
            {
                "revision_id": "2",
                "page_title": "Test_Page",
                "title": "NPOV",
                "description": "Neutral tone",
                "target": "Other target text",
                "suggestion_id": "def-456",
                "wiki_id": "enwiki",
            },
            {
                "revision_id": "3",
                "page_title": "Other_Page",
                "title": "simplify language",
                "description": "Shorter sentences",
                "target": "Long sentence here.",
                "suggestion_id": "ghi-789",
                "wiki_id": "frwiki",
            },
        ]
    )


@pytest.fixture
def suggestions_csv(tmp_path, sample_dataframe):
    path = tmp_path / "suggestions.csv"
    sample_dataframe.to_csv(path, index=False)
    return path


def test_to_dict(sample_dataframe):
    result = to_dict(sample_dataframe)

    assert set(result.keys()) == {"enwiki", "frwiki"}
    assert len(result["enwiki"]["Test_Page"]) == 2
    assert result["enwiki"]["Test_Page"][0]["suggestion_id"] == "abc-123"
    assert result["frwiki"]["Other_Page"][0]["wiki_id"] == "frwiki"


def test_load_reads_csv_from_model_path(suggestions_csv):
    model = EditingSuggestionsModel("editing-suggestions", str(suggestions_csv))

    assert model.ready is True
    result = model.predict({"wiki_id": "enwiki", "page_title": "Test_Page"})
    assert len(result["suggestions"]) == 2
    suggestion_ids = {s["suggestion_id"] for s in result["suggestions"]}
    assert suggestion_ids == {"abc-123", "def-456"}
    assert all(s["page_title"] == "Test_Page" for s in result["suggestions"])
    assert all(s["wiki_id"] == "enwiki" for s in result["suggestions"])


@pytest.fixture
def model(sample_dataframe):
    with patch.object(EditingSuggestionsModel, "load", return_value=None):
        m = EditingSuggestionsModel("editing-suggestions", "/unused")
        m.editing_suggestions = to_dict(sample_dataframe)
        m.ready = True
        return m


def test_predict_returns_matching_suggestions(model):
    result = model.predict({"wiki_id": "enwiki", "page_title": "Test_Page"})

    assert len(result["suggestions"]) == 2
    assert all(
        suggestion["page_title"] == "Test_Page" for suggestion in result["suggestions"]
    )
    assert all(
        suggestion["wiki_id"] == "enwiki" for suggestion in result["suggestions"]
    )


def test_predict_returns_empty_list_for_unknown_page(model):
    result = model.predict({"wiki_id": "enwiki", "page_title": "Nonexistent_Page"})

    assert result == {"suggestions": []}


def test_predict_returns_empty_list_for_unknown_wiki(model):
    result = model.predict({"wiki_id": "xxwiki", "page_title": "Test_Page"})

    assert result == {"suggestions": []}


@pytest.mark.parametrize(
    "inputs,missing_param",
    [
        ({"page_title": "Test_Page"}, "wiki_id"),
        ({"wiki_id": "enwiki"}, "page_title"),
    ],
)
def test_preprocess_raises_for_missing_params(model, inputs, missing_param):
    with pytest.raises(InvalidInput, match=missing_param):
        model.preprocess(inputs)


def test_preprocess_raises_for_non_string_wiki_id(model):
    with pytest.raises(InvalidInput, match="wiki_id"):
        model.preprocess({"wiki_id": 123, "page_title": "Test_Page"})


def test_preprocess_raises_for_non_string_page_title(model):
    with pytest.raises(InvalidInput, match="page_title"):
        model.preprocess({"wiki_id": "enwiki", "page_title": 123})


def test_preprocess_returns_inputs(model):
    result = model.preprocess({"wiki_id": "enwiki", "page_title": "Test_Page"})

    assert result == {
        "wiki_id": "enwiki",
        "page_title": "Test_Page",
    }

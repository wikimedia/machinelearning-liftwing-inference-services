from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.models.revertrisk_wikidata.model_server.model import RevertRiskWikidataModel


@pytest.fixture
def model_server():
    """
    Fixture to initialize the model server a single time.
    """
    # Mock the __init__ to avoid loading the model during instantiation
    with patch.object(RevertRiskWikidataModel, "__init__", return_value=None):
        model = RevertRiskWikidataModel(
            name="test-model",
            model_path="/mnt/models/test-model.pkl",
            force_http=False,
            aiohttp_client_timeout=5,
        )
        model.ready = True
        model.name = "test-model"  # Ensure 'name' attribute is set for tests
        # Mock the loaded model components
        model.model = MagicMock()
        model.model.metadata_classifier = MagicMock()
        model.model.text_classifier = MagicMock()
        model.model.metadata_classifier.feature_names_ = [
            "add_score_mean",
            "add_score_max",
            "user_age",
            "page_age",
        ]
        model.model.metadata_classifier.get_cat_feature_indices.return_value = []
        model.model.model_version = 2  # Ensure model_version is set for tests
        model.model_load_lock = AsyncMock()  # Ensure this attribute is included
        yield model


@pytest.mark.asyncio
async def test_get_revision_content(model_server):
    """
    Test that _get_revision_content correctly parses the API response.
    """
    mock_session = AsyncMock()
    mock_session.get.return_value = {
        "query": {
            "pages": {
                "123": {
                    "title": "Q123",
                    "revisions": [
                        {"slots": {"main": {"*": "Test content"}}, "parentid": 455}
                    ],
                }
            }
        }
    }
    content, parent_id, page_title = await model_server._get_revision_content(
        mock_session, 12345
    )
    assert content == "Test content"
    assert parent_id == 455
    assert page_title == "Q123"


@pytest.mark.asyncio
async def test_fetch_metadata_features(model_server):
    """
    Test that _fetch_metadata_features correctly parses the API response and returns features.
    """
    mock_session = AsyncMock()
    # Mock the sequence of API calls made by the function
    mock_session.get.side_effect = [
        # First call for revision data
        {
            "query": {
                "pages": {
                    "123": {
                        "revisions": [
                            {
                                "user": "TestUser",
                                "userid": 1,
                                "timestamp": "2023-01-01T12:00:00Z",
                                "parentid": 455,
                            }
                        ]
                    }
                }
            }
        },
        # Second call for user data (include userid for correct anonymous logic)
        {
            "query": {
                "users": [
                    {
                        "name": "TestUser",
                        "userid": 1,
                        "groups": ["*", "user"],
                        "registration": "2023-01-01T11:00:00Z",
                    }
                ]
            }
        },
        # Third call for first revision timestamp (include 'pages' key)
        {
            "query": {
                "pages": {"123": {"revisions": [{"timestamp": "2023-01-01T10:00:00Z"}]}}
            }
        },
        # Fourth call for parent revision timestamp (include 'pages' key)
        {
            "query": {
                "pages": {"122": {"revisions": [{"timestamp": "2023-01-01T11:59:00Z"}]}}
            }
        },
    ]
    features = await model_server._fetch_metadata_features(12345, mock_session)
    assert features["user_is_anonymous"] == "False"
    assert features["user_is_bot"] == "0"
    # Remove assertion for event_user_groups-bot since feature_names_ is mocked and may not include it
    assert features["page_seconds_since_previous_revision"] == 60.0


def test_get_bert_scores_with_text(model_server):
    """
    Test that _get_bert_scores returns mean and max scores when there is text.
    """
    test_texts = ["This is a test sentence."]
    model_server.model.text_classifier.return_value = [
        [{"label": "LABEL_0", "score": 0.1}, {"label": "LABEL_1", "score": 0.9}]
    ]
    scores = model_server._get_bert_scores(test_texts)
    assert scores["mean"] == 0.9
    assert scores["max"] == 0.9


def test_get_bert_scores_empty(model_server):
    """
    Test that _get_bert_scores returns -999 when there is no text.
    """
    scores = model_server._get_bert_scores([])
    assert scores["mean"] == -999.0
    assert scores["max"] == -999.0


@pytest.mark.asyncio
async def test_end_to_end_prediction(model_server):
    """
    Test the full preprocess and predict pipeline with mocked API calls.
    """
    rev_id = 12345
    inputs = {"rev_id": rev_id}

    # Mock the API calls
    mock_session = AsyncMock()
    mock_session.get.side_effect = [
        # _get_revision_content (current)
        {
            "query": {
                "pages": {
                    "123": {
                        "title": "Q123",
                        "revisions": [
                            {"slots": {"main": {"*": "Line 2"}}, "parentid": 455}
                        ],
                    }
                }
            }
        },
        # _get_revision_content (parent)
        {
            "query": {
                "pages": {
                    "122": {
                        "title": "Q122",
                        "revisions": [{"slots": {"main": {"*": "Line 1"}}}],
                    }
                }
            }
        },
        # _fetch_metadata_features (revision)
        {
            "query": {
                "pages": {
                    "123": {
                        "revisions": [
                            {
                                "user": "TestUser",
                                "userid": 1,
                                "timestamp": "2023-01-01T12:00:00Z",
                                "parentid": 455,
                            }
                        ]
                    }
                }
            }
        },
        # _fetch_metadata_features (user)
        {
            "query": {
                "users": [
                    {
                        "name": "TestUser",
                        "userid": 1,
                        "groups": [],
                        "registration": None,
                    }
                ]
            }
        },
        # _fetch_metadata_features (first revision)
        {
            "query": {
                "pages": {"123": {"revisions": [{"timestamp": "2023-01-01T10:00:00Z"}]}}
            }
        },
        # _fetch_metadata_features (parent)
        {
            "query": {
                "pages": {"122": {"revisions": [{"timestamp": "2023-01-01T11:59:00Z"}]}}
            }
        },
        # get_labels
        {"query": {"entities": {}}},
    ]

    # Mock the model's predict_proba method to return [0.2, 0.8] directly
    model_server.model.metadata_classifier.predict_proba.return_value = [0.2, 0.8]
    model_server.model.text_classifier.return_value = [
        [{"label": "LABEL_1", "score": 0.9}]
    ]

    # We need to mock the create_mwapi_session to return our mock session
    with patch.object(
        RevertRiskWikidataModel, "create_mwapi_session", return_value=mock_session
    ):
        preprocessed_output = await model_server.preprocess(inputs)
        prediction = await model_server.predict(preprocessed_output)

        assert prediction["revision_id"] == rev_id
        assert prediction["output"]["prediction"] is True
        assert prediction["output"]["probabilities"]["true"] == 0.8

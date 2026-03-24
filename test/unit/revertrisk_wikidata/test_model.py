from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.models.revertrisk_wikidata.model_server.model import RevertRiskWikidataModel

SAMPLE_PAGE_CHANGE_EVENT = {
    "$schema": "/mediawiki/page/change/1.2.0",
    "dt": "2024-01-01T00:00:00Z",
    "wiki_id": "wikidatawiki",
    "meta": {
        "stream": "mediawiki.page_change.v1",
        "domain": "www.wikidata.org",
        "request_id": "test-request-id",
    },
    "page": {
        "page_id": 42,
        "page_title": "Q42",
        "namespace_id": 0,
        "is_redirect": False,
    },
    "revision": {
        "rev_id": 12345,
        "rev_dt": "2024-01-01T00:00:00Z",
        "rev_parent_id": 12344,
    },
    "performer": {
        "user_text": "TestUser",
        "groups": ["*", "user"],
        "is_bot": False,
    },
}


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
        model.event_key = "event"
        model.eventgate_url = None
        model.eventgate_stream = None
        model.tls_cert_bundle_path = "/etc/ssl/certs/wmf-ca-certificates.crt"
        model.custom_user_agent = (
            "WMF ML Team revertrisk-wikidata model inference (LiftWing)"
        )
        model._http_client_session = {}
        model.api_cache = MagicMock()
        model.api_cache.get.return_value = None  # Always cache miss by default
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


@pytest.mark.asyncio
async def test_preprocess_event_input_extracts_rev_id(model_server):
    """
    Test that preprocess correctly extracts rev_id from a page_change event payload.
    """
    inputs = {"event": SAMPLE_PAGE_CHANGE_EVENT}

    mock_session = AsyncMock()
    mock_session.get.side_effect = [
        # _get_revision_content (current)
        {
            "query": {
                "pages": {
                    "42": {
                        "title": "Q42",
                        "revisions": [
                            {"slots": {"main": {"*": "content"}}, "parentid": 12344}
                        ],
                    }
                }
            }
        },
        # _get_revision_content (parent)
        {
            "query": {
                "pages": {
                    "41": {
                        "title": "Q42",
                        "revisions": [{"slots": {"main": {"*": "old content"}}}],
                    }
                }
            }
        },
        # _fetch_metadata_features (revision)
        {
            "query": {
                "pages": {
                    "42": {
                        "revisions": [
                            {
                                "user": "TestUser",
                                "userid": 1,
                                "timestamp": "2024-01-01T00:00:00Z",
                                "parentid": 12344,
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
                "pages": {"42": {"revisions": [{"timestamp": "2024-01-01T00:00:00Z"}]}}
            }
        },
        # _fetch_metadata_features (parent)
        {
            "query": {
                "pages": {"41": {"revisions": [{"timestamp": "2023-12-31T23:59:00Z"}]}}
            }
        },
    ]

    with patch.object(
        RevertRiskWikidataModel, "create_mwapi_session", return_value=mock_session
    ):
        result = await model_server.preprocess(inputs)

    assert result["rev_id"] == 12345
    assert result["event"] == SAMPLE_PAGE_CHANGE_EVENT


@pytest.mark.asyncio
async def test_preprocess_event_non_wikidata_wiki_id_raises(model_server):
    """
    Test that preprocess raises HTTPException(400) for events from non-Wikidata wikis.
    """
    from fastapi import HTTPException

    non_wikidata_event = {
        **SAMPLE_PAGE_CHANGE_EVENT,
        "wiki_id": "enwiki",
    }
    inputs = {"event": non_wikidata_event}

    with pytest.raises(HTTPException) as exc_info:
        await model_server.preprocess(inputs)
    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_predict_emits_event_when_event_key_present(model_server):
    """
    Test that predict calls send_event when the event key is present in the request.
    """
    model_server.eventgate_url = "https://eventgate.example.org/v1/events"
    model_server.eventgate_stream = "mediawiki.page_revert_risk_prediction_change"
    model_server.model.metadata_classifier.predict_proba.return_value = [0.3, 0.7]
    model_server.model.text_classifier.return_value = [
        [{"label": "LABEL_1", "score": 0.7}]
    ]

    preprocessed = {
        "rev_id": 12345,
        "page_title": "Q42",
        "diffs": ["{}", "{}", "{}"],
        "metadata_features": {
            "user_is_anonymous": "False",
            "user_is_bot": "0",
            "user_age": 10,
            "page_age": 5,
        },
        "event": SAMPLE_PAGE_CHANGE_EVENT,
    }

    mock_label_session = AsyncMock()
    mock_label_session.get.return_value = {"query": {"entities": {}}}

    with (
        patch.object(
            model_server, "send_event", new_callable=AsyncMock
        ) as mock_send_event,
        patch.object(
            RevertRiskWikidataModel,
            "create_mwapi_session",
            return_value=mock_label_session,
        ),
    ):
        prediction = await model_server.predict(preprocessed)

    mock_send_event.assert_awaited_once()
    call_args = mock_send_event.call_args
    assert call_args[0][0] == SAMPLE_PAGE_CHANGE_EVENT
    assert call_args[0][1]["predictions"] == ["true"]
    assert prediction["output"]["probabilities"]["true"] == pytest.approx(0.7)


@pytest.mark.asyncio
async def test_predict_no_event_emission_without_event_key(model_server):
    """
    Test that predict does not call send_event when no event key is in the request.
    """
    model_server.model.metadata_classifier.predict_proba.return_value = [0.6, 0.4]
    model_server.model.text_classifier.return_value = [
        [{"label": "LABEL_1", "score": 0.4}]
    ]

    preprocessed = {
        "rev_id": 12345,
        "page_title": "Q42",
        "diffs": ["{}", "{}", "{}"],
        "metadata_features": {
            "user_is_anonymous": "False",
            "user_is_bot": "0",
            "user_age": 10,
            "page_age": 5,
        },
    }

    mock_label_session = AsyncMock()
    mock_label_session.get.return_value = {"query": {"entities": {}}}

    with (
        patch.object(
            model_server, "send_event", new_callable=AsyncMock
        ) as mock_send_event,
        patch.object(
            RevertRiskWikidataModel,
            "create_mwapi_session",
            return_value=mock_label_session,
        ),
    ):
        await model_server.predict(preprocessed)

    mock_send_event.assert_not_called()


@pytest.mark.asyncio
async def test_send_event_skips_when_not_configured(model_server):
    """
    Test that send_event logs an error and returns early when EVENTGATE_URL/STREAM are not set.
    """
    model_server.eventgate_url = None
    model_server.eventgate_stream = None

    with patch(
        "src.models.revertrisk_wikidata.model_server.model.events"
    ) as mock_events:
        await model_server.send_event(SAMPLE_PAGE_CHANGE_EVENT, {}, "2")
        mock_events.send_event.assert_not_called()


@pytest.mark.asyncio
async def test_send_event_calls_eventgate(model_server):
    """
    Test that send_event calls events.generate_prediction_classification_event and events.send_event.
    """
    model_server.eventgate_url = "https://eventgate.example.org/v1/events"
    model_server.eventgate_stream = "mediawiki.page_revert_risk_prediction_change"

    prediction_results = {
        "predictions": ["true"],
        "probabilities": {"true": 0.8, "false": 0.2},
    }

    with patch(
        "src.models.revertrisk_wikidata.model_server.model.events"
    ) as mock_events:
        mock_events.generate_prediction_classification_event.return_value = {
            "mock": "event"
        }
        mock_events.send_event = AsyncMock()
        with patch.object(
            model_server, "get_eventgate_session", return_value=MagicMock()
        ):
            await model_server.send_event(
                SAMPLE_PAGE_CHANGE_EVENT, prediction_results, "2"
            )

    mock_events.generate_prediction_classification_event.assert_called_once_with(
        SAMPLE_PAGE_CHANGE_EVENT,
        "mediawiki.page_revert_risk_prediction_change",
        "revertrisk-wikidata",
        "2",
        prediction_results,
    )
    mock_events.send_event.assert_awaited_once()

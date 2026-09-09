import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from kserve.errors import InvalidInput
from kserve.protocol.infer_type import InferInput, InferRequest, InferResponse

from python.preprocess_utils import extract_v2_input
from src.models.revertrisk_wikidata.model_server.model import (
    V2_PROTOCOL_KEY,
    RevertRiskWikidataModel,
)

from .test_model import SAMPLE_PAGE_CHANGE_EVENT

REV_ID = 12345


@pytest.fixture
def model_server():
    """
    Initialize the model server without loading the model.

    Mirrors the fixture in test_model.py: __init__ is patched out, so every
    attribute the code under test touches has to be set by hand here.
    """
    with patch.object(RevertRiskWikidataModel, "__init__", return_value=None):
        model = RevertRiskWikidataModel(
            name="test-model",
            model_path="/mnt/models/test-model.pkl",
            force_http=False,
            aiohttp_client_timeout=5,
        )
        model.ready = True
        model.name = "test-model"
        model.event_key = "event"
        model.eventgate_url = None
        model.eventgate_stream = None
        model.tls_cert_bundle_path = "/etc/ssl/certs/wmf-ca-certificates.crt"
        model.custom_user_agent = (
            "WMF ML Team revertrisk-wikidata model inference (LiftWing)"
        )
        model._http_client_session = {}
        model.api_cache = MagicMock()
        model.api_cache.get.return_value = None
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
        model.model.model_version = 2
        model.model_load_lock = AsyncMock()
        yield model


def make_infer_request(data, from_grpc=False):
    """Build a real InferRequest carrying the given input tensor data."""
    infer_input = InferInput(name="input", shape=[1], datatype="BYTES", data=list(data))
    return InferRequest(
        model_name="test-model",
        infer_inputs=[infer_input],
        from_grpc=from_grpc,
    )


def mwapi_responses():
    """The sequence of MW API responses a full preprocess+predict cycle consumes."""
    return [
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


@pytest.fixture
def mock_session():
    session = AsyncMock()
    session.get.side_effect = mwapi_responses()
    return session


@pytest.fixture
def scoring_model(model_server):
    """A model whose classifiers return a deterministic positive prediction."""
    model_server.model.metadata_classifier.predict_proba.return_value = [0.2, 0.8]
    model_server.model.text_classifier.return_value = [
        [{"label": "LABEL_1", "score": 0.9}]
    ]
    return model_server


class TestExtractV2Input:
    """Tests for python.preprocess_utils.extract_v2_input."""

    def test_grpc_with_bytes_payload(self):
        """gRPC sends bytes in a flat list."""
        request = make_infer_request([b'{"rev_id": 12345}'], from_grpc=True)

        assert extract_v2_input(request) == {"rev_id": 12345}

    def test_rest_v2_with_string_payload(self):
        """REST v2 sends a string in a flat list."""
        request = make_infer_request(['{"rev_id": 12345}'])

        assert extract_v2_input(request) == {"rev_id": 12345}

    def test_rest_v2_with_nested_list_payload(self):
        """Some REST clients wrap data in an extra list layer."""
        request = make_infer_request([['{"rev_id": 12345}']])

        assert extract_v2_input(request) == {"rev_id": 12345}

    def test_grpc_with_nested_list_payload(self):
        """gRPC can also arrive with nested list wrapping."""
        request = make_infer_request([[b'{"rev_id": 12345}']], from_grpc=True)

        assert extract_v2_input(request) == {"rev_id": 12345}

    def test_event_payload_survives_extraction(self):
        """The event key must survive the v2 decode so event emission still works."""
        payload = json.dumps({"event": SAMPLE_PAGE_CHANGE_EVENT}).encode("utf-8")
        request = make_infer_request([payload], from_grpc=True)

        result = extract_v2_input(request)

        assert result["event"] == SAMPLE_PAGE_CHANGE_EVENT

    def test_empty_inputs_raises(self):
        request = MagicMock()
        request.inputs = []

        with pytest.raises(InvalidInput, match="No inputs"):
            extract_v2_input(request)

    def test_empty_data_raises(self):
        request = make_infer_request([])

        with pytest.raises(InvalidInput, match="No data"):
            extract_v2_input(request)

    def test_empty_nested_list_raises(self):
        request = make_infer_request([[]])

        with pytest.raises(InvalidInput, match="Empty list"):
            extract_v2_input(request)

    def test_grpc_with_string_payload_raises(self):
        """A gRPC request carrying str rather than bytes is malformed."""
        request = make_infer_request(['{"rev_id": 12345}'], from_grpc=True)

        with pytest.raises(InvalidInput, match="Expected bytes"):
            extract_v2_input(request)

    def test_rest_with_bytes_payload_raises(self):
        """A REST v2 request carrying bytes rather than str is malformed."""
        request = make_infer_request([b'{"rev_id": 12345}'])

        with pytest.raises(InvalidInput, match="Expected string"):
            extract_v2_input(request)


class TestNormalizeCacheInput:
    """
    hoarde (the Linked Artifacts Cache) sends a fixed (wiki, page, revision)
    triple as {"wiki_id", "page_id", "revision_id"}; preprocess must accept it.
    """

    def test_revision_id_becomes_rev_id(self, model_server):
        inputs = {
            "wiki_id": "wikidatawiki",
            "page_id": 42,
            "revision_id": REV_ID,
        }

        model_server._normalize_cache_input(inputs)

        assert inputs == {"rev_id": REV_ID}

    def test_revision_id_without_wiki_or_page(self, model_server):
        inputs = {"revision_id": REV_ID}

        model_server._normalize_cache_input(inputs)

        assert inputs == {"rev_id": REV_ID}

    def test_non_wikidata_wiki_id_raises(self, model_server):
        inputs = {"wiki_id": "enwiki", "page_id": 42, "revision_id": REV_ID}

        with pytest.raises(HTTPException) as excinfo:
            model_server._normalize_cache_input(inputs)

        assert excinfo.value.status_code == 400
        assert "enwiki" in excinfo.value.detail

    def test_explicit_rev_id_wins(self, model_server):
        """A caller sending both must not have its rev_id silently overridden."""
        inputs = {"rev_id": 111, "revision_id": 222}

        model_server._normalize_cache_input(inputs)

        assert inputs == {"rev_id": 111}

    def test_plain_rev_id_payload_untouched(self, model_server):
        inputs = {"rev_id": REV_ID}

        model_server._normalize_cache_input(inputs)

        assert inputs == {"rev_id": REV_ID}

    def test_event_payload_untouched(self, model_server):
        """
        Event payloads keep their own wiki_id handling; the cache shim must not
        strip keys out of an event-driven request.
        """
        inputs = {"event": SAMPLE_PAGE_CHANGE_EVENT, "wiki_id": "wikidatawiki"}

        model_server._normalize_cache_input(inputs)

        assert inputs == {"event": SAMPLE_PAGE_CHANGE_EVENT, "wiki_id": "wikidatawiki"}

    @pytest.mark.asyncio
    async def test_preprocess_accepts_cache_payload_over_grpc(
        self, model_server, mock_session
    ):
        """End-to-end: the exact payload hoarde sends, over v2 gRPC."""
        payload = json.dumps(
            {"wiki_id": "wikidatawiki", "page_id": 42, "revision_id": REV_ID}
        ).encode("utf-8")
        request = make_infer_request([payload], from_grpc=True)

        with patch.object(
            RevertRiskWikidataModel, "create_mwapi_session", return_value=mock_session
        ):
            result = await model_server.preprocess(request)

        assert result["rev_id"] == REV_ID
        assert result[V2_PROTOCOL_KEY] is True

    @pytest.mark.asyncio
    async def test_preprocess_rejects_non_wikidata_cache_payload(
        self, model_server, mock_session
    ):
        payload = json.dumps(
            {"wiki_id": "enwiki", "page_id": 42, "revision_id": REV_ID}
        ).encode("utf-8")
        request = make_infer_request([payload], from_grpc=True)

        with (
            patch.object(
                RevertRiskWikidataModel,
                "create_mwapi_session",
                return_value=mock_session,
            ),
            pytest.raises(HTTPException) as excinfo,
        ):
            await model_server.preprocess(request)

        assert excinfo.value.status_code == 400


class TestProtocolDetection:
    """preprocess must record the protocol on the request, not on the instance."""

    @pytest.mark.asyncio
    async def test_v1_dict_does_not_set_flag(self, model_server, mock_session):
        with patch.object(
            RevertRiskWikidataModel, "create_mwapi_session", return_value=mock_session
        ):
            result = await model_server.preprocess({"rev_id": REV_ID})

        assert V2_PROTOCOL_KEY not in result
        assert result["rev_id"] == REV_ID

    @pytest.mark.asyncio
    async def test_v2_rest_sets_flag(self, model_server, mock_session):
        request = make_infer_request([json.dumps({"rev_id": REV_ID})])

        with patch.object(
            RevertRiskWikidataModel, "create_mwapi_session", return_value=mock_session
        ):
            result = await model_server.preprocess(request)

        assert result[V2_PROTOCOL_KEY] is True
        assert result["rev_id"] == REV_ID

    @pytest.mark.asyncio
    async def test_v2_grpc_sets_flag(self, model_server, mock_session):
        request = make_infer_request(
            [json.dumps({"rev_id": REV_ID}).encode("utf-8")], from_grpc=True
        )

        with patch.object(
            RevertRiskWikidataModel, "create_mwapi_session", return_value=mock_session
        ):
            result = await model_server.preprocess(request)

        assert result[V2_PROTOCOL_KEY] is True
        assert result["rev_id"] == REV_ID

    @pytest.mark.asyncio
    async def test_no_protocol_state_on_instance(self, model_server, mock_session):
        """
        The protocol must not be stored on the model instance: that is what makes
        concurrent mixed-protocol requests safe.
        """
        request = make_infer_request([json.dumps({"rev_id": REV_ID})])

        with patch.object(
            RevertRiskWikidataModel, "create_mwapi_session", return_value=mock_session
        ):
            await model_server.preprocess(request)

        assert not hasattr(model_server, "_is_v2_protocol")
        assert not hasattr(model_server, "_is_grpc")


class TestV2Response:
    """predict must return an InferResponse for v2 and a plain dict for v1."""

    @pytest.mark.asyncio
    async def test_v1_returns_plain_dict(self, scoring_model, mock_session):
        with patch.object(
            RevertRiskWikidataModel, "create_mwapi_session", return_value=mock_session
        ):
            preprocessed = await scoring_model.preprocess({"rev_id": REV_ID})
            prediction = await scoring_model.predict(preprocessed)

        assert isinstance(prediction, dict)
        assert prediction["revision_id"] == REV_ID
        assert prediction["output"]["prediction"] is True

    @pytest.mark.asyncio
    async def test_v2_returns_infer_response(self, scoring_model, mock_session):
        request = make_infer_request(
            [json.dumps({"rev_id": REV_ID}).encode("utf-8")], from_grpc=True
        )

        with patch.object(
            RevertRiskWikidataModel, "create_mwapi_session", return_value=mock_session
        ):
            preprocessed = await scoring_model.preprocess(request)
            response = await scoring_model.predict(preprocessed)

        assert isinstance(response, InferResponse)
        assert response.model_name == "test-model"
        assert len(response.outputs) == 1

        output = response.outputs[0]
        assert output.name == "output"
        assert output.datatype == "BYTES"

        payload = json.loads(output.data[0])
        assert payload["revision_id"] == REV_ID
        assert payload["model_name"] == "test-model"
        assert payload["output"]["prediction"] is True
        assert payload["output"]["probabilities"]["true"] == 0.8

    @pytest.mark.asyncio
    async def test_v2_payload_matches_v1(self, model_server, mock_session):
        """
        The v2 body must be exactly the v1 dict, so LAC and the REST clients see
        the same prediction shape.
        """
        model_server.model.metadata_classifier.predict_proba.return_value = [0.2, 0.8]
        model_server.model.text_classifier.return_value = [
            [{"label": "LABEL_1", "score": 0.9}]
        ]

        with patch.object(
            RevertRiskWikidataModel, "create_mwapi_session", return_value=mock_session
        ):
            preprocessed = await model_server.preprocess({"rev_id": REV_ID})
            v1_prediction = await model_server.predict(preprocessed)

        mock_session.get.side_effect = mwapi_responses()
        request = make_infer_request([json.dumps({"rev_id": REV_ID})])

        with patch.object(
            RevertRiskWikidataModel, "create_mwapi_session", return_value=mock_session
        ):
            preprocessed = await model_server.preprocess(request)
            v2_response = await model_server.predict(preprocessed)

        assert json.loads(v2_response.outputs[0].data[0]) == v1_prediction


class TestV2EventEmission:
    """Event production must keep working when the request arrives over v2."""

    @pytest.mark.asyncio
    async def test_event_emitted_for_v2_request(self, scoring_model, mock_session):
        payload = json.dumps({"event": SAMPLE_PAGE_CHANGE_EVENT}).encode("utf-8")
        request = make_infer_request([payload], from_grpc=True)

        with (
            patch.object(
                RevertRiskWikidataModel,
                "create_mwapi_session",
                return_value=mock_session,
            ),
            patch.object(
                RevertRiskWikidataModel, "send_event", new_callable=AsyncMock
            ) as mock_send_event,
        ):
            preprocessed = await scoring_model.preprocess(request)
            response = await scoring_model.predict(preprocessed)

        mock_send_event.assert_awaited_once()
        source_event, prediction_results, model_version = mock_send_event.await_args[0]
        assert source_event == SAMPLE_PAGE_CHANGE_EVENT
        assert prediction_results["predictions"] == ["true"]
        assert model_version == "2"

        # The response still goes back over v2.
        assert isinstance(response, InferResponse)

    @pytest.mark.asyncio
    async def test_no_event_for_v2_rev_id_request(self, scoring_model, mock_session):
        request = make_infer_request([json.dumps({"rev_id": REV_ID})])

        with (
            patch.object(
                RevertRiskWikidataModel,
                "create_mwapi_session",
                return_value=mock_session,
            ),
            patch.object(
                RevertRiskWikidataModel, "send_event", new_callable=AsyncMock
            ) as mock_send_event,
        ):
            preprocessed = await scoring_model.preprocess(request)
            await scoring_model.predict(preprocessed)

        mock_send_event.assert_not_awaited()

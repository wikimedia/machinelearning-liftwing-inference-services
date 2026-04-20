import json
from unittest.mock import MagicMock, patch

import pytest
from kserve.errors import InvalidInput
from kserve.protocol.infer_type import InferRequest

from src.models.outlink_topic_model.model_server.model import OutlinksTopicModel


@pytest.fixture
def model():
    """Create a model instance with mocked load."""
    with patch.object(OutlinksTopicModel, "load", return_value=True):
        m = OutlinksTopicModel("test-model")
        m.ready = True
        return m


class TestExtractV2Input:
    """Tests for _extract_v2_input method."""

    def test_grpc_with_bytes_payload(self, model):
        """gRPC sends bytes in a flat list."""
        model._is_grpc = True

        input_tensor = MagicMock()
        input_tensor.data = [b'{"page_id": 5355, "lang": "en"}']

        infer_request = MagicMock()
        infer_request.inputs = [input_tensor]

        result = model._extract_v2_input(infer_request)

        assert result == {"page_id": 5355, "lang": "en"}

    def test_rest_v2_with_string_payload(self, model):
        """REST v2 sends string in a flat list."""
        model._is_grpc = False

        input_tensor = MagicMock()
        input_tensor.data = ['{"page_id": 5355, "lang": "en"}']

        infer_request = MagicMock()
        infer_request.inputs = [input_tensor]

        result = model._extract_v2_input(infer_request)

        assert result == {"page_id": 5355, "lang": "en"}

    def test_rest_v2_with_nested_list_payload(self, model):
        """Some REST clients wrap data in an extra list layer."""
        model._is_grpc = False

        input_tensor = MagicMock()
        input_tensor.data = [['{"page_id": 5355, "lang": "en"}']]

        infer_request = MagicMock()
        infer_request.inputs = [input_tensor]

        result = model._extract_v2_input(infer_request)

        assert result == {"page_id": 5355, "lang": "en"}

    def test_grpc_with_nested_list_payload(self, model):
        """gRPC can also have nested list wrapping."""
        model._is_grpc = True

        input_tensor = MagicMock()
        input_tensor.data = [[b'{"page_id": 5355, "lang": "en"}']]

        infer_request = MagicMock()
        infer_request.inputs = [input_tensor]

        result = model._extract_v2_input(infer_request)

        assert result == {"page_id": 5355, "lang": "en"}

    def test_empty_inputs_raises_error(self, model):
        """Empty inputs list raises InvalidInput."""
        infer_request = MagicMock()
        infer_request.inputs = []

        with pytest.raises(InvalidInput, match="No inputs in v2 request"):
            model._extract_v2_input(infer_request)

    def test_empty_data_raises_error(self, model):
        """Empty data list raises InvalidInput."""
        model._is_grpc = False

        input_tensor = MagicMock()
        input_tensor.data = []

        infer_request = MagicMock()
        infer_request.inputs = [input_tensor]

        with pytest.raises(InvalidInput, match="No data in v2 request input tensor"):
            model._extract_v2_input(infer_request)

    def test_grpc_wrong_type_raises_error(self, model):
        """gRPC receiving string instead of bytes raises InvalidInput."""
        model._is_grpc = True

        input_tensor = MagicMock()
        input_tensor.data = ['{"page_id": 5355, "lang": "en"}']

        infer_request = MagicMock()
        infer_request.inputs = [input_tensor]

        with pytest.raises(InvalidInput, match="Expected bytes for gRPC request"):
            model._extract_v2_input(infer_request)

    def test_rest_wrong_type_raises_error(self, model):
        """REST receiving bytes instead of string raises InvalidInput."""
        model._is_grpc = False

        input_tensor = MagicMock()
        input_tensor.data = [b'{"page_id": 5355, "lang": "en"}']

        infer_request = MagicMock()
        infer_request.inputs = [input_tensor]

        with pytest.raises(InvalidInput, match="Expected string for REST request"):
            model._extract_v2_input(infer_request)


class TestPreprocessProtocolDetection:
    """Tests for v1 vs v2 protocol detection in preprocess."""

    @pytest.mark.asyncio
    async def test_v1_input_sets_flags_false(self, model):
        """v1 dict input should set both flags to False."""
        v1_input = {"page_id": 5355, "lang": "en", "threshold": 0.5}

        model._is_v2_protocol = True  # simulate previous v2 call
        model._is_grpc = True

        # Mock the rest to avoid actual MW API calls
        with patch.object(
            model, "retrieve_page_id_and_title", return_value=(5355, None)
        ):
            with patch(
                "src.models.outlink_topic_model.model_server.model.validate_json_input",
                side_effect=lambda x: x,
            ):
                with patch(
                    "src.models.outlink_topic_model.model_server.model.get_lang",
                    return_value="en",
                ):
                    await model.preprocess(v1_input)

        assert model._is_v2_protocol is False
        assert model._is_grpc is False

    @pytest.mark.asyncio
    async def test_v2_rest_input_sets_flags_correctly(self, model):
        """v2 REST input should set flags correctly: v2=True, gRPC=False."""
        input_tensor = MagicMock()
        input_tensor.data = ['{"page_id": 5355, "lang": "en", "threshold": 0.5}']

        infer_request = MagicMock(spec=InferRequest)
        infer_request.inputs = [input_tensor]
        infer_request.from_grpc = False

        with patch.object(
            model,
            "_extract_v2_input",
            return_value={"page_id": 5355, "lang": "en", "threshold": 0.5},
        ):
            with patch.object(
                model, "retrieve_page_id_and_title", return_value=(5355, None)
            ):
                with patch(
                    "src.models.outlink_topic_model.model_server.model.validate_json_input",
                    side_effect=lambda x: x,
                ):
                    with patch(
                        "src.models.outlink_topic_model.model_server.model.get_lang",
                        return_value="en",
                    ):
                        await model.preprocess(infer_request)

        assert model._is_v2_protocol is True
        assert model._is_grpc is False

    @pytest.mark.asyncio
    async def test_v2_grpc_input_sets_flags_correctly(self, model):
        """v2 gRPC input should set flags correctly: v2=True, gRPC=True."""
        input_tensor = MagicMock()
        input_tensor.data = [b'{"page_id": 5355, "lang": "en", "threshold": 0.5}']

        infer_request = MagicMock(spec=InferRequest)
        infer_request.inputs = [input_tensor]
        infer_request.from_grpc = True

        with patch.object(
            model,
            "_extract_v2_input",
            return_value={"page_id": 5355, "lang": "en", "threshold": 0.5},
        ):
            with patch.object(
                model, "retrieve_page_id_and_title", return_value=(5355, None)
            ):
                with patch(
                    "src.models.outlink_topic_model.model_server.model.validate_json_input",
                    side_effect=lambda x: x,
                ):
                    with patch(
                        "src.models.outlink_topic_model.model_server.model.get_lang",
                        return_value="en",
                    ):
                        await model.preprocess(infer_request)

        assert model._is_v2_protocol is True
        assert model._is_grpc is True


class TestPostprocess:
    """Tests for postprocess method."""

    @pytest.mark.asyncio
    async def test_v1_returns_prediction_format(self, model):
        """v1 should return prediction format with article URL and results."""
        model._is_v2_protocol = False

        result = {
            "topics": [["Culture.Food_and_drink", 0.95]],
            "lang": "en",
            "page_id": 5355,
            "page_title": "Douglas_Adams",
        }

        response = await model.postprocess(result)

        assert "prediction" in response
        assert (
            response["prediction"]["article"]
            == "https://en.wikipedia.org/wiki/Douglas_Adams"
        )
        assert response["prediction"]["results"] == [
            {"topic": "Culture.Food_and_drink", "score": 0.95}
        ]

    @pytest.mark.asyncio
    async def test_v2_returns_infer_response(self, model):
        """v2 should return InferResponse with encoded data."""
        model._is_v2_protocol = True
        model._is_grpc = False

        result = {"topics": [["Culture.Food_and_drink", 0.95]], "lang": "en"}

        response = await model.postprocess(result)

        assert hasattr(response, "model_name")
        assert response.model_name == "test-model"
        assert len(response.outputs) == 1

        # Check the data is properly encoded JSON string
        output_data = response.outputs[0].data[0]
        parsed = json.loads(output_data)
        assert parsed == result

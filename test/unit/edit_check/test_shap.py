import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


def mock_modules(modules):
    for mod in modules:
        sys.modules[mod] = MagicMock()


# Mocking the modules that are not available (or relevant) in the test environment
mock_modules(
    [
        "kserve",
        "kserve.errors",
        "kserve.constants",
        "fastapi.middleware.cors",
        "torch",
        "transformers",
    ]
)

import kserve  # noqa: E402

kserve.Model = type("DummyKserveModel", (), {})
kserve.constants.KSERVE_LOGLEVEL = 0

from src.models.edit_check.model_server.model import EditCheckModel  # noqa: E402

# Create a dummy EditCheckModel instance to avoid calling __init__ attempting to load a model
dummy_model = EditCheckModel.__new__(EditCheckModel)
dummy_model.name = "dummy_model"


@pytest.mark.asyncio
async def test_preprocess_text_for_explanation():
    inputs = {
        "instances": [
            {
                "lang": "en",
                "check_type": "peacock",
                "original_text": "original text 1",
                "modified_text": "modified text 1",
            },
            {
                "lang": "en",
                "check_type": "peacock",
                "original_text": "original text 2",
                "modified_text": "modified text 2",
                "return_shap_values": True,
            },
        ]
    }
    # Call preprocess
    _, text_for_explanation, _ = await dummy_model.preprocess(inputs)
    assert len(text_for_explanation) == 1
    assert text_for_explanation == ["modified text 2"]


class DummyExplainerOutput:
    def __init__(self, data, values):
        self.data = data
        self.values = values


class DummyRequestInstance:
    def __init__(self, check_type="peacock", lang="en", return_shap_values=False):
        self.check_type = check_type
        self.lang = lang
        self.return_shap_values = return_shap_values


@pytest.mark.asyncio
@patch("shap.plots._text.unpack_shap_explanation_contents")
@patch("shap.plots._text.process_shap_values")
async def test_postprocess_details(mock_process_shap, mock_unpack_shap):
    # Mock unpack_shap_explanation_contents
    mock_unpack_shap.return_value = (
        np.array([[0, 0.5], [0, 0.7], [0, 0.2]]),
        "clustering",
    )

    # Mock process_shap_values return tokens and shap values
    mock_tokens = np.array(["token1", "token2", "token3"])
    mock_values = np.array([0.5, 0.7, 0.2])
    mock_process_shap.return_value = (mock_tokens, mock_values, None)
    sorted_shap = [("token2", 0.7), ("token1", 0.5), ("token3", 0.2)]

    predictions = (
        # model_outputs: original and modified (length must be even)
        [
            {"label": "dummy_0", "score": 0.1},
            {"label": "dummy_1", "score": 0.9},
            {"label": "dummy_0", "score": 0.1},
            {"label": "dummy_1", "score": 0.9},
        ],
        # explainer_outputs
        [
            DummyExplainerOutput(data="some data", values="some values"),
        ],
        # processed_requests
        {
            "Valid": [
                {
                    "index": 0,
                    "status_code": 200,
                    "instance": DummyRequestInstance(),
                },
                {
                    "index": 1,
                    "status_code": 200,
                    "instance": DummyRequestInstance(return_shap_values=True),
                },
            ],
            "Malformed": [],
        },
    )
    # Call postprocess
    response = await dummy_model.postprocess(predictions)
    predictions_list = response["predictions"]

    assert predictions_list[0]["details"] == {}
    assert predictions_list[1]["details"]["violations"] == sorted_shap
    assert predictions_list[1]["details"]["shap_values"] == mock_values.tolist()
    assert predictions_list[1]["details"]["tokens"] == mock_tokens.tolist()

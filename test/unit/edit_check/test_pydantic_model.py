import sys
from unittest.mock import MagicMock
import asyncio
import pytest
from pydantic import ValidationError


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

from src.models.edit_check.model_server.request_model import RequestModel  # noqa: E402
from src.models.edit_check.model_server.model import EditCheckModel  # noqa: E402


def test_valid_instance():
    """
    Create a valid instance with proper text differences.
    """
    valid_data = {
        "instances": [
            {
                "lang": "en",
                "check_type": "peacock",
                "original_text": "original text of a paragraph",
                "modified_text": "modified text of a paragraph!",
            }
        ]
    }
    model = RequestModel(**valid_data)
    responses = model.process_instances()
    assert len(responses["Valid"]) == 1
    assert len(responses["Malformed"]) == 0
    instance = responses["Valid"][0]["instance"]
    assert instance.lang == "en"
    assert instance.check_type == "peacock"


def test_identical_texts():
    """
    Test instance where original_text and modified_text are identical.
    """
    invalid_data = {
        "instances": [
            {
                "lang": "en",
                "check_type": "peacock",
                "original_text": "Same text",
                "modified_text": "Same text",
            }
        ]
    }
    model = RequestModel(**invalid_data)
    responses = model.process_instances()
    assert len(responses["Valid"]) == 0
    assert len(responses["Malformed"]) == 1
    error_msg = responses["Malformed"][0]["errors"][0]
    assert "Original text and modified text must be different" in error_msg


def test_empty_instances():
    """
    Test that an empty instances list raises a ValidationError.
    """
    with pytest.raises(ValidationError):
        RequestModel(instances=[])


def test_postprocess_sorting_order():
    """
    Test that the postprocess function sorts the predictions based on the index.
    """

    # Create a dummy instance to simulate the pydantic instance attribute.
    class DummyInstance:
        def __init__(self, check_type, lang):
            self.check_type = check_type
            self.lang = lang

    # Create a dummy EditCheckModel instance to avoid calling __init__ attempting to load a model
    dummy_model = EditCheckModel.__new__(EditCheckModel)
    dummy_model.name = "dummy_model"

    # Create valid requests in unsorted order.
    valid1 = {
        "index": 2,
        "status_code": 200,
        "instance": DummyInstance("peacock", "en"),
    }
    valid2 = {
        "index": 1,
        "status_code": 200,
        "instance": DummyInstance("peacock", "en"),
    }
    malformed1 = {"index": 0, "status_code": 400, "errors": ["Malformed request"]}
    valid_requests = [valid1, valid2]

    # Create corresponding model outputs for each valid request.
    # Each valid request contributes two outputs (original and modified).
    model_outputs = [
        {"label": "dummy_0", "score": 0.6},
        {"label": "dummy_1", "score": 0.7},
        {"label": "dummy_0", "score": 0.8},
        {"label": "dummy_1", "score": 0.9},
    ]

    # Assemble the processed_requests dictionary.
    processed_requests = {"Valid": valid_requests, "Malformed": [malformed1]}
    predictions_tuple = (model_outputs, processed_requests)

    # Call postprocess with return_index True to preserve the index for verification.
    final_response = asyncio.run(
        dummy_model.postprocess(predictions_tuple, return_index=True)
    )
    predictions_list = final_response["predictions"]

    # Extract all indexes and assert they are in ascending order.
    indexes = [prediction["index"] for prediction in predictions_list]
    assert indexes == sorted(indexes)

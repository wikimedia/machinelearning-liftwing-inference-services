from typing import Any
from unittest.mock import patch

import pytest
from knowledge_integrity.schema import Revision

from src.models.revert_risk_model.model_server.base_model import RevisionRevertRiskModel


@pytest.fixture
def input_data() -> dict[str, Any]:
    return {
        "revision_data": {
            "id": 1234,
            "bytes": 2800,
            "comment": "Added category baz.",
            "text": """This is a lead.
                == Section I ==
                Section I body. {{and a|template}}
                === Section I.A ===
                Section I.A [[body]].
                === Section I.B ===
                Section I.B body.

                [[Category:bar]]
                [[Category:baz]]
            """,
            "timestamp": "2022-02-15T04:30:00Z",
            "tags": [],
            "parent": {
                "id": 1200,
                "bytes": 2600,
                "comment": "Added section I.B",
                "text": """This is a lead.
                == Section I ==
                Section I body. {{and a|template}}
                === Section I.A ===
                Section I.A [[body]].
                === Section I.B ===
                Section I.B body.

                [[Category:bar]]
            """,
                "timestamp": "2021-01-01T02:00:00Z",
                "tags": [],
                "lang": "en",
            },
            "user": {
                "id": 0,  # id is 0 when the user is anonymous
            },
            "page": {
                "id": 1008,
                "title": "this is a title",
                "first_edit_timestamp": "2018-01-01T10:02:02Z",
            },
            "lang": "en",
        }
    }


def test_get_revision_from_input(input_data: str):
    with patch.object(RevisionRevertRiskModel, "load", return_value=None):
        model = RevisionRevertRiskModel(
            "my_model",
            "revertrisk",
            "my_model_path",
            "my_wiki_url",
            5,
            False,
            False,
        )
        preprocessed_data = model.get_revision_from_input(input_data)
        assert isinstance(preprocessed_data["revision"], Revision)
        assert preprocessed_data["rev_id"] == -1
        assert preprocessed_data["lang"] == "en"

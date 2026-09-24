"""Tests for the domain mode of the reference-risk model server.

The model server imports kserve, mwapi, aiohttp and knowledge_integrity. The
test environment installs none of them, so this module replaces them, the same
way `test/unit/edit_check/test_pydantic_model.py` does.

The tests cover the code of the model server only. `parse_url_domain` and
`normalize_domain` belong to knowledge_integrity and have their own tests
upstream, so the tests here replace them and assert the value that the model
server sends to them.
"""

import asyncio
import sys
from unittest.mock import MagicMock

import pytest


def mock_modules(modules):
    for mod in modules:
        sys.modules[mod] = MagicMock()


# Mocking the modules that are not available (or relevant) in the test environment
mock_modules(
    [
        "aiohttp",
        "kserve",
        "kserve.constants",
        "kserve.errors",
        "kserve.logging",
        "kserve.utils",
        "knowledge_integrity",
        "knowledge_integrity.mediawiki",
        "knowledge_integrity.models",
        "knowledge_integrity.models.reference_need",
        "knowledge_integrity.models.reference_risk",
        "knowledge_integrity.models.reference_risk.domain",
        "mwapi",
        "prometheus_client",
        "yaml",
    ]
)

import kserve  # noqa: E402


class DummyInvalidInput(Exception):
    pass


class DummyInferenceError(Exception):
    pass


# `python.preprocess_utils` and the model server both raise these classes. Both
# modules must raise the same class, so set them before either module imports.
kserve_errors_mock = MagicMock()
kserve_errors_mock.InvalidInput = DummyInvalidInput
kserve_errors_mock.InferenceError = DummyInferenceError
sys.modules["kserve.errors"] = kserve_errors_mock

kserve.Model = type("DummyKserveModel", (), {})
kserve.constants.KSERVE_LOGLEVEL = 0

from src.models.reference_quality.model_server import model as ref_model  # noqa: E402


class FakeDomainMetadata:
    """Stands in for `knowledge_integrity` DomainMetadata."""

    def __init__(
        self,
        survival_ratio=0.92,
        page_count=123,
        editors_count=3450,
        ps_label_local="Generally reliable",
        ps_label_enwiki="No consensus",
        is_risky=False,
    ):
        self.survival_ratio = survival_ratio
        self.page_count = page_count
        self.editors_count = editors_count
        self.ps_label_local = ps_label_local
        self.ps_label_enwiki = ps_label_enwiki
        self.is_risky = is_risky


@pytest.fixture
def risk_model():
    """A ReferenceRiskModel that loads no model file.

    `__init__` opens the sqlite database of domain metadata, so the fixture
    bypasses it and sets only the attributes that the domain path reads.
    """
    instance = ref_model.ReferenceRiskModel.__new__(ref_model.ReferenceRiskModel)
    instance.name = "reference-risk"
    instance.model = MagicMock()
    instance.model.supported_wikis = {"en", "fr"}
    instance.model.model_version = "2024-07"
    instance.model._fetch_domain_metadata.return_value = {}
    return instance


@pytest.fixture
def identity_domain_parsing(monkeypatch):
    """Replace the two knowledge_integrity functions with a record of the calls.

    `parse_url_domain` returns the value it receives, so a test can assert the
    exact string that the model server builds from the input.
    """
    calls = []

    def fake_parse_url_domain(url):
        calls.append(url)
        return url

    monkeypatch.setattr(ref_model, "parse_url_domain", fake_parse_url_domain)
    monkeypatch.setattr(ref_model, "normalize_domain", lambda domain: domain)
    return calls


def preprocess(model, inputs):
    return asyncio.run(model.preprocess(inputs))


class TestParseDomainInput:
    def test_bare_domain_gets_a_scheme_relative_prefix(self, identity_domain_parsing):
        # urlparse reads a bare domain as a path, so the model server must add
        # the prefix. Without it the domain never reaches the database lookup.
        result = ref_model.ReferenceRiskModel.parse_domain_input("example.com")
        assert identity_domain_parsing == ["//example.com"]
        assert result == "//example.com"

    def test_bare_domain_with_a_path_gets_the_prefix(self, identity_domain_parsing):
        ref_model.ReferenceRiskModel.parse_domain_input("example.com/a/b")
        assert identity_domain_parsing == ["//example.com/a/b"]

    def test_full_url_keeps_its_scheme(self, identity_domain_parsing):
        ref_model.ReferenceRiskModel.parse_domain_input("https://example.com/a")
        assert identity_domain_parsing == ["https://example.com/a"]

    def test_scheme_relative_input_keeps_its_prefix(self, identity_domain_parsing):
        ref_model.ReferenceRiskModel.parse_domain_input("//example.com")
        assert identity_domain_parsing == ["//example.com"]

    def test_scheme_less_archive_url_gets_the_prefix(self, identity_domain_parsing):
        # An archive URL holds a second scheme in its path. Only the start of
        # the input decides whether the first scheme is absent.
        archive = "web.archive.org/web/2018/https://example.com/a"
        ref_model.ReferenceRiskModel.parse_domain_input(archive)
        assert identity_domain_parsing == [f"//{archive}"]

    def test_non_string_domain_raises_invalid_input(self, identity_domain_parsing):
        # JSON allows any type here. Without the guard the membership test
        # raises a TypeError, and the caller gets a 500 in place of a 400.
        with pytest.raises(DummyInvalidInput):
            ref_model.ReferenceRiskModel.parse_domain_input(123)

    def test_over_long_domain_raises_invalid_input(self, identity_domain_parsing):
        # The length check runs before the regular expressions read the input,
        # so the error names the limit and not a parse failure.
        long_domain = "a" * (ref_model.MAX_DOMAIN_INPUT_LENGTH + 1) + ".com"
        with pytest.raises(DummyInvalidInput, match="maximum"):
            ref_model.ReferenceRiskModel.parse_domain_input(long_domain)
        assert identity_domain_parsing == []

    def test_domain_below_the_length_limit_is_parsed(self, identity_domain_parsing):
        below_limit = "a" * (ref_model.MAX_DOMAIN_INPUT_LENGTH - 5) + ".com"
        ref_model.ReferenceRiskModel.parse_domain_input(below_limit)
        assert identity_domain_parsing == [f"//{below_limit}"]

    def test_unparseable_domain_raises_invalid_input(self, monkeypatch):
        monkeypatch.setattr(ref_model, "parse_url_domain", lambda url: None)
        monkeypatch.setattr(ref_model, "normalize_domain", lambda domain: domain)
        with pytest.raises(DummyInvalidInput):
            ref_model.ReferenceRiskModel.parse_domain_input("!!!")

    def test_domain_that_does_not_normalize_raises_invalid_input(self, monkeypatch):
        monkeypatch.setattr(ref_model, "parse_url_domain", lambda url: "localhost")
        monkeypatch.setattr(ref_model, "normalize_domain", lambda domain: None)
        with pytest.raises(DummyInvalidInput):
            ref_model.ReferenceRiskModel.parse_domain_input("localhost")


class TestPreprocess:
    def test_domain_request_makes_no_mediawiki_call(
        self, risk_model, identity_domain_parsing
    ):
        # The parent class reads the revision from the MediaWiki API. The domain
        # path must not call it, because the database holds all the data.
        risk_model.get_http_client_session = MagicMock(
            side_effect=AssertionError("the domain path must open no session")
        )
        result = preprocess(risk_model, {"lang": "en", "domain": "example.com"})
        assert result["normalized_domain"] == "//example.com"

    def test_rev_id_and_domain_together_raise_invalid_input(
        self, risk_model, identity_domain_parsing
    ):
        with pytest.raises(DummyInvalidInput):
            preprocess(
                risk_model, {"lang": "en", "rev_id": 1234, "domain": "example.com"}
            )

    def test_empty_domain_raises_invalid_input(
        self, risk_model, identity_domain_parsing
    ):
        with pytest.raises(DummyInvalidInput):
            preprocess(risk_model, {"lang": "en", "domain": ""})

    def test_missing_lang_raises_invalid_input(
        self, risk_model, identity_domain_parsing
    ):
        with pytest.raises(DummyInvalidInput):
            preprocess(risk_model, {"domain": "example.com"})

    def test_lang_with_a_wiki_suffix_raises_invalid_input(
        self, risk_model, identity_domain_parsing
    ):
        with pytest.raises(DummyInvalidInput):
            preprocess(risk_model, {"lang": "enwiki", "domain": "example.com"})

    def test_unsupported_lang_raises_invalid_input(
        self, risk_model, identity_domain_parsing
    ):
        with pytest.raises(DummyInvalidInput):
            preprocess(risk_model, {"lang": "zz", "domain": "example.com"})

    def test_rev_id_request_with_unsupported_lang_makes_no_mediawiki_call(
        self, risk_model
    ):
        # The parent class does not check the supported wikis, so the check
        # must run before the request goes to the parent class.
        risk_model.get_http_client_session = MagicMock(
            side_effect=AssertionError("an unsupported lang must open no session")
        )
        with pytest.raises(DummyInvalidInput, match="Unsupported lang"):
            preprocess(risk_model, {"lang": "zz", "rev_id": 1234})

    def test_rev_id_request_with_supported_lang_goes_to_the_parent(
        self, risk_model, monkeypatch
    ):
        async def fake_parent_preprocess(self, inputs, headers=None):
            inputs["revision"] = "parent revision"
            return inputs

        monkeypatch.setattr(
            ref_model.ReferenceNeedModel, "preprocess", fake_parent_preprocess
        )
        result = preprocess(risk_model, {"lang": "en", "rev_id": 1234})
        assert result["revision"] == "parent revision"
        assert "normalized_domain" not in result


class TestPredictDomain:
    def test_known_domain_gives_its_metadata(self, risk_model):
        risk_model.model._fetch_domain_metadata.return_value = {
            "example.com": FakeDomainMetadata()
        }
        output = risk_model.predict({"lang": "en", "normalized_domain": "example.com"})
        risk_model.model._fetch_domain_metadata.assert_called_once_with(
            wiki_db="enwiki", domains=["example.com"]
        )
        assert output == {
            "model_name": "reference-risk",
            "model_version": "2024-07",
            "wiki_db": "enwiki",
            "domain": "example.com",
            "domain_metadata": {
                "survival_ratio": 0.92,
                "page_count": 123,
                "editors_count": 3450,
                "ps_label_local": "Generally reliable",
                "ps_label_enwiki": "No consensus",
                "is_risky": False,
            },
        }

    def test_risky_domain_reports_is_risky(self, risk_model):
        risk_model.model._fetch_domain_metadata.return_value = {
            "bad.example.com": FakeDomainMetadata(
                ps_label_local="Deprecated", is_risky=True
            )
        }
        output = risk_model.predict(
            {"lang": "en", "normalized_domain": "bad.example.com"}
        )
        assert output["domain_metadata"]["is_risky"] is True
        assert output["domain_metadata"]["ps_label_local"] == "Deprecated"

    def test_unknown_domain_gives_null_metadata(self, risk_model):
        # The database has no row for every domain. The response says so
        # instead of failing.
        risk_model.model._fetch_domain_metadata.return_value = {}
        output = risk_model.predict(
            {"lang": "en", "normalized_domain": "unknown.example.com"}
        )
        assert output["domain_metadata"] is None
        assert output["domain"] == "unknown.example.com"

    def test_predict_still_routes_a_revision_request(self, risk_model):
        # A request with no normalized_domain must keep the revision behaviour.
        result = MagicMock()
        result.model_version = "2024-07"
        result.reference_count = 2
        result.survival_ratio = MagicMock()
        result.reference_risk_score = 0.5
        risk_model.model.classify.return_value = result
        output = risk_model.predict(
            {"lang": "en", "rev_id": 1234, "revision": MagicMock()}
        )
        risk_model.model._fetch_domain_metadata.assert_not_called()
        assert output["revision_id"] == 1234
        assert output["reference_risk_score"] == 0.5

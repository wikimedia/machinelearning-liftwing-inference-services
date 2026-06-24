import pytest

from python.events import (
    _build_page_entity,
    _build_revision_entity,
    _build_user_entity,
    generate_prediction_classification_event,
)

# Shared fixtures

PREDICTION_RESULTS = {
    "predictions": ["yes"],
    "probabilities": {"yes": 0.9, "no": 0.1},
}

MINIMAL_PAGE_CHANGE = {
    "$schema": "/mediawiki/page/change/1.2.0",
    "changelog_kind": "update",
    "page_change_kind": "edit",
    "dt": "2021-01-01T00:00:00.0Z",
    "meta": {
        "stream": "mediawiki.page_change.v1",
        "domain": "en.wikipedia.org",
        "id": "abc123",
        "request_id": "req123",
    },
    "page": {
        "page_id": 1,
        "page_title": "Example",
        "namespace_id": 0,
        "is_redirect": False,
    },
    "revision": {
        "rev_id": 2,
        "rev_dt": "2021-01-01T00:00:00.0Z",
    },
    "wiki_id": "enwiki",
}


# _build_user_entity


def test_build_user_entity_keeps_known_fields():
    user = {
        "user_text": "Alice",
        "user_id": 42,
        "user_central_id": 99,
        "edit_count": 100,
        "groups": ["sysop"],
        "is_bot": False,
        "is_system": False,
        "is_temp": False,
        "registration_dt": "2020-01-01T00:00:00.0Z",
        "wiki_id": "enwiki",
    }
    assert _build_user_entity(user) == user


def test_build_user_entity_strips_unknown_fields():
    user = {
        "user_text": "Alice",
        "is_bot": False,
        "new_upstream_field": "should be stripped",
    }
    result = _build_user_entity(user)
    assert "new_upstream_field" not in result
    assert result == {"user_text": "Alice", "is_bot": False}


def test_build_user_entity_handles_missing_optional_fields():
    user = {"user_text": "Alice"}
    result = _build_user_entity(user)
    assert result == {"user_text": "Alice"}


def test_build_user_entity_propagates_wiki_id():
    # wiki_id is allowed by user/1.2.0 and propagated so consumers can tell
    # which wiki a (possibly global) account belongs to.
    user = {"user_text": "Alice", "wiki_id": "commonswiki"}
    result = _build_user_entity(user)
    assert result["wiki_id"] == "commonswiki"


# _build_page_entity


def test_build_page_entity_keeps_known_fields():
    page = {
        "page_id": 1,
        "page_title": "Example",
        "namespace_id": 0,
        "namespace_is_content": True,
        "is_redirect": False,
        "revision_count": 5,
    }
    assert _build_page_entity(page) == page


def test_build_page_entity_strips_unknown_fields():
    page = {
        "page_id": 1,
        "page_title": "Example",
        "new_upstream_field": "should be stripped",
    }
    result = _build_page_entity(page)
    assert "new_upstream_field" not in result
    assert result == {"page_id": 1, "page_title": "Example"}


def test_build_page_entity_propagates_namespace_is_content():
    # namespace_is_content is allowed by page/2.2.0 and propagated so consumers
    # can filter for content pages without hardcoding per-wiki namespace_ids.
    page = {
        "page_id": 1,
        "page_title": "Example",
        "namespace_is_content": True,
    }
    result = _build_page_entity(page)
    assert result["namespace_is_content"] is True


def test_build_page_entity_includes_redirect_link_by_default():
    page = {
        "page_id": 1,
        "page_title": "Example",
        "redirect_page_link": {"page_id": 2, "page_title": "Target"},
    }
    result = _build_page_entity(page)
    assert "redirect_page_link" in result
    assert result["redirect_page_link"] == {"page_id": 2, "page_title": "Target"}


def test_build_page_entity_excludes_redirect_link_when_flagged():
    page = {
        "page_id": 1,
        "page_title": "Example",
        "redirect_page_link": {"page_id": 2, "page_title": "Target"},
    }
    result = _build_page_entity(page, include_redirect_link=False)
    assert "redirect_page_link" not in result


def test_build_page_entity_strips_unknown_fields_in_redirect_link():
    page = {
        "page_id": 1,
        "page_title": "Example",
        "redirect_page_link": {
            "page_id": 2,
            "page_title": "Target",
            "new_upstream_field": "should be stripped",
        },
    }
    result = _build_page_entity(page)
    assert "new_upstream_field" not in result["redirect_page_link"]


def test_build_page_entity_preserves_negative_namespace_id_in_redirect_link():
    # Special-namespace redirects carry namespace_id == -1. The
    # prediction_classification_change/1.3.0 schema relaxed the non-negative
    # constraint, so these must pass through unchanged rather than be dropped.
    page = {
        "page_id": 1,
        "page_title": "User:Example",
        "namespace_id": 2,
        "redirect_page_link": {
            "page_id": 1,
            "page_title": "Special:RecentChanges",
            "namespace_id": -1,
        },
    }
    result = _build_page_entity(page)
    assert result["redirect_page_link"]["namespace_id"] == -1


# _build_revision_entity


def test_build_revision_entity_keeps_known_fields():
    revision = {
        "rev_id": 2,
        "rev_dt": "2021-01-01T00:00:00.0Z",
        "comment": "edit summary",
        "is_minor_edit": False,
        "is_comment_visible": True,
        "is_content_visible": True,
        "is_editor_visible": True,
        "rev_parent_id": 1,
        "rev_sha1": "abc",
        "rev_size": 1024,
    }
    assert _build_revision_entity(revision) == revision


def test_build_revision_entity_strips_unknown_fields():
    revision = {
        "rev_id": 2,
        "rev_dt": "2021-01-01T00:00:00.0Z",
        "new_upstream_field": "should be stripped",
    }
    result = _build_revision_entity(revision)
    assert "new_upstream_field" not in result
    assert result == {"rev_id": 2, "rev_dt": "2021-01-01T00:00:00.0Z"}


def test_build_revision_entity_filters_editor_via_user_builder():
    revision = {
        "rev_id": 2,
        "rev_dt": "2021-01-01T00:00:00.0Z",
        "editor": {
            "user_text": "Alice",
            "new_upstream_field": "should be stripped",
        },
    }
    result = _build_revision_entity(revision)
    assert "new_upstream_field" not in result["editor"]
    assert result["editor"] == {"user_text": "Alice"}


def test_build_revision_entity_omits_editor_when_absent():
    revision = {"rev_id": 2, "rev_dt": "2021-01-01T00:00:00.0Z"}
    result = _build_revision_entity(revision)
    assert "editor" not in result


# generate_prediction_classification_event


def test_generate_prediction_classification_event_basic():
    event = generate_prediction_classification_event(
        MINIMAL_PAGE_CHANGE, "test.stream", "my_model", "1.0.0", PREDICTION_RESULTS
    )
    assert event["$schema"] == "/mediawiki/page/prediction_classification_change/1.3.0"
    assert event["wiki_id"] == "enwiki"
    assert event["changelog_kind"] == "update"
    assert event["page_change_kind"] == "edit"
    assert event["predicted_classification"] == {
        "model_name": "my_model",
        "model_version": "1.0.0",
        "predictions": ["yes"],
        "probabilities": {"yes": 0.9, "no": 0.1},
    }


def test_generate_prediction_classification_event_strips_extra_nested_fields():
    source = {
        **MINIMAL_PAGE_CHANGE,
        "page": {
            **MINIMAL_PAGE_CHANGE["page"],
            "new_page_field": "should be stripped",
        },
        "revision": {
            **MINIMAL_PAGE_CHANGE["revision"],
            "new_revision_field": "should be stripped",
        },
        "performer": {
            "user_text": "Alice",
            "new_user_field": "should be stripped",
        },
    }
    event = generate_prediction_classification_event(
        source, "test.stream", "my_model", "1.0.0", PREDICTION_RESULTS
    )
    assert "new_page_field" not in event["page"]
    assert "new_revision_field" not in event["revision"]
    assert "new_user_field" not in event["performer"]


def test_generate_prediction_classification_event_no_performer():
    event = generate_prediction_classification_event(
        MINIMAL_PAGE_CHANGE, "test.stream", "my_model", "1.0.0", PREDICTION_RESULTS
    )
    assert "performer" not in event


def test_generate_prediction_classification_event_with_performer():
    source = {
        **MINIMAL_PAGE_CHANGE,
        "performer": {"user_text": "Alice", "is_bot": False},
    }
    event = generate_prediction_classification_event(
        source, "test.stream", "my_model", "1.0.0", PREDICTION_RESULTS
    )
    assert event["performer"] == {"user_text": "Alice", "is_bot": False}


def test_generate_prediction_classification_event_with_prior_state():
    source = {
        **MINIMAL_PAGE_CHANGE,
        "prior_state": {
            "page": {
                "page_id": 1,
                "page_title": "Old Title",
                "new_page_field": "should be stripped",
            },
            "revision": {
                "rev_id": 1,
                "rev_dt": "2020-01-01T00:00:00.0Z",
                "new_field": "x",
            },
        },
    }
    event = generate_prediction_classification_event(
        source, "test.stream", "my_model", "1.0.0", PREDICTION_RESULTS
    )
    assert "new_page_field" not in event["prior_state"]["page"]
    assert "new_field" not in event["prior_state"]["revision"]
    assert "redirect_page_link" not in event["prior_state"]["page"]


def test_generate_prediction_classification_event_with_created_redirect_page():
    source = {
        **MINIMAL_PAGE_CHANGE,
        "created_redirect_page": {
            "page_id": 3,
            "page_title": "Redirect",
            "new_upstream_field": "should be stripped",
        },
    }
    event = generate_prediction_classification_event(
        source, "test.stream", "my_model", "1.0.0", PREDICTION_RESULTS
    )
    assert "new_upstream_field" not in event["created_redirect_page"]
    assert "redirect_page_link" not in event["created_redirect_page"]


def test_generate_prediction_classification_event_unsupported_schema():
    source = {**MINIMAL_PAGE_CHANGE, "$schema": "/mediawiki/revision/create/1.0.0"}
    with pytest.raises(RuntimeError, match="Unsupported event of schema"):
        generate_prediction_classification_event(
            source, "test.stream", "my_model", "1.0.0", PREDICTION_RESULTS
        )

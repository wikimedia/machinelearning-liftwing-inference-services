import json
import logging
import ssl
import uuid
from typing import Any

import aiohttp


def _meta(source_event: dict[str, Any], eventgate_stream: str) -> dict[str, Any]:
    """Generates the metadata field for new events emitted by the inference-services
    it sets the mandatory "stream" field with eventgate_stream but also propagates
    the domain and request_id from the source event and generates a new unique ID."""
    metadata = {"stream": eventgate_stream, "id": str(uuid.uuid4())}

    if "meta" in source_event:
        source_event_metadata = source_event["meta"]
        if "request_id" in source_event_metadata:
            metadata["request_id"] = source_event_metadata["request_id"]
        if "domain" in source_event_metadata:
            metadata["domain"] = source_event_metadata["domain"]
        if "uri" in source_event_metadata:
            metadata["uri"] = source_event_metadata["uri"]

    return metadata


def _revision_score_from_revision_create(
    rev_create_event: dict[str, Any], eventgate_stream: str
) -> dict:
    """Generates a revision-score event (most of it, excluding the score bits)
    from a revision-create one's data."""
    revision_score_event = {
        "$schema": "/mediawiki/revision/score/3.0.0",
        "dt": rev_create_event["rev_timestamp"],
        "meta": _meta(rev_create_event, eventgate_stream),
        "database": rev_create_event["database"],
        "page_id": rev_create_event["page_id"],
        "page_title": rev_create_event["page_title"],
        "page_namespace": rev_create_event["page_namespace"],
        "page_is_redirect": rev_create_event["page_is_redirect"],
        "rev_id": rev_create_event["rev_id"],
        "rev_timestamp": rev_create_event["rev_timestamp"],
    }
    # The following fields are not mandatory in a mediawiki.revision-create
    # event, so we optionally add it to the final revision-score event as well.
    if "performer" in rev_create_event:
        revision_score_event["performer"] = rev_create_event["performer"]
    if "rev_parent_id" in rev_create_event:
        revision_score_event["rev_parent_id"] = rev_create_event["rev_parent_id"]

    return revision_score_event


def _revision_score_from_page_change(
    page_change_event: dict[str, Any], eventgate_stream: str
) -> dict:
    """Generates a revision-score event (most of it, excluding the score bits)
    from a page_change one's data."""

    if (
        "namespace_id" not in page_change_event["page"]
        or "is_redirect" not in page_change_event["page"]
    ):
        logging.error(
            "The page_change event provided as input does not carry either "
            "page namespace_id or is_redirect fields. Event: %s",
            page_change_event,
        )
        raise RuntimeError(
            "The page_change event provided as input does not carry either "
            "page namespace_id or is_redirect fields. "
            "They are mandatory to build a revision-score event so the input "
            "cannot be processed (and the revision-score event is not emitted)."
        )

    revision_score_event = {
        "$schema": "/mediawiki/revision/score/3.0.0",
        "dt": page_change_event["dt"],
        "meta": _meta(page_change_event, eventgate_stream),
        "database": page_change_event["wiki_id"],
        "page_id": page_change_event["page"]["page_id"],
        "page_title": page_change_event["page"]["page_title"],
        "page_namespace": page_change_event["page"]["namespace_id"],
        "page_is_redirect": page_change_event["page"]["is_redirect"],
        "rev_id": page_change_event["revision"]["rev_id"],
        "rev_timestamp": page_change_event["revision"]["rev_dt"],
        "performer": {
            "user_text": page_change_event["performer"]["user_text"],
            "user_groups": page_change_event["performer"]["groups"],
            "user_is_bot": page_change_event["performer"]["is_bot"],
        },
    }
    if "rev_parent_id" in page_change_event:
        revision_score_event["rev_parent_id"] = page_change_event["revision"][
            "rev_parent_id"
        ]

    return revision_score_event


def generate_revision_score_event(
    event: dict[str, Any],
    eventgate_stream: str,
    model_version: str,
    predictions: dict,
    model_name: str,
) -> dict:
    """Generates a revision-score event, tailored for a given model,
    from a revision-create event.
    """
    # revision-score event schema has a prediction field of array type,
    # see https://github.com/wikimedia/mediawiki-event-schemas/blob/master/jsonschema/mediawiki/revision/score/current.yaml#L36,
    # so we convert the prediction field to array type accordingly.
    p = predictions["prediction"]
    if isinstance(p, bool):
        prediction = [str(p).lower()]
    elif isinstance(p, list):
        prediction = p
    else:
        prediction = [p]

    if event["$schema"].startswith("/mediawiki/revision/create/1"):
        revision_score_event = _revision_score_from_revision_create(
            event, eventgate_stream
        )
    elif event["$schema"].startswith("/mediawiki/page/change/1"):
        revision_score_event = _revision_score_from_page_change(event, eventgate_stream)
    else:
        raise RuntimeError(
            f"Unsupported event of schema {event['$schema']}, please contact "
            "the ML team."
        )

    revision_score_event["scores"] = {
        model_name: {
            "model_name": model_name,
            "model_version": model_version,
            "prediction": prediction,
            "probability": predictions["probability"],
        }
    }

    return revision_score_event


def _build_user_entity(user: dict[str, Any]) -> dict[str, Any]:
    """Builds a user entity with only the fields defined in
    fragment/mediawiki/state/entity/user/1.2.0, preventing extra fields
    from upstream schema changes leaking into outbound events.
    Note: user/1.2.0 also allows wiki_id, which we deliberately omit."""
    fields = (
        "edit_count",
        "groups",
        "is_bot",
        "is_system",
        "is_temp",
        "registration_dt",
        "user_central_id",
        "user_id",
        "user_text",
    )
    return {f: user[f] for f in fields if f in user}


def _build_page_entity(
    page: dict[str, Any], include_redirect_link: bool = True
) -> dict[str, Any]:
    """Builds a page entity with only the fields defined in
    fragment/mediawiki/state/entity/page/2.2.0.
    Set include_redirect_link=False for prior_state.page and
    created_redirect_page, which omit that sub-object in the schema.
    Note: page/2.2.0 also allows namespace_is_content, which we
    deliberately omit."""
    result: dict[str, Any] = {
        "page_id": page["page_id"],
        "page_title": page["page_title"],
    }
    for f in ("is_redirect", "namespace_id", "revision_count"):
        if f in page:
            result[f] = page[f]
    if include_redirect_link and "redirect_page_link" in page:
        link = page["redirect_page_link"]
        link_fields = (
            "interwiki_prefix",
            "is_redirect",
            "namespace_id",
            "page_id",
            "page_title",
        )
        result["redirect_page_link"] = {f: link[f] for f in link_fields if f in link}
    return result


def _build_revision_entity(revision: dict[str, Any]) -> dict[str, Any]:
    """Builds a revision entity with only the fields defined in
    fragment/mediawiki/state/entity/revision/2.0.0.
    Note: revision/2.0.0 also allows a revert object, which we
    deliberately omit."""
    result: dict[str, Any] = {
        "rev_id": revision["rev_id"],
        "rev_dt": revision["rev_dt"],
    }
    for f in (
        "comment",
        "is_comment_visible",
        "is_content_visible",
        "is_editor_visible",
        "is_minor_edit",
        "rev_parent_id",
        "rev_sha1",
        "rev_size",
    ):
        if f in revision:
            result[f] = revision[f]
    if "editor" in revision:
        result["editor"] = _build_user_entity(revision["editor"])
    return result


def generate_prediction_classification_event(
    source_event: dict[str, Any],
    eventgate_stream: str,
    model_name: str,
    model_version: str,
    prediction_results: dict,
) -> dict:
    """Generates a prediction_classification event, tailored for a given model,
    from a page_change event.
    """
    if not source_event["$schema"].startswith("/mediawiki/page/change/1"):
        raise RuntimeError(
            f"Unsupported event of schema {source_event['$schema']}, please contact "
            "the ML team."
        )

    event: dict[str, Any] = {
        "$schema": "/mediawiki/page/prediction_classification_change/1.3.0",
        "changelog_kind": source_event["changelog_kind"],
        "page_change_kind": source_event["page_change_kind"],
        "dt": source_event["dt"],
        "meta": _meta(source_event, eventgate_stream),
        "page": _build_page_entity(source_event["page"]),
        "revision": _build_revision_entity(source_event["revision"]),
        "wiki_id": source_event["wiki_id"],
        "predicted_classification": {
            "model_name": model_name,
            "model_version": model_version,
            "predictions": prediction_results["predictions"],
            "probabilities": prediction_results["probabilities"],
        },
    }
    if "comment" in source_event:
        event["comment"] = source_event["comment"]
    if "performer" in source_event:
        event["performer"] = _build_user_entity(source_event["performer"])
    if "prior_state" in source_event:
        prior = source_event["prior_state"]
        prior_state: dict[str, Any] = {}
        if "page" in prior:
            prior_state["page"] = _build_page_entity(
                prior["page"], include_redirect_link=False
            )
        if "revision" in prior:
            prior_state["revision"] = _build_revision_entity(prior["revision"])
        event["prior_state"] = prior_state
    if "created_redirect_page" in source_event:
        event["created_redirect_page"] = _build_page_entity(
            source_event["created_redirect_page"], include_redirect_link=False
        )
    return event


def generate_page_weighted_tags_event(
    source_event: dict[str, Any], eventgate_stream: str, weighted_tags: dict[str, Any]
) -> dict:
    """
    Generates a page_weighted_tags_change event, tailored for a given model,
    from a page_change event.

    Args:
        source_event: The source page_change event
        eventgate_stream: The EventGate stream name
        weighted_tags: The weighted_tags structure, can contain "set", "clear", or both
    """
    if not source_event["$schema"].startswith("/mediawiki/page/change/1"):
        raise RuntimeError(
            f"Unsupported event of schema {source_event['$schema']}, please contact "
            "the ML team."
        )

    event = {
        "$schema": "/mediawiki/cirrussearch/page_weighted_tags_change/1.0.0",
        "dt": source_event["dt"],
        "meta": _meta(source_event, eventgate_stream),
        "page": {
            "namespace_id": source_event["page"]["namespace_id"],
            "page_id": source_event["page"]["page_id"],
            "page_title": source_event["page"]["page_title"],
        },
        "weighted_tags": weighted_tags,
        "wiki_id": source_event["wiki_id"],
        "rev_based": True,
    }
    return event


async def send_event(
    event: dict[str, Any],
    eventgate_url: str,
    tls_bundle_path: str,
    user_agent: str,
    aio_http_client,
) -> None:
    """Sends a revision-score-event to EventGate."""
    try:
        sslcontext = ssl.create_default_context(cafile=tls_bundle_path)
        async with aio_http_client.post(
            eventgate_url,
            ssl=sslcontext,
            json=event,
            headers={
                "Content-type": "application/json",
                "UserAgent": user_agent,
            },
        ) as resp:
            log_msg = (
                "Sent the following event to "
                "EventGate, that returned a HTTP response with status "
                f"{resp.status} and text '{await resp.text()}'"
                f": {json.dumps(event)}"
            )
            if resp.status >= 400:
                logging.error(log_msg)
            else:
                logging.debug(log_msg)
    except aiohttp.ClientResponseError as e:
        logging.error(
            "The event has been rejected by EventGate, "
            "that returned a HTTP {} with the following error: {}".format(
                e.status, e.message
            )
        )
        raise RuntimeError(
            "The event posted to EventGate has been rejected, "
            "please contact the ML team if the issue persists."
        )
    except aiohttp.ClientError as e:
        logging.error(f"Connection error while sending an event to EventGate: {e}")
        raise RuntimeError(
            "Connection error while trying to post the event to EventGate. "
            "Please contact the ML team if the issue persists."
        )
    except Exception as e:
        logging.error(
            f"Unexpected error while trying to send an event to EventGate: {e}"
        )
        raise RuntimeError(
            "Unexpected error happened while the event was posted to EventGate, "
            "there is the possibility that it never reached it. "
            "Please contact the ML team if the issue persists."
        )

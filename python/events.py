import logging
import ssl
import uuid
from typing import Any, Dict

import aiohttp


def _meta(source_event: Dict[str, Any], eventgate_stream: str) -> Dict[str, Any]:
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
    rev_create_event: Dict[str, Any], eventgate_stream: str
) -> Dict:
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
    page_change_event: Dict[str, Any], eventgate_stream: str
) -> Dict:
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
    event: Dict[str, Any],
    eventgate_stream: str,
    model_version: str,
    predictions: Dict,
    model_name: str,
) -> Dict:
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


def generate_prediction_classification_event(
    source_event: Dict[str, Any],
    eventgate_stream: str,
    model_name: str,
    model_version: str,
    prediction_results: Dict,
) -> Dict:
    """Generates a prediction_classification event, tailored for a given model,
    from a page_change event.
    """
    if source_event["$schema"].startswith("/mediawiki/page/change/1"):
        event = {k: v for k, v in source_event.items()}
        # remove content_slots field in .prior_state.revision and .revision
        if "revision" in event and "content_slots" in event["revision"]:
            del event["revision"]["content_slots"]
        if (
            "prior_state" in event
            and "revision" in event["prior_state"]
            and "content_slots" in event["prior_state"]["revision"]
        ):
            del event["prior_state"]["revision"]["content_slots"]
        event["$schema"] = "mediawiki/page/prediction_classification_change/1.1.0"
        event["meta"] = _meta(source_event, eventgate_stream)
        event["predicted_classification"] = {
            "model_name": model_name,
            "model_version": model_version,
            "predictions": prediction_results["predictions"],
            "probabilities": prediction_results["probabilities"],
        }
    else:
        raise RuntimeError(
            f"Unsupported event of schema {event['$schema']}, please contact "
            "the ML team."
        )
    return event


async def send_event(
    event: Dict[str, Any],
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
                f":\n{event}"
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

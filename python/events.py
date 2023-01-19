import json
import logging
import tornado

from typing import Dict, Any
from http import HTTPStatus


def generate_revision_score_event(
    rev_create_event: Dict[str, Any],
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
    revision_score_event = {
        "$schema": "/mediawiki/revision/score/2.0.0",
        "meta": {
            "stream": eventgate_stream,
        },
        "database": rev_create_event["database"],
        "page_id": rev_create_event["page_id"],
        "page_title": rev_create_event["page_title"],
        "page_namespace": rev_create_event["page_namespace"],
        "page_is_redirect": rev_create_event["page_is_redirect"],
        "rev_id": rev_create_event["rev_id"],
        "rev_timestamp": rev_create_event["rev_timestamp"],
        "scores": {
            model_name: {
                "model_name": model_name,
                "model_version": model_version,
                "prediction": prediction,
                "probability": predictions["probability"],
            }
        },
    }

    # The following fields are not mandatory in a mediawiki.revision-create
    # event, so we optionally add it to the final revision-score event as well.
    if "performer" in rev_create_event:
        revision_score_event["performer"] = rev_create_event["performer"]
    if "rev_parent_id" in rev_create_event:
        revision_score_event["rev_parent_id"] = rev_create_event["rev_parent_id"]

    return revision_score_event


async def send_event(
    revision_score_event: Dict[str, Any],
    eventgate_url: str,
    tls_bundle_path: str,
    user_agent: str,
    aio_http_client,
) -> None:
    """Sends a revision-score-event to EventGate."""
    try:
        await aio_http_client.fetch(
            eventgate_url,
            method="POST",
            ca_certs=tls_bundle_path,
            body=json.dumps(revision_score_event),
            headers={"Content-type": "application/json"},
            user_agent=user_agent,
        )
        logging.debug(
            "Successfully sent the following event to "
            "EventGate: {}".format(revision_score_event)
        )
    except tornado.httpclient.HTTPError as e:
        logging.error(
            "The revision score event has been rejected by EventGate, "
            "that returned a non-200 HTTP return code "
            "with the following error: {}".format(e)
        )
        raise tornado.web.HTTPError(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            reason=(
                "Eventgate rejected the revision-score event "
                "(a non-HTTP-200 response was returned). "
                "Please contact the ML team for more info."
            ),
        )
    except Exception as e:
        logging.error(
            "Unexpected error while trying to send a revision score "
            "event to EventGate: {}".format(e)
        )
        raise tornado.web.HTTPError(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            reason=(
                "An unexpected error has occurred while trying "
                "to send the revision-score event to Eventgate. "
                "Please contact the ML team for more info."
            ),
        )

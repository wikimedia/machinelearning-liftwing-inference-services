import aiohttp
import logging
import ssl

from typing import Dict, Any


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
        sslcontext = ssl.create_default_context(cafile=tls_bundle_path)
        async with aio_http_client.post(
            eventgate_url,
            ssl=sslcontext,
            json=revision_score_event,
            headers={
                "Content-type": "application/json",
                "UserAgent": user_agent,
            },
        ) as resp:
            logging.debug(
                "Sent the following event to "
                "EventGate, that returned a HTTP response with status "
                f"{resp.status} and text '{await resp.text()}'"
                f":\n{revision_score_event}"
            )
    except aiohttp.ClientError as e:
        logging.error(f"Connection error while sending an event to EventGate: {e}")
        # FIXME: after all model-servers are migrated to KServe 0.10,
        # this RuntimeError should probably become kserve.errors.InferenceError
        # We don't want to leak internal error info from aiohttp to the external
        # client, this is why the logging msg is richer in content.
        raise RuntimeError(
            "Connection error while trying to post the event to EventGate. "
            "Please contact the ML team if the issue persists."
        )
    except aiohttp.ClientResponseError as e:
        logging.error(
            "The event has been rejected by EventGate, "
            "that returned a HTTP {} with the following error: {}".format(
                e.status, e.message
            )
        )
        # FIXME: after all model-servers are migrated to KServe 0.10,
        # this RuntimeError should probably become kserve.errors.InferenceError
        # We don't want to leak internal error info from aiohttp to the external
        # client, this is why the logging msg is richer in content.
        raise RuntimeError(
            "The event posted to EventGate has been rejected, "
            "please contact the ML team if the issue persists."
        )
    except Exception as e:
        logging.error(
            f"Unexpected error while trying to send an event to EventGate: {e}"
        )
        # FIXME: after all model-servers are migrated to KServe 0.10,
        # this RuntimeError should probably become kserve.errors.InferenceError
        # We don't want to leak internal error info from aiohttp to the external
        # client, this is why the logging msg is richer in content.
        raise RuntimeError(
            "Unexpected error happened while the event was posted to EventGate, "
            "there is the possibility that it never reached it. "
            "Please contact the ML team if the issue persists."
        )

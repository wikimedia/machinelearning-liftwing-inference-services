import re

import tornado
from typing import Dict, Optional
from http import HTTPStatus


def get_revision_event(inputs: Dict, event_input_key) -> Optional[str]:
    try:
        return inputs[event_input_key]
    except KeyError:
        return None


def get_rev_id(inputs: Dict, event_input_key) -> Dict:
    """Get a revision id from the inputs provided.
    The revision id can be contained into an event dict
    or passed directly as value.
    """
    try:
        # If a revision create event is passed as input,
        # its rev-id is considerate the one to score.
        # Otherwise, we look for a specific "rev_id" input.
        if event_input_key in inputs:
            rev_id = inputs[event_input_key]["rev_id"]
        else:
            rev_id = inputs["rev_id"]
    except KeyError:
        raise tornado.web.HTTPError(
            status_code=HTTPStatus.BAD_REQUEST,
            reason='Expected "rev_id" in input data.',
        )
    if not isinstance(rev_id, int):
        raise tornado.web.HTTPError(
            status_code=HTTPStatus.BAD_REQUEST,
            reason='Expected "rev_id" to be an integer.',
        )
    return rev_id


def get_lang(inputs: Dict, event_input_key) -> Dict:
    try:
        if event_input_key in inputs:
            lang = re.match(r"(\w+)wiki", inputs[event_input_key]["database"]).group(1)
        else:
            lang = inputs["lang"]
    except KeyError:
        raise tornado.web.HTTPError(
            status_code=HTTPStatus.BAD_REQUEST,
            reason='Missing "lang" in input data.',
        )
    return lang


def get_page_title(inputs: Dict, event_input_key) -> Dict:
    try:
        if event_input_key in inputs:
            page_title = inputs[event_input_key]["page_title"]
        else:
            page_title = inputs["page_title"]
    except KeyError:
        raise tornado.web.HTTPError(
            status_code=HTTPStatus.BAD_REQUEST,
            reason='Missing "page_title" in input data.',
        )
    return page_title

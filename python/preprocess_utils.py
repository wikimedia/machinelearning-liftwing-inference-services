import re
import logging

from typing import Dict

from kserve.errors import InvalidInput


def get_lang(inputs: Dict, event_input_key) -> Dict:
    try:
        if event_input_key in inputs:
            lang = re.match(r"(\w+)wiki", inputs[event_input_key]["database"]).group(1)
        else:
            lang = inputs["lang"]
    except KeyError:
        logging.error("Missing lang in input data.")
        raise InvalidInput('Missing "lang" in input data.')
    return lang


def get_page_title(inputs: Dict, event_input_key) -> Dict:
    try:
        if event_input_key in inputs:
            page_title = inputs[event_input_key]["page_title"]
        else:
            page_title = inputs["page_title"]
    except KeyError:
        logging.error("Missing page_title in input data.")
        raise InvalidInput('Missing "page_title" in input data.')
    return page_title

import json
import logging.config
import os
import time
from collections import defaultdict
from copy import copy
from functools import wraps
from typing import Any, List

import yaml
from fastapi import HTTPException, status
from starlette.responses import Response

logger = logging.getLogger(__name__)

with open(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "config/available_models.yaml"
    )
) as f:
    available_models = yaml.safe_load(f)


def get_check_models(context: str, models=None):
    try:
        models_in_context = copy(available_models[context])
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": {
                    "code": "not found",
                    "message": f"No scorers available for {context}",
                }
            },
        )
    if not models:
        models_list = models_in_context["models"]
    else:
        models_list = models.split("|")
        filtered_models = {
            key: value
            for key, value in models_in_context["models"].items()
            if key in models_list
        }
        models_in_context["models"] = filtered_models

    for model in models_list:
        if model not in models_in_context["models"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": {
                        "code": "not found",
                        "message": f"Models ('{model}',) not available for {context}",
                    }
                },
            )
    return models_list, models_in_context


def merge_liftwing_responses(context: str, responses: List[str]) -> defaultdict:
    result = defaultdict(lambda: defaultdict(lambda: defaultdict()))
    for d in responses:
        if not d:
            continue
        for k, v in d[context].items():
            if isinstance(v, dict) and k == "scores":
                for rev_id, scores in v.items():
                    if rev_id in result[context][k]:
                        result[context][k][rev_id].update(scores)
                    else:
                        result[context][k][rev_id] = scores
            else:
                result[context][k].update(v)
    return result


def manipulate_wp10_call(func: callable):
    """
    This function is meant to be used as a decorator to manipulate the input and output of a response
    in order to make it compatible with the old ORES API since wp10 has been renamed to articlequality
    """

    async def wrapper_func(*args, **kwargs):
        requested_model = kwargs.get("model_name")
        kwargs["model_name"] = (
            requested_model if requested_model != "wp10" else "articlequality"
        )
        context = kwargs["db"]
        rev_id = str(kwargs["rev_id"])
        response = await func(*args, **kwargs)
        if isinstance(response, dict) and (requested_model == "wp10"):
            response[context]["models"]["wp10"] = response[context]["models"].pop(
                "articlequality"
            )
            response[context]["scores"][rev_id]["wp10"] = response[context]["scores"][
                rev_id
            ].pop("articlequality")
        return response

    return wrapper_func


class PrettyJSONResponse(Response):
    media_type = "application/json"

    def render(self, content: Any) -> bytes:
        return json.dumps(
            content,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            separators=(", ", ": "),
        ).encode("utf-8")


async def create_error_response(
    error_message: str,
    error_type: str,
    context: str,
    model_name: str,
    rev_id: int,
    models_info: dict = available_models,
):
    return {
        context: {
            "models": {
                model_name: {
                    "version": models_info[context]["models"][model_name]["version"]
                },
            },
            "scores": {
                str(rev_id): {
                    model_name: {
                        "error": {
                            "message": error_message,
                            "type": error_type,
                        },
                    }
                }
            },
        }
    }


def log_user_request(func: callable):
    @wraps(func)
    async def wrapper_func(*args, **kwargs):
        start_time = time.time()
        request = kwargs.get("request")
        logger.info(
            f"IP:{request.client.host}, User-Agent:{request.headers.get('User-Agent')}"
        )
        logger.debug(f"method_called:{func.__name__} url_path:{request.url.path}")
        response = await func(*args, **kwargs)
        logger.info(f"response_time:{time.time() - start_time}s")
        return response

    return wrapper_func

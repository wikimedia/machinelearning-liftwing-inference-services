import json
import logging.config
import os
from distutils.util import strtobool
from typing import Union

import yaml
from app.liftwing.response import make_liftiwing_calls
from app.response_models import ResponseModel
from app.utils import (
    PrettyJSONResponse,
    check_unsupported_features,
    get_check_models,
    log_user_request,
    merge_liftwing_responses,
)
from fastapi import FastAPI, HTTPException, Query, Request, status
from starlette.responses import RedirectResponse

logger = logging.getLogger(__name__)


description = """
> This is a simple API to translate ORES API calls to our new ML serving infrastructure,  [Lift Wing](https://wikitech.wikimedia.org/wiki/Machine_Learning/LiftWing).
> You are encouraged to migrate to Lift Wing since the Wikimedia ML team is decommissioning the ORES infrastructure. This service will be available to support users transitioning to the new infrastructure until December 2023. Please note that Lift Wing differs from ORES, check the [docs](https://wikitech.wikimedia.org/wiki/Machine_Learning/LiftWing/Usage#Differences_using_Lift_Wing_instead_of_ORES) for more info.
> If you want to start experimenting with Lift Wing, please check this [doc](https://wikitech.wikimedia.org/wiki/Machine_Learning/LiftWing/Usage) page containing all the info that you need.
> Last but not the least, please reach out to the Wikimedia ML team if you have any question or doubt (see link below to get to our [Phabricator](https://phabricator.wikimedia.org/) board). We are also available in the `#wikimedia-ml` channel on Libera IRC.
"""

app = FastAPI(
    debug=True,
    title="ORES legacy service 🤖",
    description=description,
    version="1.0.0",
    contact={
        "name": "Machine Learning team",
        "url": "https://phabricator.wikimedia.org/tag/machine-learning-team/",
    },
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    },
)

with open(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "config/available_models.yaml"
    )
) as f:
    available_models = yaml.safe_load(f)

with open(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "config/ores_example_responses.json"
    )
) as f:
    examples = json.load(f)["responses"]

liftwing_url = os.environ.get("LIFTWING_URL")


@app.get("/", include_in_schema=False)
@app.get("", include_in_schema=False)
@log_user_request
async def root(request: Request):
    return RedirectResponse(url="/docs")


@app.get(
    "/v3/scores",
    response_class=PrettyJSONResponse,
    response_model=ResponseModel,
    responses={
        200: {
            "description": "Item requested by ID",
            "content": {
                "application/json": {
                    "example": examples["list_available_scores"],
                }
            },
        },
    },
)
@app.get(
    "/v3/scores/",
    response_class=PrettyJSONResponse,
    response_model=ResponseModel,
    responses={
        200: {
            "description": "Item requested by ID",
            "content": {
                "application/json": {
                    "example": examples["list_available_scores"],
                }
            },
        },
    },
    include_in_schema=False,
)
@log_user_request
async def list_available_scores(
    request: Request, model_info: str = Query(None, include_in_schema=False)
):
    """
    **Implementation Notes**

    This route provides a list of available contexts for scoring. Generally a wiki is 1:1 with a
    context and a context is expressed as the database name of the wiki. For example "enwiki" is
    English Wikipedia and "wikidatawiki" is Wikidata
    """
    check_unsupported_features(model_info=model_info)
    return available_models


@app.get(
    "/v3/scores/{context}",
    response_class=PrettyJSONResponse,
    response_model=ResponseModel,
    responses={
        200: {
            "description": "Item requested by ID",
            "content": {
                "application/json": {
                    "example": examples["get_scores"],
                }
            },
        },
    },
)
@app.get(
    "/v3/scores/{context}/",
    response_class=PrettyJSONResponse,
    response_model=ResponseModel,
    responses={
        200: {
            "description": "Item requested by ID",
            "content": {
                "application/json": {
                    "example": examples["get_scores"],
                }
            },
        },
    },
    include_in_schema=False,
)
@log_user_request
async def get_scores(
    context: str,
    request: Request,
    models: str = None,
    revids: Union[str, None] = None,
    model_info: str = Query(None, include_in_schema=False),
):
    """
    **Implementation Notes**

    This route provides access to all {models} within a {context}. This path is useful for either
    exploring information about {models} available within a {context} or scoring one or more {
    revids} using one or more {models} at the same time.
    **Warning**
    Scoring more than 20 scores in a single request is not supported anymore by this endpoint.
    In case you need to score more please break down your request into smaller chunks or consider using the /v3/scores/{context}/{revid} endpoint to score one revision at a time.
    For more information please visit https://wikitech.wikimedia.org/wiki/ORES
    """
    check_unsupported_features(model_info=model_info)
    models_list, models_in_context = get_check_models(context, models)
    revids_list = list(map(int, revids.split("|") if revids else []))
    lw_request_limit = int(os.getenv("LW_REQUEST_LIMIT", 20))
    number_of_requests = len(revids_list) * len(models_list)
    if number_of_requests > lw_request_limit:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": {
                    "code": "too many requests",
                    "message": f"Scoring more than {lw_request_limit} scores is not supported anymore by this endpoint. "
                    f"You are requesting {number_of_requests} ORES scores which is above the supported limit of {lw_request_limit} for a single request. "
                    f"Please break down your request into smaller chunks or consider using the /v3/scores/{{context}}/{{revid}} endpoint to score one revision at a time. "
                    f"For more information please visit https://wikitech.wikimedia.org/wiki/ORES",
                }
            },
        )
    responses = await make_liftiwing_calls(
        context, models_list, revids_list, False, liftwing_url
    )
    responses = merge_liftwing_responses(context, responses)
    if responses:
        return responses
    else:
        return {context: models_in_context}


@app.get(
    "/v3/scores/{context}/{revid}",
    response_class=PrettyJSONResponse,
    response_model=ResponseModel,
    responses={
        200: {
            "description": "Item requested by ID",
            "content": {
                "application/json": {
                    "example": examples["get_context_scores"],
                }
            },
        },
    },
)
@app.get(
    "/v3/scores/{context}/{revid}/",
    response_class=PrettyJSONResponse,
    response_model=ResponseModel,
    responses={
        200: {
            "description": "Item requested by ID",
            "content": {
                "application/json": {
                    "example": examples["get_context_scores"],
                }
            },
        },
    },
    include_in_schema=False,
)
@log_user_request
async def get_context_scores(
    context: str,
    revid: int,
    request: Request,
    models: str = None,
    features: str = None,
    model_info: str = Query(None, include_in_schema=False),
):
    if features == "":
        # adding this to support previous functionality of ores that was using ?features without
        # specifying a boolean value
        features = True
    elif features is None:
        features = False
    else:
        features = strtobool(features)
    check_unsupported_features(model_info=model_info)
    models_list, _ = get_check_models(context, models)
    responses = await make_liftiwing_calls(
        context, models_list, [revid], features, liftwing_url
    )
    response = merge_liftwing_responses(context, responses)
    return response


@app.get(
    "/v3/scores/{context}/{revid}/{model}",
    response_class=PrettyJSONResponse,
    response_model=ResponseModel,
    responses={
        200: {
            "description": "Item requested by ID",
            "content": {
                "application/json": {
                    "example": examples["get_model_scores"],
                }
            },
        },
    },
)
@app.get(
    "/v3/scores/{context}/{revid}/{model}/",
    response_class=PrettyJSONResponse,
    response_model=ResponseModel,
    responses={
        200: {
            "description": "Item requested by ID",
            "content": {
                "application/json": {
                    "example": examples["get_model_scores"],
                }
            },
        },
    },
    include_in_schema=False,
)
@log_user_request
async def get_model_scores(
    context: str,
    revid: int,
    model: str,
    request: Request,
    features: str = None,
    model_info: str = Query(None, include_in_schema=False),
    inject: str = Query(None, include_in_schema=False),
):
    if features == "":
        # adding this to support previous functionality of ores that was using ?features without
        # specifying a boolean value
        features = True
    elif features is None:
        features = False
    else:
        features = strtobool(features)
    check_unsupported_features(model_info=model_info, inject=inject)
    responses = await make_liftiwing_calls(
        context, [model], [revid], features, liftwing_url
    )
    response = merge_liftwing_responses(context, responses)
    return response


@app.get("/scores/{context}", include_in_schema=False)
@app.get("/v1/scores/{context}", include_in_schema=False)
@app.get("/scores/{context}/{revid}", include_in_schema=False)
@app.get("/v1/scores/{context}/{revid}", include_in_schema=False)
@app.get("/scores/{context}/{revid}/{model}", include_in_schema=False)
@app.get("/v1/scores/{context}/{revid}/{model}", include_in_schema=False)
@log_user_request
async def get_scores_v1(request: Request):
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail={
            "error": {
                "code": "not found",
                "message": "ORES support for v1 has been deprecated. Please use v3.",
            }
        },
    )

import logging.config
import os
from typing import Union

import yaml
from fastapi import FastAPI, Request

from app.liftwing.response import make_liftiwing_calls
from app.utils import (
    PrettyJSONResponse,
    get_check_models,
    log_user_request,
    merge_liftwing_responses,
)

logging.config.fileConfig("logging.conf", disable_existing_loggers=False)
logger = logging.getLogger(__name__)

app = FastAPI(
    debug=True,
    title="ORES/LiftWing calls legacy service",
    description="""
    This is a simple API to translate ORES API calls to LiftWing calls.
    It is meant to be used as a temporary solution until ORES is migrated
    to the new scoring service.""",
    version="1.0.0",
    contact={
        "name": "Machine Learning team",
        "url": "https://www.mediawiki.org/wiki/Machine_Learning",
    },
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    },
)

with open("app/config/available_models.yaml") as f:
    available_models = yaml.safe_load(f)

liftwing_url = os.environ.get("LIFTWING_URL")


@app.get("/", response_class=PrettyJSONResponse)
@log_user_request
async def root(request: Request):
    return {"message": "ORES/LiftWing calls legacy service"}


@app.get("/v3/scores", response_class=PrettyJSONResponse)
@log_user_request
async def list_available_scores(request: Request):
    """
    **Implementation Notes**

    This route provides a list of available contexts for scoring. Generally a wiki is 1:1 with a
    context and a context is expressed as the database name of the wiki. For example "enwiki" is
    English Wikipedia and "wikidatawiki" is Wikidata
    """
    return available_models


@app.get("/v3/scores/{context}", response_class=PrettyJSONResponse)
@log_user_request
async def get_scores(
    context: str, request: Request, models: str = None, revids: Union[str, None] = None
):
    """
    **Implementation Notes**

    This route provides access to all {models} within a {context}. This path is useful for either
    exploring information about {models} available within a {context} or scoring one or more {
    revids} using one or more {models} at the same time.
    """
    models_list, models_in_context = get_check_models(context, models)
    revids_list = list(map(int, revids.split("|") if revids else []))
    responses = await make_liftiwing_calls(
        context, models_list, revids_list, liftwing_url
    )
    responses = merge_liftwing_responses(context, responses)
    if responses:
        return responses
    else:
        return {context: models_in_context}


@app.get("/v3/scores/{context}/{revid}", response_class=PrettyJSONResponse)
@log_user_request
async def get_context_scores(
    context: str, revid: int, request: Request, models: str = None
):
    models_list, _ = get_check_models(context, models)
    responses = await make_liftiwing_calls(context, models_list, [revid], liftwing_url)
    responses = merge_liftwing_responses(context, responses)
    return responses


@app.get("/v3/scores/{context}/{revid}/{model}", response_class=PrettyJSONResponse)
@log_user_request
async def get_model_scores(context: str, revid: int, model: str, request: Request):
    response = await make_liftiwing_calls(context, [model], [revid], liftwing_url)
    responses = merge_liftwing_responses(context, response)
    return responses

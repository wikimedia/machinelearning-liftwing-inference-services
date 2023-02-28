from fastapi import FastAPI
import yaml
import os
import time
import logging
from typing import Union
from app.liftwing.response import make_liftiwing_calls
from app.utils import get_check_models, merge_liftwing_responses

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

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


@app.get("/")
async def root():
    return {"message": "ORES/LiftWing calls legacy service"}


@app.get("/v3/scores")
async def list_available_scores():
    """
    **Implementation Notes**

    This route provides a list of available contexts for scoring. Generally a wiki is 1:1 with a
    context and a context is expressed as the database name of the wiki. For example "enwiki" is
    English Wikipedia and "wikidatawiki" is Wikidata
    """
    start_time = time.time()
    logger.info(f"Response returned in  {time.time() - start_time} sec")
    return available_models


@app.get("/v3/scores/{context}")
async def get_scores(context: str, models: str = None, revids: Union[str, None] = None):
    """
    **Implementation Notes**

    This route provides access to all {models} within a {context}. This path is useful for either
    exploring information about {models} available within a {context} or scoring one or more {
    revids} using one or more {models} at the same time.
    """
    start_time = time.time()
    models_list, models_in_context = get_check_models(context, models)
    revids_list = list(map(int, revids.split("|") if revids else []))
    responses = await make_liftiwing_calls(
        context, models_list, revids_list, liftwing_url
    )
    logger.info(f"Made #{len(responses)} calls to LiftWing")
    responses = merge_liftwing_responses(context, responses)
    logger.info(f"Response returned in  {time.time() - start_time} sec")
    if responses:
        return responses
    else:
        return {context: models_in_context}


@app.get("/v3/scores/{context}/{revid}")
async def get_context_scores(context: str, revid: int, models: str = None):
    print("called /v3/scores/{context}/{revid}")
    start_time = time.time()
    models_list, _ = get_check_models(context, models)
    responses = await make_liftiwing_calls(context, models_list, [revid], liftwing_url)
    logger.info(f"Made #{len(responses)} calls to LiftWing")
    responses = merge_liftwing_responses(context, responses)
    logger.info(f"Response returned in  {time.time() - start_time} sec")
    return responses


@app.get("/v3/scores/{context}/{revid}/{model}")
async def get_model_scores(context: str, revid: int, model: str):
    start_time = time.time()
    response = await make_liftiwing_calls(context, [model], [revid], liftwing_url)
    responses = merge_liftwing_responses(context, response)
    logger.info(f"Response returned in  {time.time() - start_time} sec")
    return responses

import logging
from typing import List
import aiohttp
import asyncio
from app.utils import manipulate_wp10_call
from app.utils import create_error_response


@manipulate_wp10_call
async def get_liftwing_response(
    session: aiohttp.ClientSession,
    db: str,
    model_name: str,
    rev_id: int,
    liftwing_url: str,
) -> dict:
    url = f"{liftwing_url}:30443/v1/models/{db}-{model_name}:predict"
    model_hostname = f"revscoring{'-editquality-' if model_name in ['damaging', 'reverted', 'goodfaith'] else '-'}{model_name}"
    headers = {
        "Content-type": "application/json",
        "Host": f"{db}-{model_name}.{model_hostname}.wikimedia.org",
    }
    data = {"rev_id": rev_id}
    try:
        async with session.post(url, headers=headers, json=data) as response:
            if response.status != 200:
                response_json = await response.json()
                error_message = response_json["error"]
                logging.error(
                    f"LiftWing call for model {model_name} returned {response.status} with message {error_message}"
                )
                logging.error(f"Raw Response: {response_json}")
                return await create_error_response(
                    error_message, response.reason, db, model_name, rev_id
                )
            return await response.json()
    except aiohttp.ClientError as e:
        return await create_error_response(e, "ClientError", db, model_name, rev_id)


async def make_liftiwing_calls(
    context: str,
    models: List[str],
    rev_ids: List[int],
    liftwing_url: str = "https://inference.discovery.wmnet",
):
    async with aiohttp.ClientSession() as session:
        tasks = [
            get_liftwing_response(
                session=session,
                db=context,
                model_name=model,
                rev_id=revid,
                liftwing_url=liftwing_url,
            )
            for revid in rev_ids
            for model in models
        ]
        result = await asyncio.gather(*tasks)
    return result

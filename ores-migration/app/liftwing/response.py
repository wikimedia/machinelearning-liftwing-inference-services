import logging
from typing import List
import aiohttp
import asyncio


async def get_liftwing_response(
    session,
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
    async with session.post(url, headers=headers, json=data) as response:
        if response.status != 200:
            logging.error(
                f"LiftWing call for model {model_name} returned {response.status}"
            )
            return {}
        return await response.json()


async def make_liftiwing_calls(
    context: str,
    models: List[str],
    rev_ids: List[int],
    liftwing_url: str = "https://inference.discovery.wmnet",
):
    async with aiohttp.ClientSession() as session:
        tasks = [
            get_liftwing_response(session, context, model, revid, liftwing_url)
            for revid in rev_ids
            for model in models
        ]
        result = await asyncio.gather(*tasks)
    return result

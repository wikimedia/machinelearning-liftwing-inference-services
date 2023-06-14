import aiohttp
import asyncio
import logging
import mwapi


from kserve.errors import InvalidInput, InferenceError
from typing import Dict, Optional

from mwapi.errors import (
    APIError,
    ConnectionError,
    RequestError,
    TimeoutError,
    TooManyRedirectsError,
)
from revscoring.extractors.api import MWAPICache, Extractor
from revscoring.errors import MissingResource, UnexpectedContentType
from decorators import elapsed_time_async, elapsed_time


@elapsed_time_async
async def get_revscoring_extractor_cache(
    rev_id: int,
    user_agent: str,
    client_session: aiohttp.ClientSession,
    wiki_url: str,
    wiki_host: str = None,
    fetch_extra_info: bool = False,
    mwapi_session: mwapi.AsyncSession = None,
) -> MWAPICache:
    """Build a revscoring extractor HTTP cache using async HTTP calls.
    The revscoring API extractor can automatically fetch data from
    the MW API as well, but sadly only with blocking IO (namely, using Session
    from the mwapi package). Since KServe works asyncio,
    we prefer to use mwapi's AsyncSession and pass the data (as MWAPICache)
    to revscoring.
    From tests in T309623, the API extractor fetches:
    - info related to the rev-id
    - user and parent-rev-id data as well
    The total is 3 MW API calls, but it varies between model implementations.
    Sometimes only the rev-id info are needed, meanwhile other times parent-rev-id
    and user info are needed as well. This function offers a parameter to tune
    the number of async HTTP calls to the MW API and the correspondent MWAPICache
    entries.

        Parameters:
            rev_id: The MediaWiki revision id to check.
            user_agent: HTTP User Agent to use in HTTP calls to the MW API.
            client_session: the aiohttp's ClientSession to use when calling
                            the MWAPI.
            wiki_url: The URL of the MW API to use.
            wiki_host: The HTTP Host header to set in calls to the MW API.
            fetch_extra_info: if True, a total of 3 async HTTP calls to the MW
                              API will be made. By default (False) only one is
                              made.
            mwapi_session: A custom mwapi.AsyncSession to use in the code.
                           If not specified one will be created instead.

        Returns:
            The revscoring api extractor's MWAPICache fetched via async HTTP calls.
    """
    if wiki_host:
        client_session.headers.update({"Host": wiki_host})

    if mwapi_session:
        session = mwapi_session
    else:
        session = mwapi.AsyncSession(
            wiki_url, user_agent=user_agent, session=client_session
        )

    # The parameters are always the same across revscoring models, so
    # we kept it static. If there is the need to tune those in the future
    # it should be easy to move them to a funtion's parameter.
    params = {
        "rvprop": {
            "content",
            "userid",
            "size",
            "contentmodel",
            "ids",
            "user",
            "comment",
            "timestamp",
        }
    }
    try:
        # This API call is needed by all model implementations so it is
        # done by default.
        rev_id_doc = await session.get(
            action="query", prop="revisions", revids=[rev_id], rvslots="main", **params
        )

        # If 'badrevids' is returned by the MW API then there is something wrong
        # with the revision id provided. If the error message is changed in the InvalidInput exception
        # then the check in the ores-legacy app should be changed as well.
        if "query" in rev_id_doc and "badrevids" in rev_id_doc["query"]:
            logging.error(
                "Received a badrevids error from the MW API for rev-id {}. "
                "Complete response: {}".format(rev_id, rev_id_doc)
            )
            raise InvalidInput(
                (
                    "The MW API does not have any info related to the rev-id "
                    "provided as input ({}), therefore it is not possible to "
                    "extract features properly. One possible cause is the deletion "
                    "of the page related to the revision id. "
                    "Please contact the ML-Team if you need more info.".format(rev_id)
                ),
            )

        # The output returned by the MW API is a little
        # convoluted and probably meant for batches of rev-ids.
        # In our case we fetch only one rev-id at the time,
        # so we can use assumptions about how many elements
        # there will be in the results.
        try:
            revision_info = list(rev_id_doc.get("query").get("pages").values())[0][
                "revisions"
            ][0]
        except Exception as e:
            logging.error(
                "The rev-id doc retrieved from the MW API "
                "does not contain all the data needed "
                "to extract features properly. "
                "The error is {} and the document is: {}".format(e, rev_id_doc)
            )
            raise InferenceError(
                "The rev-id doc retrieved from the MW API "
                "does not contain all the data needed "
                "to extract features properly. "
                "Please contact the ML-Team if the issue persists."
            )

        if fetch_extra_info:
            parent_rev_id = revision_info.get("parentid")
            user = revision_info.get("user")
            user_params = {"usprop": {"groups", "registration", "editcount", "gender"}}

            parent_rev_id_doc, user_doc = await asyncio.gather(
                session.get(
                    action="query",
                    prop="revisions",
                    revids=[parent_rev_id],
                    rvslots="main",
                    **params,
                ),
                session.get(
                    action="query", list="users", ususers=[user], **user_params
                ),
            )
    except (
        APIError,
        ConnectionError,
        RequestError,
        TimeoutError,
        TooManyRedirectsError,
    ) as e:
        logging.error(
            "An error has occurred while fetching feature "
            "values from the MW API: {}".format(e)
        )
        raise InferenceError(
            (
                "An error happened while fetching feature values from "
                "the MediaWiki API, please contact the ML-Team "
                "if the issue persists."
            ),
        )

    # Populate the MWAPICache
    http_cache = MWAPICache()
    http_cache.add_revisions_batch_doc([rev_id], rev_id_doc)
    if fetch_extra_info:
        http_cache.add_revisions_batch_doc([parent_rev_id], parent_rev_id_doc)
        http_cache.add_users_batch_doc([user], user_doc)

    return http_cache


@elapsed_time
def fetch_features(
    rev_id, model_features: tuple, extractor: Extractor, cache: Optional[Dict] = None
) -> Dict:
    """Retrieve model features using a Revscoring extractor provided
    as input.
     Parameters:
         rev_id: The MediaWiki revision id to check.
         model_features: The tuple representing the Revscoring model's features.
         extractor: The Revscoring extractor instance to use.
         cache: Optional revscoring cache to ease recomputation of features
                for the same rev-id.

     Returns:
         The feature values computed by the Revscoring extractor.
    """
    try:
        feature_values = list(extractor.extract(rev_id, model_features, cache=cache))
    except MissingResource as e:
        raise InvalidInput(
            f"Missing resource for rev-id {rev_id}: {e}",
        )
    except UnexpectedContentType:
        raise InvalidInput(
            "Unexpected content type for rev-id {rev_id}: {e}",
        )
    except Exception as e:
        raise InvalidInput(
            "Generic error while extracting features "
            "for rev-id {}: {}".format(rev_id, e)
        )

    return feature_values

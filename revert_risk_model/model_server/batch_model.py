import asyncio
import logging
from http import HTTPStatus
from typing import Any, List, Dict, Sequence, Optional

import mwapi
from base_model import RevisionRevertRiskModel
from fastapi import HTTPException
from knowledge_integrity.mediawiki import get_revision
from knowledge_integrity.schema import Revision
from kserve.errors import InferenceError, InvalidInput

from python.preprocess_utils import validate_json_input


class RevisionRevertRiskModelBatch(RevisionRevertRiskModel):
    def __init__(
        self,
        name: str,
        module_name: str,
        model_path: str,
        wiki_url: str,
        aiohttp_client_timeout: int,
        force_http: bool,
    ) -> None:
        super().__init__(
            name,
            module_name,
            model_path,
            wiki_url,
            aiohttp_client_timeout,
            force_http,
        )

    async def get_current_revisions(
        self, session: mwapi.AsyncSession, rev_ids: List[int], lang: str
    ) -> Optional[Sequence[Revision]]:
        tasks = [get_revision(session, rev_id, lang) for rev_id in rev_ids]
        return await asyncio.gather(*tasks)

    async def preprocess(
        self, inputs: Dict[str, Any], headers: Dict[str, str] = None
    ) -> Dict[str, Any]:
        inputs = validate_json_input(inputs)
        # During the experimental phase, we only accept requesting multiple
        # rev_ids with the same lang. This allows us to use the same async
        # session.
        if len({x.get("lang") for x in inputs["instances"]}) > 1:
            logging.error("More than one language in the request.")
            raise InvalidInput(
                "Requesting multiple revisions should have the same language."
            )
        rev_ids = [x.get("rev_id") for x in inputs["instances"]]
        lang = inputs["instances"][0].get("lang")
        # TODO:
        # self.validate_inputs(lang, rev_id)
        mw_host = self.get_mediawiki_host(lang)
        session = mwapi.AsyncSession(
            # Host is set to http://api-ro.discovery.wmnet within WMF
            # network in Lift Wing. Alternatively, it can be set to
            # https://{lang}.wikipedia.org to call MW API publicly.
            host=self.wiki_url or mw_host,
            user_agent="WMF ML Team revert-risk-model isvc",
            session=self.get_http_client_session("mwapi"),
        )
        # Additional HTTP Host header must be set if the host is http://api-ro.discovery.wmnet
        session.headers["Host"] = mw_host.replace("https://", "").replace("http://", "")
        try:
            rev = await self.get_current_revisions(session, rev_ids, lang)
        except Exception as e:
            # TODO: think about how to do error handling for multiple rev_ids requests,
            # For example, we want to (1) return predictions for the rest of valid
            # revisions, even though there is one revision has problems (e.g. page
            # missing), or (2) raise exception if any revision has problems
            logging.error(
                "An error has occurred while fetching info for revision: "
                f" {rev_ids} ({lang}). Reason: {e}"
            )
            raise InferenceError(
                "An error happened while fetching info for revision "
                "from the MediaWiki API, please contact the ML-Team "
                "if the issue persists."
            )
        # TODO: same as above
        if rev is None:
            logging.error(
                "get_revision returned empty results "
                f"for revision {rev_ids} ({lang})"
            )
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST,
                detail=(
                    "The necessary features cannot be obtained from the "
                    "MediaWiki API. It can be the revision, parent revision, "
                    "page information, or user information. This could be "
                    "because the data does not exist or has been deleted."
                ),
            )
        inputs["revision"] = rev
        return inputs

    def predict(
        self, request: Dict[str, Any], headers: Dict[str, str] = None
    ) -> Dict[str, Any]:
        n_rev = len(request["revision"])
        logging.info(f"Getting {n_rev} rev_ids in the request")
        results = self.ModelLoader.classify_batch(self.model, request["revision"])
        if self.name == "revertrisk-wikidata":
            for i, results in enumerate(results):
                self.check_wikidata_result(results[i], request["revision"][i].comment)
        predictions = [
            {
                "model_name": self.name,
                "model_version": str(self.model.model_version),
                "wiki_db": request["revision"][i].lang + "wiki",
                "revision_id": request["revision"][i].id,
                "output": {
                    "prediction": results[i].prediction,
                    "probabilities": {
                        "true": results[i].probability,
                        "false": 1 - results[i].probability,
                    },
                },
            }
            for i in range(len(results))
        ]
        return {"predictions": predictions}

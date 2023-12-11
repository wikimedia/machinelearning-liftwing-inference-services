import logging
from http import HTTPStatus
from typing import Any, Dict

import mwapi
from base_model import RevisionRevertRiskModel
from fastapi import HTTPException
from knowledge_integrity.revision import get_current_revision
from kserve.errors import InferenceError
from python.preprocess_utils import validate_json_input


class RevisionRevertRiskModelBatch(RevisionRevertRiskModel):
    def __init__(
        self,
        name: str,
        module_name: str,
        model_path: str,
        wiki_url: str,
        aiohttp_client_timeout: int,
    ) -> None:
        super().__init__(
            name, module_name, model_path, wiki_url, aiohttp_client_timeout
        )

    async def preprocess(
        self, inputs: Dict[str, Any], headers: Dict[str, str] = None
    ) -> Dict[str, Any]:
        inputs = validate_json_input(inputs)
        inputs["revision"] = []
        for instance in inputs["instances"]:
            lang = instance.get("lang")
            rev_id = instance.get("rev_id")
            self.validate_inputs(lang, rev_id)
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
            session.headers["Host"] = mw_host.replace("https://", "")
            try:
                rev = await get_current_revision(session, rev_id, lang)
            except Exception as e:
                logging.error(
                    "An error has occurred while fetching info for revision: "
                    f" {rev_id} ({lang}). Reason: {e}"
                )
                raise InferenceError(
                    "An error happened while fetching info for revision "
                    "from the MediaWiki API, please contact the ML-Team "
                    "if the issue persists."
                )
            if rev is None:
                logging.error(
                    "get_current_revision returned empty results "
                    f"for revision {rev_id} ({lang})"
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
            inputs["revision"].append(rev)
        return inputs

    def predict(
        self, request: Dict[str, Any], headers: Dict[str, str] = None
    ) -> Dict[str, Any]:
        predictions = []
        num_inst = len(request["instances"])
        logging.info(f"Getting {num_inst} rev_ids in the request")
        for i in range(num_inst):
            result = self.ModelLoader.classify(self.model, request["revision"][i])
            edit_summary = request["revision"][i].comment
            if not result:
                logging.info(
                    f"Edit type {edit_summary} is not supported at the moment."
                )
                raise HTTPException(
                    status_code=HTTPStatus.BAD_REQUEST,
                    detail=(
                        "Prediction for this type of edit is not supported "
                        "at the moment. Currently only 'claim' edits and "
                        "'description' edits are supported."
                    ),
                )
            prediction = {
                "model_name": self.name,
                "model_version": str(self.model.model_version),
                "wiki_db": request["instances"][i].get("lang") + "wiki",
                "revision_id": request["instances"][i].get("rev_id"),
                "output": {
                    "prediction": result.prediction,
                    "probabilities": {
                        "true": result.probability,
                        "false": 1 - result.probability,
                    },
                },
            }
            predictions.append(prediction)
        return {"predictions": predictions}

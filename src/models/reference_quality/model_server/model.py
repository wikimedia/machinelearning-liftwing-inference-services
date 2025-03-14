import os
import logging
import aiohttp
import kserve
import mwapi
from typing import Any, Dict
from distutils.util import strtobool

from knowledge_integrity.mediawiki import get_parent_revision
from knowledge_integrity.mediawiki import Error
from knowledge_integrity.models.reference_need import load_model, classify
from knowledge_integrity.models.reference_risk import (
    ReferenceRiskModel as BaseReferenceRiskModel,
)

from kserve.errors import InferenceError, InvalidInput

from python.config_utils import get_config
from python.preprocess_utils import (
    validate_json_input,
    check_input_param,
    check_wiki_suffix,
    check_supported_wikis,
)

logging.basicConfig(level=kserve.constants.KSERVE_LOGLEVEL)


class ReferenceNeedModel(kserve.Model):
    def __init__(
        self, name: str, model_path: str, force_http: bool, batch_size: int = 1
    ) -> None:
        super().__init__(name)
        self.name = name
        self.model_path = model_path
        self.force_http = force_http
        self.batch_size = batch_size
        self.host_rewrite_config = get_config(key="mw_host_replace")
        self.aiohttp_client_timeout = 5
        self._http_client_session = {}
        self.ready = False
        self.load()

    def load(self) -> None:
        self.model = load_model(self.model_path)
        logging.info(f"{self.name} supported wikis: {self.model.supported_wikis}.")
        self.ready = True

    def get_mediawiki_host(self, lang):
        protocol = "http" if self.force_http else "https"
        updated_lang = self.host_rewrite_config.get(lang, lang)
        return f"{protocol}://{updated_lang}.wikipedia.org"

    def get_http_client_session(self, endpoint):
        """Returns a aiohttp session for the specific endpoint passed as input.
        We need to do it since sharing a single session leads to unexpected
        side effects (like sharing headers, most notably the Host one)."""
        timeout = aiohttp.ClientTimeout(total=self.aiohttp_client_timeout)
        if (
            self._http_client_session.get(endpoint, None) is None
            or self._http_client_session[endpoint].closed
        ):
            logging.info(f"Opening a new Asyncio session for {endpoint}.")
            self._http_client_session[endpoint] = aiohttp.ClientSession(
                timeout=timeout, raise_for_status=True
            )
        return self._http_client_session[endpoint]

    async def preprocess(
        self, inputs: Dict[str, Any], headers: Dict[str, str] = None
    ) -> Dict[str, Any]:
        inputs = validate_json_input(inputs)
        lang = inputs.get("lang")
        rev_id = inputs.get("rev_id")
        logging.info(f"Received request for revision {rev_id} ({lang}).")
        check_input_param(lang=lang, rev_id=rev_id)
        check_wiki_suffix(lang)
        check_supported_wikis(self.model, lang)
        session = mwapi.AsyncSession(
            host=self.get_mediawiki_host(lang),
            user_agent="WMF ML Team Reference Quality isvc",
            session=self.get_http_client_session("mwapi"),
        )
        try:
            # get_parent_revision fetches only revision information
            rev = await get_parent_revision(session, rev_id, lang)
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
        if isinstance(rev, Error):
            logging.info(f"revision {rev_id} ({lang}): {rev.response_body}")
            raise InvalidInput(
                f"Could not make prediction for revision {rev_id} ({lang})."
                f" Reason: {rev.response_body}"
            )
        inputs["revision"] = rev
        return inputs

    def predict(
        self, request: Dict[str, Any], headers: Dict[str, str] = None
    ) -> Dict[str, Any]:
        result = classify(self.model, request["revision"], self.batch_size)
        return {
            "model_name": self.name,
            "model_version": self.model.model_version,
            "wiki_db": f'{request.get("lang")}wiki',
            "revision_id": request.get("rev_id"),
            "reference_need_score": result.rn_score,
        }


class ReferenceRiskModel(ReferenceNeedModel):
    def __init__(self, name: str, model_path: str, force_http: bool) -> None:
        super().__init__(name, model_path, force_http)

    def load(self) -> None:
        self.model = BaseReferenceRiskModel(self.model_path)
        logging.info(f"{self.name} supported wikis: {self.model.supported_wikis}.")
        self.ready = True

    def predict(
        self, request: Dict[str, Any], headers: Dict[str, str] = None
    ) -> Dict[str, Any]:
        result = self.model.classify(request["revision"])
        output = {
            "model_name": self.name,
            "model_version": result.model_version,
            "wiki_db": f'{request.get("lang")}wiki',
            "revision_id": request.get("rev_id"),
            "reference_count": result.reference_count,
            "survival_ratio": result.survival_ratio,
            "reference_risk_score": result.reference_risk_score,
        }
        extended_output = strtobool(request.get("extended_output", "False"))
        if extended_output:
            output["references"] = result.references
        return output


if __name__ == "__main__":
    model_path = os.environ.get("MODEL_PATH", "/mnt/models/reference-need/model.pkl")
    features_db_path = os.environ.get(
        "FEATURES_DB_PATH", "/mnt/models/reference-risk/features.db"
    )
    force_http = strtobool(os.environ.get("FORCE_HTTP", "False"))

    num_of_workers = int(os.environ.get("NUM_OF_WORKERS", 1))
    model_to_deploy = os.environ.get("MODEL_TO_DEPLOY")
    batch_size = int(os.environ.get("BATCH_SIZE", 1))

    REFERENCE_NEED = "reference-need"
    REFERENCE_RISK = "reference-risk"

    def ref_need():
        return ReferenceNeedModel(
            name=REFERENCE_NEED,
            model_path=model_path,
            force_http=force_http,
            batch_size=batch_size,
        )

    def ref_risk():
        return ReferenceRiskModel(
            name=REFERENCE_RISK, model_path=features_db_path, force_http=force_http
        )

    allowed_models = {REFERENCE_NEED, REFERENCE_RISK}
    if model_to_deploy:
        if model_to_deploy not in allowed_models:
            raise ValueError(
                f"Invalid MODEL_TO_DEPLOY value: {model_to_deploy}. "
                f"Expected one of {allowed_models}."
            )
        models = [ref_need()] if model_to_deploy == REFERENCE_NEED else [ref_risk()]
    else:
        models = [ref_need(), ref_risk()]
    kserve.ModelServer(workers=num_of_workers).start(models)

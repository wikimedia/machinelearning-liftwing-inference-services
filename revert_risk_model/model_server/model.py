import logging
import os
from distutils.util import strtobool

import kserve
from base_model import RevisionRevertRiskModel
from batch_model import RevisionRevertRiskModelBatch

logging.basicConfig(level=kserve.constants.KSERVE_LOGLEVEL)

if __name__ == "__main__":
    model_name = os.environ.get("MODEL_NAME")
    model_path = os.environ.get("MODEL_PATH", "/mnt/models/model.pkl")
    wiki_url = os.environ.get("WIKI_URL")
    force_http = strtobool(os.environ.get("FORCE_HTTP", "False"))
    aiohttp_client_timeout = os.environ.get("AIOHTTP_CLIENT_TIMEOUT", 5)
    if model_name == "revertrisk-language-agnostic":
        module_name = "revertrisk"
        model = RevisionRevertRiskModelBatch(
            model_name,
            module_name,
            model_path,
            wiki_url,
            aiohttp_client_timeout,
            force_http,
        )
    else:
        module_name = model_name.replace("-", "_")
        model = RevisionRevertRiskModel(
            model_name,
            module_name,
            model_path,
            wiki_url,
            aiohttp_client_timeout,
            force_http,
        )
    kserve.ModelServer(workers=1).start([model])

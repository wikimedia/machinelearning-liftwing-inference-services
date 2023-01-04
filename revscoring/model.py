import os
from distutils.util import strtobool

import kserve
from model_server_mp import RevscoringModelMP
from model_servers import RevscoringModel, RevscoringModelType

if __name__ == "__main__":
    inference_name = os.environ.get("INFERENCE_NAME")
    model_type = RevscoringModelType.get_model_type(inference_name)
    mp = strtobool(os.environ.get("ASYNCIO_USE_PROCESS_POOL", "False"))
    if mp:
        model = RevscoringModelMP(inference_name, model_type)
    else:
        model = RevscoringModel(inference_name, model_type)
    kserve.ModelServer(workers=1).start([model])

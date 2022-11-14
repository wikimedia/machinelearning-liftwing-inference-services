import logging
import os

import kserve

from model_servers import RevscoringModel, RevscoringModelType

if __name__ == "__main__":
    inference_name = os.environ.get("INFERENCE_NAME")
    if "articletopic" in inference_name:
        model_type = RevscoringModelType.ARTICLETOPIC
    elif "itemtopic" in inference_name:
        model_type = RevscoringModelType.ITEMTOPIC
    else:
        model_type = RevscoringModelType.DRAFTTOPIC
    model = RevscoringModel(inference_name, model_type)
    kserve.ModelServer(workers=1).start([model])

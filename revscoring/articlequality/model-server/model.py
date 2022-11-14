import logging
import os

import kserve

from model_servers import RevscoringModel, RevscoringModelType

if __name__ == "__main__":
    inference_name = os.environ.get("INFERENCE_NAME")
    if "itemquality" in inference_name:
        model_type = RevscoringModelType.ITEMQUALITY
    else:
        model_type = RevscoringModelType.ARTICLEQUALITY
    model = RevscoringModel(inference_name, model_type)
    kserve.ModelServer(workers=1).start([model])

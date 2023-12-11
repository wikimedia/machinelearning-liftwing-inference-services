import base64
import binascii
import io
from typing import Dict

import kserve
import PIL
from keras.applications.mobilenet import preprocess_input
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from kserve.errors import InvalidInput


def preprocess(img):
    # Converts image to RGB,
    # resizes to 224X224 and
    # reshapes it for the MobileNet V1 Model
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize((224, 224))
    img = img_to_array(img)
    img = img.reshape((1, img.shape[0], img.shape[1], 3))
    img = preprocess_input(img)
    return img


class ImageContentFiltrationModel(kserve.Model):
    def __init__(self, name: str):
        super().__init__(name)
        self.name = name
        self.ready = False

    def load(self):
        model = load_model("/mnt/models/model.h5")
        self.model = model
        self.ready = True

    def predict(self, request: Dict, headers: Dict[str, str] = None) -> Dict:
        try:
            inputs = request["instances"]
            # Input follows the Tensorflow V1 HTTP API for binary values
            # https://www.tensorflow.org/tfx/serving/api_rest#encoding_binary_values
            data = inputs[0]["image"]["b64"]
        except (KeyError, TypeError):
            raise InvalidInput(
                "Input data should have the following format:"
                ' {"instances": [{"image": {"b64": "<base64 encoded string>"}}]}'
            )
        try:
            raw_img_data = base64.b64decode(data)
            input_image = PIL.Image.open(io.BytesIO(raw_img_data))
        except binascii.Error:
            raise InvalidInput("Base64 encoded string is incorrectly padded.")
        except PIL.UnidentifiedImageError:
            raise InvalidInput("Image cannot be opened and identified.")
        preprocessed_image = preprocess(input_image)

        output = self.model.predict(preprocessed_image).tolist()

        result = {}
        result["prob_nsfw"] = output[0][0]
        result["prob_sfw"] = output[0][1]
        return result


if __name__ == "__main__":
    model = ImageContentFiltrationModel("nsfw-model")
    model.load()
    kserve.ModelServer(workers=1).start([model])

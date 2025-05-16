import asyncio
import base64
import logging
import os
import shutil
import tempfile
import time
from typing import Any

import aiohttp
import keras
import kserve
from kserve.errors import InferenceError, InvalidInput
from tensorflow.data import Dataset

from python.preprocess_utils import validate_json_input

logging.basicConfig(level=kserve.constants.KSERVE_LOGLEVEL)


class LogoDetectionModel(kserve.Model):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.name = name
        self.model_path = os.environ.get("MODEL_PATH", "/mnt/models/logo_max_all.keras")
        self.batch_size = int(os.environ.get("BATCH_SIZE", 32))
        self.image_height = int(os.environ.get("IMAGE_HEIGHT", 224))
        self.image_width = int(os.environ.get("IMAGE_WIDTH", 224))
        self.image_size = (self.image_height, self.image_width)
        self.target = "logo"
        self.chunk_size = int(os.environ.get("CHUNK_SIZE", 1024))
        self.images_max_num = int(os.environ.get("IMAGES_MAX_NUM", 50))
        self.image_max_size = int(os.environ.get("IMAGE_MAX_SIZE", 4194304))  # 4MBs
        self.ready = False
        self.model = self.load()

    def load(self) -> None:
        model = keras.models.load_model(self.model_path)
        self.ready = True
        return model

    async def preprocess(
        self, payload: dict[str, Any], headers: dict[str, str] = None
    ) -> tuple[Dataset, str]:
        latency = {}
        preprocess_start_time = time.time()
        payload = validate_json_input(payload)
        self.validate_input_data(payload)
        # Create a request-specific temporary directory to store images
        temp_dir = tempfile.mkdtemp()
        # Save base64 images to a temporary directory
        await self.save_base64_images(payload.get("instances"), temp_dir)
        dataset = self.create_image_dataset(temp_dir)
        # Return dataset, request-specific temp_dir, and latency
        latency["preprocess (s)"] = time.time() - preprocess_start_time
        debug = payload.get("debug", False)  # default to non debug mode
        return dataset, temp_dir, latency, debug

    def predict(
        self, preprocess_results: tuple[Dataset, str], headers: dict[str, str] = None
    ) -> dict[str, Any]:
        predict_start_time = time.time()
        dataset, temp_dir, latency, debug = preprocess_results
        predictions = self.generate_predictions(dataset, temp_dir)
        # Delete the temporary directory after use
        shutil.rmtree(temp_dir)
        latency["predict (s)"] = time.time() - predict_start_time
        latency["total (s)"] = latency["preprocess (s)"] + latency["predict (s)"]
        if debug:
            predictions["latency"] = latency
        return predictions

    def generate_predictions(self, dataset: Dataset, temp_dir: str) -> dict[str, Any]:
        """
        Generates predictions for the given dataset using the specified
        model.
        """
        try:
            predictions_response = []
            raw_predictions = []
            # Iterate through the dataset and make batched predictions
            # NOTE The dataset is small enough for unbatched predictions,
            #      but the model expects inputs with a batch dimension anyway.
            for batch in dataset:
                raw_predictions.extend(self.model(batch))
            for raw_prediction, file_path in zip(raw_predictions, dataset.file_paths):
                prediction = {
                    "filename": os.path.basename(file_path),
                    "target": self.target,
                    "prediction": round(float(raw_prediction[1]), ndigits=4),
                    "out_of_domain": round(float(raw_prediction[0]), ndigits=4),
                }
                predictions_response.append(prediction)
            predictions = {"predictions": predictions_response}
            return predictions
        except Exception as e:
            error_message = f"Error generating predictions: {e}"
            self.cleanup_temp_dir_on_error(temp_dir, error_message)

    def create_image_dataset(self, temp_dir: str) -> Dataset:
        """
        Creates an image dataset based on the specified directory,
        batch_size, and image_size.
        """
        try:
            dataset = keras.utils.image_dataset_from_directory(
                temp_dir,
                labels=None,
                label_mode=None,
                batch_size=self.batch_size,
                image_size=self.image_size,
                shuffle=False,
            )
            return dataset
        except Exception as e:
            error_message = f"Error creating image dataset: {e}"
            self.cleanup_temp_dir_on_error(temp_dir, error_message)

    async def download_image(
        self,
        session: aiohttp.ClientSession,
        image_url: str,
        image_filename: str,
        temp_dir: str,
    ) -> None:
        """
        Downloads an image from the specified URL asynchronously and
        saves it to the given filename, reading the image data in chunks
        of 1024 bytes i.e defined chunk size.
        """
        try:
            async with session.get(image_url) as response:
                response.raise_for_status()  # Raise an exception for HTTP errors
                # Confirm content length size before downloading the image.
                # This is a safety net to prevent the download of big images.
                content_length = response.headers.get("Content-Length")
                if content_length is None:
                    error_message = "Content-Length header is missing"
                    self.cleanup_temp_dir_on_error(temp_dir, error_message)
                elif int(content_length) > self.image_max_size:
                    error_message = f"Image: {image_url} \
                        exceeds the maximum allowed size of {self.image_max_size} bytes."
                    self.cleanup_temp_dir_on_error(temp_dir, error_message)
                with open(image_filename, "wb") as f:
                    while True:
                        chunk = await response.content.read(self.chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
        except aiohttp.ClientError as e:
            error_message = f"Error downloading image from {image_url}: {e}"
            self.cleanup_temp_dir_on_error(temp_dir, error_message)

    async def download_images_to_temp_dir(
        self, input_data: list[dict[str, str]], temp_dir: str
    ) -> None:
        """
        Downloads images from URLs provided in the input data asynchronously
        and saves them to the specified temporary directory.
        """
        async with aiohttp.ClientSession() as session:
            tasks = []
            for data in input_data:
                image_url = data["url"]
                image_filename = os.path.join(temp_dir, data["filename"])
                task = self.download_image(session, image_url, image_filename, temp_dir)
                tasks.append(task)
            await asyncio.gather(*tasks)

    async def save_base64_image(self, data: dict[str, str], temp_dir: str) -> None:
        """
        Saves base64 encoded image from input data to the specified
        temporary directory.
        """
        image_name = data["filename"]

        try:
            # Decode base64 image string to bytes
            image_bytes = base64.b64decode(data["image"])
        except Exception as e:
            error_message = f"Error decoding image {image_name}: {e}"
            self.cleanup_temp_dir_on_error(temp_dir, error_message)

        # Check image size before saving
        if len(image_bytes) > self.image_max_size:
            error_message = f"Image: {image_name} \
                exceeds the maximum allowed size of {self.image_max_size} bytes."
            self.cleanup_temp_dir_on_error(temp_dir, error_message)

        image_path = os.path.join(temp_dir, image_name)
        with open(image_path, "wb") as f:
            f.write(image_bytes)

    async def save_base64_images(
        self, input_data: list[dict[str, str]], temp_dir: str
    ) -> None:
        """
        Saves base64 encoded images from input data to the specified
        temporary directory asynchronously.
        """
        tasks = []
        for data in input_data:
            tasks.append(self.save_base64_image(data, temp_dir))
        await asyncio.gather(*tasks)

    def cleanup_temp_dir_on_error(self, temp_dir: str, error_message: str) -> None:
        """
        Cleans up the temporary directory, logs and raises the provided error.
        """
        # First, delete the temporary directory to avoid accumulation
        # of unused temp_dirs in the pod.
        shutil.rmtree(temp_dir)
        logging.error(error_message)
        raise InferenceError(error_message)

    def validate_input_data(self, payload: dict[str, Any]) -> None:
        """
        Validates the input data to ensure it has the required fields
        and their values are of the expected types.
        """
        if "instances" not in payload:
            error_message = "Missing required key 'instances' in input data."
            logging.error(error_message)
            raise InvalidInput(error_message)

        if not isinstance(payload.get("debug", False), bool):
            error_message = "The 'debug' key should be a boolean."
            logging.error(error_message)
            raise InvalidInput(error_message)

        input_data = payload.get("instances")
        # Check maximum number of images in input data
        if len(input_data) > self.images_max_num:
            error_message = f"Input data should have less than \
                {self.images_max_num} images."
            logging.error(error_message)
            raise InvalidInput(error_message)

        required_keys = ["filename", "image", "target"]
        for data in input_data:
            # Check if all required keys are present
            if not all(key in data for key in required_keys):
                error_message = "Input data has invalid key(s). \
                    Ensure each item contains: filename, image, and target."
                logging.error(error_message)
                raise InvalidInput(error_message)

            # Check if values of required keys are strings
            for key in required_keys:
                if not isinstance(data[key], str):
                    error_message = f"Value for key '{key}' should be a string."
                    logging.error(error_message)
                    raise InvalidInput(error_message)


if __name__ == "__main__":
    model_name = os.environ.get("MODEL_NAME")
    model = LogoDetectionModel(model_name)
    kserve.ModelServer(workers=1).start([model])

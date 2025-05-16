import gc
import importlib
import logging
import os
from distutils.util import strtobool
from typing import Any

import kserve
import torch
from kserve.errors import InferenceError
from transformers import AutoModelForCausalLM, AutoTokenizer

from python.preprocess_utils import validate_json_input

logging.basicConfig(level=kserve.constants.KSERVE_LOGLEVEL)


class LLM(kserve.Model):
    def __init__(self, model_name: str) -> None:
        super().__init__(model_name)
        self.model_path = os.environ.get("MODEL_PATH", "/mnt/models/")
        self.quantized = strtobool(os.environ.get("QUANTIZED", "False"))
        self.dtype = os.environ.get("DTYPE", "torch.bloat16")
        self.attn_implementation = os.environ.get(
            "ATTN_IMPLEMENTATION", "flash_attention_2"
        )
        self.tokenizer = None
        self.model = None
        self.model_name = model_name
        self.ready = False
        self.device = None
        self.model, self.tokenizer = self.load()
        self.validate_inputs()

    def validate_inputs(self):
        """
        Check if a GPU exists and if the model is intructed for quantization.
        Only GPU models can be quantized, so if a GPU is not available we throw the same
        RuntimeError that transformers throws when trying to quantize a model without GPU.
        """
        if not torch.cuda.is_available() and self.quantized:
            raise RuntimeError(
                "RuntimeError: No GPU found. A GPU is needed for quantization."
            )

    def load(self) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            local_files_only=True,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            torch_dtype=self.dtype,
            load_in_8bit=self.quantized,
            attn_implementation=self.attn_implementation,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, local_files_only=True
        )
        self.ready = True
        return model, tokenizer

    def check_gpu(self):
        """
        Loads the model in the GPU's memory and updates its reference.
        This function needs to run after the webserver's initialization
        (that forks and creates new processes, see https://github.com/pytorch/pytorch/issues/83973).
        Since quantization happens only with GPU we skip the check as it is already on GPU and
        trying to load it again would raise an error.
        """
        if not self.device and not self.quantized:
            # The cuda keyword is internally translated to hip and rocm is used if available.
            # https://pytorch.org/docs/stable/notes/hip.html
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logging.info(f"Using device: {self.device}")
            # TODO: in the event where we have a GPU error the model should be reloaded to the GPU.

    def preprocess(
        self, inputs: dict[str, Any], headers: dict[str, str] = None
    ) -> dict[str, Any]:
        try:
            self.check_gpu()
            inputs = validate_json_input(inputs)
            prompt = inputs.get("prompt")
            result_length = inputs.get("result_length", 100)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            inputs["result_length"] = result_length + inputs["input_ids"].size()[1]
            return inputs
        except RuntimeError as e:
            logging.exception("An error has occurred in preprocess.")
            # HIP is a layer offered by AMD ROCm to translate CUDA code/runtime
            # into something hardware agnostic. If a RuntimeError containing
            # the msg "HIP etc.." is raised it means that a GPU error occurred.
            if "HIP" in str(e):
                logging.error(
                    "HIP error registered, the GPU may be into an inconsistent "
                    "state, dropping memory and forcing its re-initialization."
                )
                # Delete tensors from GPU memory
                # and force the gc collection to be sure about the deletion.
                del self.device
                gc.collect()
                # Restore the device attribute to None so it can be
                # re-initialized during the next preprocess call.
                # Call also empty_cache() to drop any extra data saved on the GPU.
                self.device = None
                torch.cuda.empty_cache()
            raise InferenceError(
                "An error has occurred in preprocess. Please contact the ML-team "
                "if the issue persists."
            )

    def predict(
        self, request: dict[str, Any], headers: dict[str, str] = None
    ) -> dict[str, Any]:
        with torch.inference_mode():
            outputs = self.model.generate(
                request["input_ids"],
                max_length=request["result_length"],
                do_sample=True,
                top_k=50,
                top_p=0.9,
            )
        response = self.tokenizer.decode(outputs[0])
        return {"model_name": self.model_name, "response": response}


if __name__ == "__main__":
    """
    We use the variable llm_class_name to dynamically import the LLM class that is required
    for the model we want to deploy. The default value is model.LLM, which means
    that the class LLM is used which supports most LLMs out of the box.
    """
    model_name = os.environ.get("MODEL_NAME")
    llm_class_name = os.environ.get("LLM_CLASS", "llm.LLM")
    try:
        module_name, class_name = llm_class_name.split(".")
        module_name = ".".join(["src.models", module_name])
        module = importlib.import_module(module_name)
        llm_class = getattr(module, class_name)
    except (ImportError, AttributeError):
        raise ImportError(
            f"Unable to load class: {llm_class_name}. LLM_CLASS environment variable must be set,"
            "and follow the format: module_name.class_name"
        )
    model = llm_class(model_name)
    kserve.ModelServer(workers=1).start([model])

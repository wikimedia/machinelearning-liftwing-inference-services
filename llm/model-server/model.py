import logging
import os
from typing import Any, Dict, Tuple

import kserve
import torch
from kserve.errors import InferenceError
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=kserve.constants.KSERVE_LOGLEVEL)


class LLM(kserve.Model):
    def __init__(self, model_name: str) -> None:
        super().__init__(model_name)
        self.tokenizer = None
        self.model = None
        self.model_name = model_name
        self.ready = False
        self.device = None
        self.model, self.tokenizer = self.load()

    def load(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        model_path = "/mnt/models/"
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.ready = True
        return model, tokenizer

    def check_gpu(self):
        """
        Loads the model in the GPU's memory and updates its reference.
        This function needs to run after the webserver's initialization
        (that forks and creates new processes, see https://github.com/pytorch/pytorch/issues/83973).
        """
        if not self.device:
            # The cuda keyword is internally translated to hip and rocm is used if available.
            # https://pytorch.org/docs/stable/notes/hip.html
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logging.info(f"Using device: {self.device}")
            self.model = self.model.to(self.device)

    def preprocess(
        self, inputs: Dict[str, Any], headers: Dict[str, str] = None
    ) -> Dict[str, Any]:
        try:
            self.check_gpu()
            prompt = inputs.get("prompt")
            result_length = inputs.get("result_length")
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            inputs["result_length"] = result_length + inputs["input_ids"].size()[1]
            return inputs
        except RuntimeError as e:
            logging.exception("An error has occurred in preprocess.")
            # HIP is a layer offered by AMD ROCm to translate CUDA code/runtime
            # into something hardware agnostic. If a RuntimeError containing
            # the msg "HIP etc.." is raised it means that a GPU error occurred.
            if "HIP" in str(e):
                torch.cuda.empty_cache()
            raise InferenceError(
                "An error has occurred in preprocess. Please contact the ML-team "
                "if the issue persists."
            )

    def predict(
        self, request: Dict[str, Any], headers: Dict[str, str] = None
    ) -> Dict[str, Any]:
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
    model_name = os.environ.get("MODEL_NAME")
    model = LLM(model_name)
    kserve.ModelServer(workers=1).start([model])

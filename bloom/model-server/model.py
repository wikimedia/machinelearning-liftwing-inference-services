import os
import logging
from typing import Any, Dict, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
import kserve

logging.basicConfig(level=kserve.constants.KSERVE_LOGLEVEL)


class BloomModel(kserve.Model):
    def __init__(self, model_name: str) -> None:
        super().__init__(model_name)
        self.tokenizer = None
        self.model = None
        self.model_name = model_name
        self.ready = False
        self.model, self.tokenizer = self.load()

    def load(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        model_path = "/mnt/models/"
        model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.ready = True
        return model, tokenizer

    def preprocess(
        self, inputs: Dict[str, Any], headers: Dict[str, str] = None
    ) -> Dict[str, Any]:
        prompt = inputs.get("prompt")
        result_length = inputs.get("result_length")
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cpu")
        inputs["result_length"] = result_length + inputs["input_ids"].size()[1]
        return inputs

    def predict(
        self, request: Dict[str, Any], headers: Dict[str, str] = None
    ) -> Dict[str, Any]:
        outputs = model.generate(
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
    model = BloomModel(model_name)
    kserve.ModelServer(workers=1).start([model])

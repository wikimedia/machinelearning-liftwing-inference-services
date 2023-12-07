import logging
import os
from typing import Any, Dict, Tuple

from kserve.errors import InferenceError
from python.preprocess_utils import validate_json_input
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from llm.model import LLM


class NLLB(LLM):
    def __init__(self, model_name: str):
        self.src_lang = os.environ.get("SRC_LANG", "eng_Latn")
        super().__init__(model_name)

    def load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            local_files_only=True,
            src_lang=self.src_lang,
            low_cpu_mem_usage=True,
        )
        return tokenizer

    def load(self) -> Tuple[AutoModelForSeq2SeqLM, AutoTokenizer]:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_path,
            local_files_only=True,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            load_in_8bit=self.quantized,
        )
        tokenizer = self.load_tokenizer()
        self.ready = True
        return model, tokenizer

    def preprocess(
        self, inputs: Dict[str, Any], headers: Dict[str, str] = None
    ) -> Dict[str, Any]:
        """
        Reading the source and the target language from the request.
        The list of available languages can be found in the paper "No Language Left Behind: Scaling Human-Centered Machine Translation"
        https://arxiv.org/pdf/2207.04672.pdf
        """

        try:
            self.check_gpu()
            inputs = validate_json_input(inputs)
            prompt = inputs.get("prompt")
            tgt_lang = inputs.get("tgt_lang")
            result_length = inputs.get("result_length", 0)
            if "src_lang" in inputs:
                self.src_lang = inputs["src_lang"]
                self.tokenizer = self.load_tokenizer()
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            inputs["result_length"] = result_length + inputs["input_ids"].size()[1]
            inputs["tgt_lang"] = tgt_lang
            return inputs
        except RuntimeError:
            logging.exception("An error has occurred in preprocess.")
            raise InferenceError(
                "An error has occurred in preprocess. Please contact the ML-team "
                "if the issue persists."
            )

    def predict(
        self, request: Dict[str, Any], headers: Dict[str, str] = None
    ) -> Dict[str, Any]:
        tgt_lang = request.get("tgt_lang")
        logging.info(f"Translating from {self.src_lang} to {tgt_lang}")
        translated_tokens = self.model.generate(
            request["input_ids"],
            forced_bos_token_id=self.tokenizer.lang_code_to_id[tgt_lang],
            max_length=request["result_length"],
        )

        response = self.tokenizer.batch_decode(
            translated_tokens, skip_special_tokens=True
        )[0]
        return {"model_name": self.model_name, "response": response}

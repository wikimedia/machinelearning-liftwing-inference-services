import logging
import os
from typing import Any

import ctranslate2 as ctr2
import sentencepiece as spm
from kserve.errors import InferenceError

from python.preprocess_utils import validate_json_input
from python.resource_utils import get_cpu_count
from src.models.llm import NLLB


class NLLBCTranslate(NLLB):
    def __init__(self, model_name: str):
        self.inter_threads = int(os.environ.get("CT2_INTER_THREADS", get_cpu_count()))
        self.intra_threads = int(os.environ.get("CT2_INTRA_THREADS", 0))
        super().__init__(model_name)

    def load_tokenizer(self):
        tokenizer = spm.SentencePieceProcessor()
        tokenizer.load(os.path.join(self.model_path, "sentencepiece.bpe.model"))
        return tokenizer

    def load(self) -> tuple[ctr2.Translator, spm.SentencePieceProcessor]:
        model = ctr2.Translator(
            self.model_path,
            intra_threads=self.intra_threads,
            inter_threads=self.inter_threads,
        )
        tokenizer = self.load_tokenizer()
        self.ready = True
        logging.info(
            f"Ctranslate2 Model loaded using intra threads: {self.intra_threads} "
            f"and inter threads: {self.inter_threads}"
        )
        return model, tokenizer

    async def preprocess(
        self, inputs: dict[str, Any], headers: dict[str, str] = None
    ) -> dict[str, Any]:
        """
        Reading the source and the target language from the request.
        The list of available languages can be found in the paper "No Language Left Behind: Scaling Human-Centered Machine Translation"
        https://arxiv.org/pdf/2207.04672.pdf
        """

        try:
            inputs = validate_json_input(inputs)
            prompt = inputs.get("prompt")
            if "src_lang" in inputs:
                self.src_lang = inputs["src_lang"]
            inputs["sentences"] = prompt.strip().splitlines()
            return inputs
        except RuntimeError:
            logging.exception("An error has occurred in preprocess.")
            raise InferenceError(
                "An error has occurred in preprocess. Please contact the ML-team "
                "if the issue persists."
            )

    async def predict(
        self, request: dict[str, Any], headers: dict[str, str] = None
    ) -> dict[str, Any]:
        tgt_lang = request.get("tgt_lang")
        sentences = request.get("sentences")
        num_beams = request.get("num_beams", 1)
        logging.info(
            f"Translating from {self.src_lang} to {tgt_lang} using {num_beams} beams."
        )
        response = await self.translate_sentences(
            sentences, self.src_lang, tgt_lang, num_beams
        )
        return {"model_name": self.model_name, "response": response}

    def encode_sentence(self, sentence: list[str], src_lang: str) -> list[str]:
        return self.tokenizer.encode(sentence, out_type=str) + ["</s>", src_lang]

    def decode_result(self, result: ctr2.TranslationResult) -> str:
        decoded_result = self.tokenizer.decode(result.hypotheses[0][1:])
        return decoded_result

    async def translate_sentences(
        self, sentences: list[str], src_lang: str, tgt_lang: str, num_beams: int
    ):
        tokenized_sentences = [
            self.encode_sentence(sentence, src_lang) for sentence in sentences
        ]
        target_prefix = [[tgt_lang] for _ in sentences]

        results = self.model.translate_iterable(
            tokenized_sentences,
            target_prefix=target_prefix,
            asynchronous=True,
            batch_type="tokens",
            max_batch_size=1024,
            beam_size=num_beams,
        )

        translations = [self.decode_result(result) for result in results]
        return "\n".join(translations)

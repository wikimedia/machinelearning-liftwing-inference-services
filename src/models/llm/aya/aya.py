import os
from typing import Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.models.llm import LLM


class Aya(LLM):
    def __init__(self, model_name: str):
        self.quantization_mode = os.environ.get("BITSANDBYTES_DTYPE", "int4")
        self.device_map = os.environ.get("DEVICE", "cuda:0")
        super().__init__(model_name)

    def load(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        if self.quantization_mode == "int4":
            quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        elif self.quantization_mode == "int8":
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            quantization_config = None
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map=self.device_map,
            local_files_only=True,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            quantization_config=quantization_config,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, local_files_only=True
        )
        self.ready = True
        return model, tokenizer

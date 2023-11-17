import torch
from transformers import MBartTokenizer
from transformers import BertModel, BertTokenizer
from transformers.tokenization_utils_base import BatchEncoding
from transformers import MBartConfig

from descartes.src.models.descartes_mbart import MBartForConditionalGenerationDescartes

BERT_PATH = "bert-base-multilingual-uncased"
LANG_DICT = {
    "en": "en_XX",
    "fr": "fr_XX",
    "it": "it_IT",
    "es": "es_XX",
    "de": "de_DE",
    "nl": "nl_XX",
    "ja": "ja_XX",
    "zh": "zh_CN",
    "ko": "ko_KR",
    "vi": "vi_VN",
    "ru": "ru_RU",
    "cs": "cs_CZ",
    "fi": "fi_FI",
    "lt": "lt_LT",
    "lv": "lv_LV",
    "et": "et_EE",
    "ar": "ar_AR",
    "tr": "tr_TR",
    "ro": "ro_RO",
    "kk": "kk_KZ",
    "gu": "gu_IN",
    "hi": "hi_IN",
    "si": "si_LK",
    "my": "my_MM",
    "ne": "ne_NP",
}


def prepare_inputs(inputs, device):
    """
    Prepare :obj:`inputs` before feeding them to the model, converting them to tensors if they are not already and
    handling potential state.
    """
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(device)
        elif isinstance(v, dict):
            for key, val in v.items():
                if isinstance(val, torch.Tensor):
                    v[key] = val.to(device)
                elif isinstance(val, BatchEncoding) or isinstance(val, dict):
                    for k1, v1 in val.items():
                        if isinstance(v1, torch.Tensor):
                            val[k1] = v1.to(device)
    return inputs


class ModelLoader:
    """
    Class responsible for loading and managing the model.

    The approach we used here is similar to the way other model-servers in this repo
    use KServe's load() and predict() methods to call encapsulated framework-specific
    load and predict methods. For example in revscoring model-servers, the KServe
    load() and predict() methods call load() and score() from revscoring. Similarly
    in this case, KServe's load() and predict() are calling load_model() and predict()
    from the ModelLoader class.

    This class is the original implementation by the Research Team that we kept when
    porting the code to LiftWing.
    """

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.tokenizer_bert = None
        self.device = None

    def load_model(self, output_dir):
        """Load model from the specified directory."""
        config = MBartConfig.from_pretrained(output_dir)
        config.graph_embd_length = 128
        model = MBartForConditionalGenerationDescartes.from_pretrained(
            output_dir, config=config
        )
        tokenizer = MBartTokenizer.from_pretrained(output_dir)
        tokenizer_bert = BertTokenizer.from_pretrained(BERT_PATH)
        bert_model = BertModel.from_pretrained(BERT_PATH)
        model.model_bert = bert_model
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        self.model = model
        self.tokenizer = tokenizer
        self.tokenizer_bert = tokenizer_bert
        self.device = device

    def predict(
        self, sources, descriptions, tgt_lang, num_beams=1, num_return_sequences=1
    ):
        """Predict text descriptions for the given sources and descriptions."""
        batch = {}
        input_ids = {}
        attention_mask = {}
        # process first paragraphs
        for lang, lang_code in LANG_DICT.items():
            if lang in sources:
                source = sources[lang]
            else:
                source = ""
            if source:
                self.tokenizer.src_lang = lang_code
                batch_enc = self.tokenizer([source], padding=True, truncation=True)
                input_ids[lang] = torch.tensor(batch_enc["input_ids"])
                attention_mask[lang] = torch.tensor(batch_enc["attention_mask"])
            else:
                input_ids[lang] = None
                attention_mask[lang] = None
        # process descriptions
        bert_inputs = {}
        for lang, description in descriptions:
            if lang != tgt_lang:
                bert_outs = self.tokenizer_bert(
                    [description], padding=True, truncation=True, return_tensors="pt"
                )
                bert_inputs[lang] = bert_outs
        batch["input_ids"] = input_ids
        batch["attention_mask"] = attention_mask
        batch["graph_embeddings"] = None
        batch["bert_inputs"] = bert_inputs
        batch = prepare_inputs(batch, self.device)
        tokens = self.model.generate(
            **batch,
            max_length=20,
            min_length=2,
            length_penalty=2.0,
            num_beams=num_beams,
            early_stopping=True,
            target_lang=LANG_DICT[tgt_lang],
            decoder_start_token_id=self.tokenizer.lang_code_to_id[LANG_DICT[tgt_lang]],
            num_return_sequences=num_return_sequences
        )
        output = self.tokenizer.batch_decode(tokens, skip_special_tokens=True)
        return output

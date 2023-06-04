from transformers import pipeline
from enum import Enum
from transformers import XLMRobertaForMaskedLM, XLMRobertaTokenizer, DataCollatorForLanguageModeling
from transformers import BertTokenizer, BertModel, BertForMaskedLM
import json 

class Models(Enum):
    XLMR = "xlm-roberta-base"
    BERT = "bert-base-uncased"

    @classmethod
    def has_value(cls, value):
        return any(value == item.value for item in cls)
        
    @classmethod
    def has_key(cls, value):
        return any(value == item.name for item in cls)

def load_model(model:Models, model_attributes:dict = {}, model_name = 'base'):
    if model.value == "xlm-roberta-base":
        assert "top_k" in model_attributes
        assert "pipeline" in model_attributes

        if model_name == 'base':
            model = XLMRobertaForMaskedLM.from_pretrained("xlm-roberta-base")
            return model

        if model_name =='french':
            model = XLMRobertaForMaskedLM.from_pretrained("xlm-roberta-finetuned\\french_model")

        if model_name =='spanish':
            model = XLMRobertaForMaskedLM.from_pretrained("xlm-roberta-finetuned\\spanish_model")

        if model_name =='greek':
            model = XLMRobertaForMaskedLM.from_pretrained("xlm-roberta-finetuned\\greek_model")

        if model_name =='english':
            model = XLMRobertaForMaskedLM.from_pretrained("xlm-roberta-finetuned\\english_model")

        
    elif model.value == "bert-base-uncased":
        assert "top_k" in model_attributes
        assert "pipeline" in model_attributes
        model = BertForMaskedLM.from_pretrained("bert-base-uncased")
    else:
        raise Exception(f"Model {model.value} not implemented.")
    return model

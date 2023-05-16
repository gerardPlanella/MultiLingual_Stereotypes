from transformers import pipeline
from enum import Enum
from transformers import XLMRobertaForMaskedLM, XLMRobertaTokenizer, DataCollatorForLanguageModeling

class Models(Enum):
    XLMR = "xlm-roberta-base"

    @classmethod
    def has_value(cls, value):
        return any(value == item.value for item in cls)
        
    @classmethod
    def has_key(cls, value):
        return any(value == item.name for item in cls)

def load_model(model:Models, model_attributes:dict = {}, pre_trained = False):
    if model.value == "xlm-roberta-base":
        assert "top_k" in model_attributes
        assert "pipeline" in model_attributes

        if pre_trained == True:
            model = XLMRobertaForMaskedLM.from_pretrained("xlm-roberta-base")
            # model = XLMRobertaForMaskedLM.from_pretrained("./checkpoint-10625")
            return model
        model = XLMRobertaForMaskedLM.from_pretrained("xlm-roberta-base")
        # model = XLMRobertaForMaskedLM.from_pretrained("xlm-roberta-finetuned/fox_news/checkpoint-500/")
        # model = pipeline(model_attributes["pipeline"], model=model.value, top_k=model_attributes["top_k"])
        model = pipeline(model_attributes["pipeline"], model = model, tokenizer=XLMRobertaTokenizer.from_pretrained("xlm-roberta-base"), top_k=model_attributes["top_k"])
         
    else:
        raise Exception(f"Model {model.value} not implemented.")
    return model

if __name__ == "__main__":
    model = Models('xlm-roberta-base')
    model_attributes = { 
        "pipeline":"fill-mask", 
        "top_k":10
    }
    unmasker = load_model(model, model_attributes)
    print(unmasker("Why are there so many <mask> in my country?"))
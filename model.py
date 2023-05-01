from transformers import pipeline


def load_model(model_name, model_attributes:dict = {}):
    if model_name == "xlm-roberta-base":
        assert "top_k" in model_attributes
        assert "pipeline" in model_attributes

        unmasker = pipeline(model_attributes["pipeline"], model=model_name, top_k=model_attributes["top_k"])
    else:
        raise Exception(f"Model {model_name} not implemented.")
    return unmasker

if __name__ == "__main__":
    model_name = 'xlm-roberta-base'
    model_attributes = { 
        "pipeline":"fill-mask", 
        "top_k":10
    }
    unmasker = load_model(model_name, model_attributes)
    print(unmasker("Why do <mask> suck it so well ?"))
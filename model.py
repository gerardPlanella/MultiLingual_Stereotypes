from transformers import pipeline

def load_model():
    unmasker = pipeline('fill-mask', model='xlm-roberta-base')
    return unmasker

if __name__ == "__main__":
    unmasker = pipeline('fill-mask', model='xlm-roberta-base', top_k = 10)
    print(unmasker("Why do madridistas think that catalan people are so <mask> ?."))
from transformers import pipeline

def load_model():
    unmasker = pipeline('fill-mask', model='xlm-roberta-base')
    return unmasker

if __name__ == "__main__":
    unmasker = pipeline('fill-mask', model='xlm-roberta-base', top_k = 200)
    print(unmasker("Why do <mask> suck it so well ?"))
import requests 
import json
from nltk.stem import SnowballStemmer

def load_lexicon(path):
    # Parse the lexicon and store it in a dictionary
    languages = {"English Word", "French", "Serbian", "Greek", "Catalan", "Spanish"}

    emolex = {}
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split('\t')
        lang_indices = [i for i, lang in enumerate(header) if lang in languages]

        french_stemmer = SnowballStemmer("french")
        
        for line in f.readlines():
            values = line.strip().split('\t')
            if len(values) == len(header):
                emotion_vector = [int(values[i]) for i in range(1, 11)]
                for i in lang_indices:
                    word = values[i]
                    if header[i] == "French":
                        word = french_stemmer.stem(word)
                    emolex[word] = emotion_vector
            else:
                print(len(values))

    with open("data/emolex.json", "w", encoding="utf-8") as f:
        json.dump(emolex, f, ensure_ascii=False, indent=4)
        
    print("EmoLex dictionary saved to emolex.json")

if __name__ == "__main__":
    load_lexicon("data/NRC-Emotion-Lexicon/NRC-Emotion-Lexicon-ForVariousLanguages.txt")

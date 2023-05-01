import requests 
import json
from nltk.stem import SnowballStemmer
from enum import Enum
from typing import List, Dict

class Stem_Language(Enum):
    English = False
    Spanish = False
    French = True
    Greek = False
    Croatian = False
    Catalan = False
    Serbian = False

class Language(Enum):
    English = "english"
    Spanish = "spanish"
    French = "french"
    Greek = "greek"
    Croatian = "croatian"
    Catalan = "catalan"
    Serbian = "serbian"

    @classmethod
    def to_dict(cls):
        result = {}
        for lang in cls:
            stem_enabled = Stem_Language[lang.name].value
            lang_dict = {'stem': stem_enabled, 'NRC': NRC_Language[lang.name].value}
            if stem_enabled and lang.value in SnowballStemmer.languages:
                lang_dict['stemmer'] = SnowballStemmer(lang.value)
            else:
                lang_dict['stemmer'] = None
                raise Exception("Language Doesnt Have Snowball Stemmer")
            result[lang.value] = lang_dict
        return result

class NRC_Language(Enum):
    English = "English Word"
    Spanish = "Spanish"
    French = "French"
    Greek = "Greek"
    Croatian = "Croatian"
    Catalan = "Catalan"
    Serbian = "Serbian"

    @classmethod
    def to_list(cls) -> List[str]:
        return [label.value for label in cls]


class Emotions(Enum):
    ANGER = "anger"
    ANTICIPATION = "anticipation"
    DISGUST = "disgust"
    FEAR = "fear"
    JOY = "joy"
    NEGATIVE = "negative"
    POSITIVE = "positive"
    SADNESS = "sadness"
    SURPRISE = "surprise"
    TRUST = "trust"

    @classmethod
    def to_list(cls) -> List[str]:
        return [label.value for label in cls]


def load_lexicon(path, language_dict, output_path="data/emolex.json"):
    # Parse the lexicon and store it in a dictionary
    emolex = {}
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split('\t')
        lang_indices = [i for i, lang in enumerate(header) if lang in NRC_Language.to_list()]
        
        for line in f.readlines():
            values = line.strip().split('\t')
            if len(values) == len(header):
                emotion_vector = [int(values[i]) for i in range(1, 11)]
                for i in lang_indices:
                    word = values[i]
                    lang = Language[NRC_Language(header[i]).name].value
                    if language_dict[lang]["stem"]:
                        word = language_dict[lang]["stemmer"].stem(word)
                    emolex[word] = emotion_vector
            else:
                continue

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(emolex, f, ensure_ascii=False, indent=4)
        
    print(f"EmoLex dictionary saved to {output_path}")

if __name__ == "__main__":
    output_path = "data/emolex.json"
    languages =  Language.to_dict()
    lexicon_path = "data/NRC-Emotion-Lexicon/NRC-Emotion-Lexicon-ForVariousLanguages.txt"
    load_lexicon(lexicon_path, languages, output_path=output_path)

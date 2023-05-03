from google.cloud import translate_v2 as translate
import json
from typing import List
from data import Language

import os 

#To create your own API credentials file check https://cloud.google.com/translate/docs/setup
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] ="data\multilingualstereotypes-b8b5de5e15d6.json"

def translate_json(input_path: str, output_path: str, input_lang: Language, output_lang: Language, ignored_words: List[str]):
    lang_codes = {
        Language.English: "en",
        Language.Spanish: "es",
        Language.French: "fr",
        Language.Greek: "el",
        Language.Croatian: "hr",
        Language.Catalan: "ca",
        Language.Serbian: "sr"
    }

    input_lang_code = lang_codes[input_lang]
    output_lang_code = lang_codes[output_lang]
    
    # Set up the translation client
    translate_client = translate.Client()

    # Load the input JSON file
    with open(input_path, "r") as f:
        input_json = json.load(f)

    # Translate the prompts and items in the JSON file
    for category in input_json:
        if category == "general_prompts":
            # Translate the general prompts
            prompts = input_json[category]
            translated_prompts = []
            for prompt in prompts:
                #print(prompt)
                translation = translate_client.translate(prompt, source_language=input_lang_code, target_language=output_lang_code)
                translation = translation['translatedText'].encode('utf-8').decode('utf-8')
                translated_prompts.append(translation)

            # Update the JSON file with the translated prompts
            input_json[category] = translated_prompts
        else:
            # Translate the prompts and items for this category
            prompts = input_json[category]["prompts"]
            items = input_json[category]["items"]

            # Translate the prompts
            translated_prompts = []
            for prompt in prompts:
                #print(prompt)
                translation = translate_client.translate(prompt, source_language=input_lang_code, target_language=output_lang_code)
                translation = translation['translatedText'].encode('utf-8').decode('utf-8')
                translated_prompts.append(translation)

            # Translate the items
            translated_items = []
            for item in items:
                #print(item)
                translation = translate_client.translate(item, source_language=input_lang_code, target_language=output_lang_code)
                translation = translation['translatedText'].encode('utf-8').decode('utf-8')
                translated_items.append(translation)

            # Update the JSON file with the translated prompts and items
            input_json[category]["prompts"] = translated_prompts
            input_json[category]["items"] = translated_items

    # Save the translated JSON file
    with open(output_path, "w") as f:
        json.dump(input_json, f, indent=4)



if __name__=='__main__':
    input_path = "social_groups\\english_data.json"
    output_path = "social_groups\\greek_data.json"
    input_lang = Language.English
    output_lang = Language.Greek
    ignored_words = ["<mask>", "<item>"]

    translate_json(input_path, output_path, input_lang, output_lang, ignored_words)
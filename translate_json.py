from google.cloud import translate_v2 as translate
import json
from typing import List
from data import Language


def translate_json(input_path: str, output_path: str, input_lang: Language, output_lang: Language, ignored_words: List[str]):
    # Set up the translation client
    translate_client = translate.Client()

    # Load the input JSON file
    with open(input_path, "r") as f:
        input_json = json.load(f)

    # Define a function to apply the ignored words filter to a string
    def ignore_words(text):
        for word in ignored_words:
            text = text.replace(word, f"<IGNORE_{word}>")
        return text

    # Define a function to remove the ignored words filter from a string
    def unignore_words(text):
        for word in ignored_words:
            text = text.replace(f"<IGNORE_{word}>", word)
        return text

    # Translate the prompts and items in the JSON file
    for category in input_json:
        prompts = input_json[category]["prompts"]
        items = input_json[category]["items"]

        # Translate the prompts
        translated_prompts = []
        for prompt in prompts:
            ignored_prompt = ignore_words(prompt)
            translation = translate_client.translate(ignored_prompt, source_language=input_lang.value, target_language=output_lang.value)
            translated_prompt = unignore_words(translation["translatedText"])
            translated_prompts.append(translated_prompt)

        # Translate the items
        translated_items = []
        for item in items:
            ignored_item = ignore_words(item)
            translation = translate_client.translate(ignored_item, source_language=input_lang.value, target_language=output_lang.value)
            translated_item = unignore_words(translation["translatedText"])
            translated_items.append(translated_item)

        # Update the JSON file with the translated prompts and items
        input_json[category]["prompts"] = translated_prompts
        input_json[category]["items"] = translated_items

    # Save the translated JSON file
    with open(output_path, "w") as f:
        json.dump(input_json, f, indent=4)


if __name__=='__main__':
    input_path = "data\english_data.json"
    output_path = "data\spanish_data.json"
    input_lang = Language.English
    output_lang = Language.Spanish
    ignored_words = ["<mask>", "<item>"]

    translate_json(input_path, output_path, input_lang, output_lang, ignored_words)
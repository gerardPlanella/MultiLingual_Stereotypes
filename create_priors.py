from model import load_model, Models
from data import Language
import argparse
import json
import os
from run_test import load_social_group_file, extract_prompts_groups
from collections import defaultdict
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException

def create_score_list(data, word_list, model):

    max_token = max(item['token'] for item in data)
    score_list = [None] * (max_token + 1)
    words_counted_in_file = 0
    words_not_in_file = 0
    for item in data:
        token_id = item['token']
        score = item['score']
        token = model.tokenizer.decode(token_id)
        if token in word_list:
            score_list[token_id] = score
            words_counted_in_file += 1
        else:
            score_list[token_id] = 100
            words_not_in_file += 1

    print(f'The words retrieved in the text file are {words_counted_in_file}')
    print(f'The words retrieved in the text file are {words_not_in_file}')
    return score_list

def create_score_list_for_greek(data, model):

    max_token = max(item['token'] for item in data)
    score_list = [None] * (max_token + 1)
    greek_words = 0 
    for item in data:
        token_id = item['token']
        score = item['score']
        token = model.tokenizer.decode(token_id)
        try:
            if detect(token) == 'el':
                score_list[token_id] = score
                greek_words += 1
            else:
                score_list[token_id] = 100
        except LangDetectException:
            pass
    print(f'The number of greek words is {greek_words}')
    return score_list


social_groups = ["religion", "age", "gender", "countries", "race", "profession", "political", "sexuality", "lifestyle"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Multilingual Model Prior probability creation.')
    parser.add_argument('--language_path', type=str, default="social_groups/greek_data.json", help="Language to analyse.")
    parser.add_argument('--output_dir', type=str, default="./prior_probs", help="Output directory for generated data.")
    parser.add_argument('--model_name', type=str, default="xlm-roberta-base", help="Model Evaluated")
    parser.add_argument('--model_top_k', type=int, default=250002, help="Top K results used for matrix generation, set this to the vocabulary size.")
    parser.add_argument('--verbose', action="store_false")

    args = parser.parse_args()
    verbose = args.verbose
    

    args.language = os.path.basename(args.language_path).split("_")[0]

    out_path = args.output_dir + "/" + args.language + "_priors.json"
    
    assert Language.has_value(args.language)

    args.language = Language(args.language)

    assert os.path.exists(args.language_path)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    assert Models.has_value(args.model_name)

    model = Models(args.model_name)
    model_attributes = None

    if model == Models.XLMR:
        model_attributes = {
            "pipeline":"fill-mask",
            "top_k":args.model_top_k
        }
    assert model_attributes is not None

    ok, language_data = load_social_group_file(args.language_path)

    assert ok

    if args.language == Language.English:
        with open(f'words_dictionnaries/en.txt', 'r') as f:
            words_in_file = set(line.strip() for line in f)
            print(len(words_in_file))

    elif args.language == Language.French:
        with open(f'words_dictionnaries/fr.txt', 'r') as f:
            words_in_file = set(line.strip() for line in f)
            print(len(words_in_file))

    if args.language == Language.Spanish:
        with open(f'words_dictionnaries/es.txt', 'r') as f:
            words_in_file = set(line.strip() for line in f)
            print(len(words_in_file))

    if args.language == Language.Croatian:
        with open(f'words_dictionnaries/cro.txt', 'r') as f:
            words_in_file = set(line.strip() for line in f)
            print(len(words_in_file))
    if verbose:
        print("Extracting Social Group data")

    prompts, _ = extract_prompts_groups(language_data, social_groups)

    priors = defaultdict(list)

    if verbose:
        print("Loading Model")

    unmasker = load_model(model, model_attributes)

    assert len(prompts) > 0

    unique_prompts = list(set(string for key in prompts for string in prompts[key]))

    for prompt in unique_prompts:
        if verbose:
            print("Analysing prompt: " + prompt)
        prompt_masked = prompt.replace("{}", "<mask>")
        out = unmasker(prompt_masked)[1]

        if args.language == Language.Greek:
            priors[prompt] = create_score_list_for_greek(out, unmasker)
        else:
            priors[prompt] = create_score_list(out, words_in_file, unmasker)

    if verbose:
        print("Saving to " + out_path)

    try:
         with open(out_path, 'w') as outfile:
            json.dump(priors, outfile)
    except TypeError:
        print("[ERROR] Unable to serialize the object")



    
    
 

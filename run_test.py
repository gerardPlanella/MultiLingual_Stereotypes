from model import load_model, Models
import json
import numpy as np 
from tqdm import tqdm
import pandas as pd
from scipy.stats import spearmanr
from data import Language, Emotions
from nltk.stem.snowball import SnowballStemmer
import argparse
import nltk
import json
import os
import pickle

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

social_groups = ["religion", "age", "gender", "countries", "race", "profession", "political", "sexuality", "lifestyle"]

def emotion_per_groups(prompts:dict, social_groups, 
                       language:Language, model_name:Models, 
                       model_attributes:dict, 
                       stemming = False, 
                       lex_path = "data/emolex.json", 
                       verbose = False):
    
    assert "general" in prompts
    
    stemmer = None
    if stemming:
        if language.value in SnowballStemmer.languages:
            stemmer = SnowballStemmer(language.value)
        else:
            raise Exception(f"No stemmer found for language {language.value}")
    #Load the LM 
    unmasker = load_model(model_name, model_attributes)

    #load emotion lexicon dictionnary
    with open(lex_path, "r", encoding="utf-8") as f:
        emolex = json.load(f)
    
    k = 0
    l = 0
    matrix_emotion = np.zeros((len(social_groups), len(emolex["sadly"])))
    for i,group in tqdm(enumerate(social_groups)):
        if group in prompts: #Local prompts
            prompt_list = prompts[group]
        else:
            prompt_list = prompts["general"]

        for j, prompt in enumerate(prompt_list):
            preds = unmasker(prompt.format(group))
            for pred in preds:
                if stemmer is not None:
                    word_pred = stemmer.stem(pred['token_str'])
                else:
                    word_pred = pred["token_str"]

                if word_pred in emolex:
                    matrix_emotion[i] += emolex[word_pred]
                    k += 1
                else:
                    l += 1

    if verbose:
        print(f"{l} words are not in the lexicon")
        print(f"{k} words are in the lexicon")

    column_labels = Emotions.to_list()
    df = pd.DataFrame(matrix_emotion, index=social_groups, columns=column_labels)
    if verbose:
        print(df)
    return matrix_emotion

def spearman_correlation(matrix_1:pd.DataFrame, matrix_2:pd.DataFrame):
    list_correlation = []
    for i in range(len(matrix_1)):
        list_correlation.append(spearmanr(matrix_1[i], matrix_2[i])[0])
    mean = np.mean(list_correlation)
    return list_correlation, mean

def load_social_group_file(path):
    try:
        with open(path) as f:
            data = json.load(f) 
        return True, data
    except Exception as e:
        print(e.message)
        return False, None
    
def check_n_prompts_groups(data1, data2, local_prompts:bool):
    ok = True
    if "general_prompts" not in data1 or "general_prompts" not in data2:
        return False
    
    for group_key in data1:
        if group_key == "general_prompts":
            if not isinstance(data1[group_key], list) or not isinstance(data2[group_key], list):
                ok = False
                break
            if len(data1[group_key]) != len(data2[group_key]) or len(data1[group_key]) == 0:
                ok = False
                break
            
            for prompt in data1[group_key] + data2[group_key]:
                if prompt.find("{}") == -1 or prompt.find("<mask>") == -1:
                    ok = False
                    return ok
                

        else:
            if "items" not in data1[group_key] or "items" not in data2[group_key]:
                ok = False
                break
            
            if len(data1[group_key]["items"]) != len(data2[group_key]["items"]):
                ok = False
                break
            
            if local_prompts:
                if "prompts" not in data1[group_key] or "prompts" not in data2[group_key]:
                    ok = False
                    break
                if len(data1[group_key]["prompts"]) != len(data2[group_key]["prompts"]):
                    ok = False
                    break
                for prompt in data1[group_key]["prompts"] + data2[group_key]["prompts"]:
                    if prompt.find("{}") == -1 or prompt.find("<mask>") == -1:
                        ok = False
                        return ok

    return ok


def extract_prompts_groups(data:dict, groups:list, local_prompts:bool):
    prompts = {}
    items = []


    for key in data:
        if key == "general_prompts":
            if "general" not in prompts:
                prompts["general"] = []
            prompts["general"] += data[key]
        else:
            if key in groups:
                if local_prompts:
                    if key not in prompts:
                        prompts[key] = []
                    prompts[key] += data[key]["prompts"]
                items += data[key]["items"]
                
    return prompts, items

                


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Multilingual Model Stereotype Analysis.')
    parser.add_argument('--social_groups', nargs='+', default=social_groups, help="Social Groups to Analyse.")
    parser.add_argument('--language_1_path', type=str, default="social_groups/french_data.json", help="Language 1 to analyse.")
    parser.add_argument('--language_2_path', type=str, default="social_groups/spanish_data.json", help="Language 2 to analyse.")
    parser.add_argument('--output_dir', type=str, default="out/", help="Output directory for generated data.")
    parser.add_argument('--stem_1', action="store_true", help="Apply stemming to Language 1.")
    parser.add_argument('--stem_2', action="store_true", help="Apply stemming to Language 2.")
    parser.add_argument('--use_local_prompts', action="store_true", help="If specified, will use social group specific prompts")
    parser.add_argument('--model_name', type=str, default="xlm-roberta-base", help="Model Evaluated")
    parser.add_argument('--model_top_k', type=int, default=10, help="Top K results used for matrix generation.")
    parser.add_argument('--lexicon_path', type=str, default="data/emolex.json", help="Path to Lexicon.")
    parser.add_argument('--verbose', action="store_true")
    parser.add_argument('--no_output_saving', action="store_false")

    args = parser.parse_args()

    args.language_1 = os.path.basename(args.language_1_path).split("_")[0]
    args.language_2 = os.path.basename(args.language_2_path).split("_")[0]
    
    assert Language.has_value(args.language_1)
    assert Language.has_value(args.language_2)

    args.language_1 = Language(args.language_1)
    args.language_2 = Language(args.language_2)

    assert os.path.exists(args.language_1_path)
    assert os.path.exists(args.language_2_path)

    if not os.path.exists(args.output_dir) and not args.no_output_saving:
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

    if args.verbose:
        print("Reading Social Group files")

    ok1, language_data_1 = load_social_group_file(args.language_1_path)
    ok2, language_data_2 = load_social_group_file(args.language_2_path)

    assert ok1 and ok2 

    if args.verbose:
        print("Checking File formats")

    file_formats_ok = check_n_prompts_groups(language_data_1, language_data_2, args.use_local_prompts)


    assert file_formats_ok

    if args.verbose:
        print("Extracting Social Group data")

    prompts_language_1, social_groups_language_1 = extract_prompts_groups(language_data_1, args.social_groups, args.use_local_prompts)
    prompts_language_2, social_groups_language_2 = extract_prompts_groups(language_data_2, args.social_groups, args.use_local_prompts)

    if args.verbose:
        print("Computing Matrix 1")
    
    matrix_1 = emotion_per_groups(prompts_language_1, social_groups_language_1, args.language_1,
                                  model, model_attributes,stemming = args.stem_1, 
                                  lex_path=args.lexicon_path, verbose=args.verbose)
    
    if args.verbose:
        print("Computing Matrix 2")
    
    matrix_2 = emotion_per_groups(prompts_language_2, social_groups_language_2, args.language_2,
                                  model, model_attributes,stemming = args.stem_2, 
                                  lex_path=args.lexicon_path, verbose=args.verbose)
    
    if args.verbose:
        print("Computing Correlation")

    coeffs = spearman_correlation(matrix_1, matrix_2)

    if args.verbose:
        print(f"----- Matrix for Language {args.language_1.name} --------")
        print(matrix_1)
        print(f"\n\n\n----- Matrix for Language {args.language_2.name} --------")
        print(matrix_2)
        print("\n\n\n------ Correlation Vector -------")
        print(coeffs[0])
        print("\n\n\n------ Mean of Correlation -------")
        print(coeffs[1])


    if not args.no_output_saving:
        if args.verbose:
            print("Saving Data...")

        matrix_1.to_pickle(args.output_dir + f"matrix_{args.language_1.name}_{args.stem_1}_{args.social_groups.join('_')}.pkl")
        matrix_2.to_pickle(args.output_dir + f"matrix_{args.language_2.name}_{args.stem_2}_{args.social_groups.join('_')}.pkl")
        with open(args.output_dir + f"correlation_{args.language_1.name}_{args.language_2.name}_{args.social_groups.join('_')}.pkl", 'wb') as f:
            pickle.dump(coeffs, f)
        
        if args.verbose:
            print("Data Saved.")
    
    
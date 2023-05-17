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
from snowballstemmer import stemmer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import XLMRobertaTokenizer
import torch 
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
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
    """
    Analyze the emotion content in a set of prompts for different social groups.

    This function uses a language model to predict words that fill in the blanks in prompts. 
    It then uses an emotion lexicon to determine the emotions associated with the predicted words.
    It returns a matrix and a DataFrame representing the emotional content associated with each social group.

    Parameters:
    prompts (dict): A dictionary containing prompts. The key "general" should be present.
    social_groups (list): A list of social groups to analyze.
    language (Language): A Language enum representing the language of the prompts and social groups.
    model_name (Models): A Models enum representing the language model to use for predictions.
    model_attributes (dict): A dictionary of attributes to pass to the language model.
    stemming (bool, optional): Whether to perform stemming on the predicted words. Default is False.
    lex_path (str, optional): The path to the emotion lexicon. Default is "data/emolex.json".
    verbose (bool, optional): Whether to print additional information during execution. Default is False.

    Returns:
    tuple: A tuple containing a numpy matrix and a pandas DataFrame. 
           Each row corresponds to a social group, and each column corresponds to an emotion from the emotion lexicon. 
           The values represent the emotional content associated with each social group.
    """
    
    assert "general" in prompts
    
    use_stemmer = None

    if language.value == "greek" and stemming==True:
        use_stemmer = stemmer("greek")

    if stemming and language.value != "greek":
        if language.value in SnowballStemmer.languages:
            use_stemmer = SnowballStemmer(language.value)
        else:
            raise Exception(f"No stemmer found for language {language.value}")
    #Load the LM 
    model = load_model(model_name, model_attributes, True)
    tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')

    #load the priors 
    with open(f'social_groups/prior_probs/{language.name.lower()}_priors.json') as file:
        priors = json.load(file)
    #load emotion lexicon dictionnary
    with open(lex_path, "r", encoding="utf-8") as f:
        emolex = json.load(f)
    
    k = 0
    l = 0
    matrix_emotion = np.zeros((len(social_groups), len(emolex["sadly"])))
    list_matrix_emotions = []
    list_dataframes = []
    for i,group in tqdm(enumerate(social_groups)):
        n_found_in_group = 0
        if group in prompts: #Local prompts
            prompt_list = prompts[group]
        else:
            prompt_list = prompts["general"]

        for j, prompt in enumerate(prompt_list):
            input_ids = tokenizer.encode(prompt.format(group), return_tensors='pt')
            # Get the position of the masked token
            mask_token_index = torch.where(input_ids == tokenizer.mask_token_id)[1]
            # mask_token_index = input_ids.index(tokenizer.mask_token_id)
            # Forward pass through the model
            outputs = model(input_ids)
            # Get the token probabilities, token_probs of size [1, 250002]
            token_probs = outputs.logits[0, mask_token_index, :].softmax(dim=1)
            # Get the scores for all words in the vocabulary
            all_scores = token_probs[0].tolist()

            results_log = [np.log(i) - np.log(j) for i, j in zip(all_scores, priors[prompt])]
            top_200_indices_log = np.argsort(results_log)[::-1]
            top_200_words_log = [tokenizer.decode(i) for i in top_200_indices_log]

            results_division = [i/j for i, j in zip(all_scores, priors[prompt])]
            top_200_indices_div = np.argsort(results_division)[::-1]
            top_200_words_div = [tokenizer.decode(i) for i in top_200_indices_div]

            results_division_square = [i/np.sqrt(j) for i, j in zip(all_scores, priors[prompt])]
            top_200_indices_div_square = np.argsort(results_division_square)[::-1]
            top_200_words_div_square = [tokenizer.decode(i) for i in top_200_indices_div_square]

            # top_200_english_words = []
            # g = 0
            # while len(top_200_english_words)<200:
            #     try:
            #         if detect(tokenizer.decode(top_200_indices[g])) == 'en':
            #             top_200_english_words.append(tokenizer.decode(top_200_indices[g]))
            #     except LangDetectException:
            #         pass
            #     g += 1

            for word in top_200_words:
                if word in emolex:
                    matrix_emotion[i] += emolex[word]
                    k += 1
                    n_found_in_group = 0
                else:
                    l += 1
                # list_matrix_emotions.append(matrix_emotion[i])
                # list_dataframes.append(pd.DataFrame(matrix_emotion[i], index=social_groups, columns=column_labels))
        matrix_emotion[i] /= n_found_in_group

    if verbose:
        print(f"{l} words are not in the lexicon")
        print(f"{k} words are in the lexicon")

    column_labels = Emotions.to_list()
    df = pd.DataFrame(matrix_emotion, index=social_groups, columns=column_labels)
    if verbose:
        print(df)
    return matrix_emotion, df

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
    """
    Checks the consistency and structure of two datasets with respect to general and local prompts.
    
    This function checks whether both datasets contain "general_prompts" and each group key has the same number of items. 
    It also verifies the existence and correct formatting of prompts.
    If the local_prompts parameter is set to True, the function also checks for the existence and consistency of local prompts.
    
    Parameters:
    data1 (dict): The first dataset to be checked. It's a dictionary where each key is a group and the values are items or prompts.
    data2 (dict): The second dataset to be checked. It should have the same structure as data1.
    local_prompts (bool): If set to True, the function checks for the presence and consistency of local prompts in both datasets.
    
    Returns:
    bool: Returns True if both datasets pass all checks, False otherwise.
    """
    
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


def similarity_matrix(matrix):
    """
    Calculate the cosine similarity between all pairs of vectors in the input matrix.
    
    Args:
        matrix (np.array): A 2D NumPy array or list of lists where each row represents a group's vector.
        
    Returns:
        np.array: A 2D NumPy array representing the similarity matrix with cosine similarity values.
    """
    # Convert input to a NumPy array if not already one
    matrix = np.array(matrix)

    # Get the number of groups
    number_groups = matrix.shape[0]

    # Initialize an empty similarity matrix
    similarity_matrix = np.zeros((number_groups, number_groups))

    # Compute cosine similarity for each pair of vectors and fill the similarity matrix
    for i in range(similarity_matrix.shape[0]):
        for j in range(similarity_matrix.shape[1]):
            similarity_matrix[i, j] = cosine_similarity([matrix[i, :]], [matrix[j, :]])

    return similarity_matrix


def extract_prompts_groups(data:dict, groups:list):
    prompts = {}
    items = {}


    for key in data:
        if key == "general_prompts":
            if "general" not in prompts:
                prompts["general"] = []
            prompts["general"] += data[key]
        else:
            if key in groups:
                if data[key]["prompts"]!= []:
                    if key not in prompts:
                        prompts[key] = []
                    prompts[key] += data[key]["prompts"]
                items[key] = []
                items[key] += data[key]["items"]
                
    return prompts, items

def run_all_groups(social_groups, language_1_path, language_2_path, model, model_attributes, stemming_l1, stemming_l2, lex_path, verbose, output_dir, use_local_prompts = True):

    ok1, language_data_1 = load_social_group_file(language_1_path)
    ok2, language_data_2 = load_social_group_file(language_2_path)

    language_1 = os.path.basename(language_1_path).split("_")[0]
    language_2 = os.path.basename(language_2_path).split("_")[0]

    assert ok1 and ok2 

    language_1 = Language(args.language_1)
    language_2 = Language(args.language_2)

    if verbose:
        print("Checking File formats")

    file_formats_ok = check_n_prompts_groups(language_data_1, language_data_2, use_local_prompts)

    if verbose:
        print("Extracting Social Group data")

    prompts_language_1, social_groups_language_1 = extract_prompts_groups(language_data_1, social_groups)
    prompts_language_2, social_groups_language_2 = extract_prompts_groups(language_data_2, social_groups)



    list_matrix_l1 = []
    list_df_l1 = []

    list_matrix_l2 = []
    list_df_l2 = []

    coeff_emotions = []
    coeff_emotions_RSA = []
    if verbose:
        print("Computing Matrices")

    for i in tqdm(range(len(social_groups))):
        emotions_extract_1 = emotion_per_groups(prompts_language_1, social_groups_language_1[social_groups[i]], language_1,
                                  model, model_attributes, stemming_l1, 
                                  lex_path, verbose)

        emotions_extract_2 = emotion_per_groups(prompts_language_2, social_groups_language_2[social_groups[i]], language_2,
                                  model, model_attributes, stemming_l2, 
                                  lex_path, verbose)
        

        list_matrix_l1.append(emotions_extract_1[0])
        list_df_l1.append(emotions_extract_1[1])

        list_matrix_l2.append(emotions_extract_2[0])
        list_df_l2.append(emotions_extract_2[1])

        emotions_extract_1[1].to_csv(f"{output_dir}/matrix_{language_1.name}_{stemming_l1}_{social_groups[i]}.csv", index = True)
        emotions_extract_2[1].to_csv(f"{output_dir}/matrix_{language_2.name}_{stemming_l2}_{social_groups[i]}.csv", index = True)
        
    
        coeff_emotions.append(spearman_correlation(emotions_extract_1[0], emotions_extract_2[0]))
        coeff_emotions_RSA.append(spearman_correlation(similarity_matrix(emotions_extract_1[0]), similarity_matrix(emotions_extract_2[0])))

    if verbose:
        print("Saving Data...")

    with open(output_dir + f"/correlation_{language_1.name}_{language_2.name}_{social_groups}.json", 'w') as f:
        json.dump(coeff_emotions, f)
    with open(output_dir + f"/correlation_RSA_{language_1.name}_{language_2.name}_{social_groups}.json", 'w') as f:
        json.dump(coeff_emotions_RSA, f)
    
    if verbose:
        print("Data Saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Multilingual Model Stereotype Analysis.')
    parser.add_argument('--social_groups', nargs='+', default=['age', 'lifestyle'], help="Social Groups to Analyse.")
    parser.add_argument('--language_1_path', type=str, default="social_groups/english_data.json", help="Language 1 to analyse.")
    parser.add_argument('--language_2_path', type=str, default="social_groups/english_data.json", help="Language 2 to analyse.")
    parser.add_argument('--output_dir', type=str, default="out/test", help="Output directory for generated data.")
    parser.add_argument('--stem_1', action="store_true", help="Apply stemming to Language 1.")
    parser.add_argument('--stem_2', action="store_true", help="Apply stemming to Language 2.")
    parser.add_argument('--use_local_prompts', action="store_true", help="If specified, will use social group specific prompts")
    parser.add_argument('--model_name', type=str, default="xlm-roberta-base", help="Model Evaluated")
    parser.add_argument('--model_top_k', type=int, default=20, help="Top K results used for matrix generation.")
    parser.add_argument('--lexicon_path_1', type=str, default="data/emolex_all_nostemmed.json", help="Path to Lexicon.")
    parser.add_argument('--lexicon_path_2', type=str, default="data/emolex_all_nostemmed.json", help="Path to Lexicon.")
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

    run_all_groups(args.social_groups, args.language_1_path, args.language_2_path, model, model_attributes, args.stem_1, args.stem_2, args.lexicon_path_1, args.verbose, args.output_dir)



    # if args.verbose:
    #     print("Reading Social Group files")

    # ok1, language_data_1 = load_social_group_file(args.language_1_path)
    # ok2, language_data_2 = load_social_group_file(args.language_2_path)

    # assert ok1 and ok2 

    # if args.verbose:
    #     print("Checking File formats")

    # file_formats_ok = check_n_prompts_groups(language_data_1, language_data_2, args.use_local_prompts)


    # assert file_formats_ok

    # if args.verbose:
    #     print("Extracting Social Group data")

    # prompts_language_1, social_groups_language_1 = extract_prompts_groups(language_data_1, args.social_groups, args.use_local_prompts)
    # prompts_language_2, social_groups_language_2 = extract_prompts_groups(language_data_2, args.social_groups, args.use_local_prompts)

    # if args.verbose:
    #     print("Computing Matrix 1")
    
    # matrix_1, df1 = emotion_per_groups(prompts_language_1, social_groups_language_1, args.language_1,
    #                               model, model_attributes,stemming = args.stem_1, 
    #                               lex_path=args.lexicon_path, verbose=args.verbose)
    
    # if args.verbose:
    #     print("Computing Matrix 2")

    
    
    # matrix_2, df2 = emotion_per_groups(prompts_language_2, social_groups_language_2, args.language_2,
    #                               model, model_attributes,stemming = args.stem_2, 
    #                               lex_path=args.lexicon_path, verbose=args.verbose)


    # sim_matrix_1 = similarity_matrix(matrix_1)
    # sim_matrix_2 = similarity_matrix(matrix_2)

    # if args.verbose:
    #     print("Computing Correlation")

    # coeffs_emotion = spearman_correlation(matrix_1, matrix_2)
    # coeffs_sim = spearman_correlation(sim_matrix_1, sim_matrix_2)

    # if args.verbose:
    #     print(f"----- Matrix for Language {args.language_1.name} --------")
    #     print(df1)
    #     print(f"\n\n\n----- Matrix for Language {args.language_2.name} --------")
    #     print(df2)
    #     print("\n\n\n------ Correlation Vector -------")
    #     print(coeffs_emotion[0])
    #     print("\n\n\n------ Mean of Correlation -------")
    #     print(coeffs_emotion[1])
    #     print("\n\n\n------ Correlation Vector RSA -------")
    #     print(coeffs_sim[0])
    #     print("\n\n\n------ Mean of Correlation RSA -------")
    #     print(coeffs_sim[1])


    # if not args.no_output_saving:
    #     if args.verbose:
    #         print("Saving Data...")

    #     df1.to_csv(f"{args.output_dir}/matrix_{args.language_1.name}_{args.stem_1}_{args.social_groups}.csv", index = False)
    #     df2.to_csv(f"{args.output_dir}/matrix_{args.language_2.name}_{args.stem_2}_{args.social_groups}.csv", index = False)
    #     with open(args.output_dir + f"correlation_{args.language_1.name}_{args.language_2.name}_{args.social_groups}.pkl", 'wb') as f:
    #         pickle.dump(coeffs_emotion, f)
    #     with open(args.output_dir + f"correlation_RSA_{args.language_1.name}_{args.language_2.name}_{args.social_groups}.pkl", 'wb') as f:
    #         pickle.dump(coeffs_sim, f)
        
    #     if args.verbose:
    #         print("Data Saved.")
    
    
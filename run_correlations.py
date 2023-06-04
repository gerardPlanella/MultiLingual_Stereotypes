from model import load_model, Models
import json
import numpy as np 
from tqdm import tqdm
import pandas as pd
from data import Language, Emotions
from run_test_normalization import spearman_correlation, similarity_matrix
import argparse
import nltk
import json
import os
import pickle

social_groups = ["religion", "age", "gender", "countries", "race", "profession", "political", "sexuality", "lifestyle", "racial_minorities"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Multilingual Model Stereotype Analysis.')
    parser.add_argument('--social_groups', nargs='+', default=social_groups, help="Social Groups to Analyse.")
    parser.add_argument('--language_1_path', type=str, default="social_groups/english_data.json", help="Language 1 to analyse.")
    parser.add_argument('--language_2_path', type=str, default="social_groups/croatian_data.json", help="Language 2 to analyse.")   
    parser.add_argument('--output_dir', type=str, default="out/french_finetune", help="Output directory for generated data.")
    parser.add_argument('--model_name', type=str, default="xlm-roberta-base", help="Model Evaluated")
    parser.add_argument('--model_top_k', type=int, default=300, help="Top K results used for matrix generation.")   
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

    if model == Models.BERT:
        model_attributes = {
            "pipeline":"fill-mask",
            "top_k":args.model_top_k
        }
    
    assert model_attributes is not None    

    language = ['english', 'french', 'croatian', 'greek', 'spanish']

    for lang in language:
        for group in social_groups:
            df_1 = pd.read_csv(f'out/pretrained_roberta/emotion_profiles/{lang}/{group}.csv').values[:,1:]
            df_2 = pd.read_csv(f'{args.output_dir}/emotion_profiles/{lang}/{group}.csv').values[:,1:]
            
            if os.path.exists(f"{args.output_dir}/spearman_correlations_RSA") == False:
                os.mkdir(f"{args.output_dir}/spearman_correlations_RSA")

            if os.path.exists(f"{args.output_dir}/spearman_correlations_RSA/{lang}_{lang}/") == False:
                os.mkdir(f"{args.output_dir}/spearman_correlations_RSA/{lang}_{lang}/")
            
            with open(f'{args.output_dir}/spearman_correlations_RSA' + f"/{lang}_{lang}/{group}.json", 'w') as f:
                json.dump(spearman_correlation(similarity_matrix(df_1), similarity_matrix(df_2)), f)
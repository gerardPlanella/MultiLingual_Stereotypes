import torch
from transformers import XLMRobertaForMaskedLM, XLMRobertaTokenizer, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from datasets import load_dataset, Dataset
from data import preprocessing_fine_tuning
from model import load_model, Models
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW
import argparse
import pandas as pd
import sys
import csv
import math

import logging
logging.basicConfig(level=logging.INFO)

# csv.field_size_limit(sys.maxsize)

def log_loss_callback(eval_args, metrics, **kwargs):
    if eval_args.step % 10 == 0:
        print(f"Step: {eval_args.step}, Loss: {metrics['loss']:.4f}")

def tokenize_function(examples, max_seq_length):
    tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
    return tokenizer(examples, return_special_tokens_mask=True, padding='max_length', truncation=True, max_length=max_seq_length)

def truncate_articles(articles):
    truncated_articles = []
    for article in articles:
        words = article.split(' ')
        if len(words) > 293:
            words = words[:293]
        truncated_article = ' '.join(words)
        truncated_articles.append(truncated_article)
    return truncated_articles

class TextDataset(Dataset):
    def __init__(self, text):
        self.text = text
        self.tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
        # Compute the maximum length
        self.max_length = 512

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        sentence = self.text[idx]
        encoded = self.tokenizer.encode_plus(sentence, add_special_tokens=True, 
                                             padding='max_length', truncation=True, max_length=self.max_length, 
                                             return_tensors='pt')
        return {'input_ids': encoded['input_ids'].flatten(), 
                'attention_mask': encoded['attention_mask'].flatten()}
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Multilingual Model Stereotype Analysis.')
    parser.add_argument('--output_directory', type=str, default="./xlm-roberta-finetuned/french_fine_tuning_2", help="Output directory for trained model.")
    parser.add_argument('--model_name', type=str, default="xlm-roberta-base", help="Model to be fine-tuned")
    parser.add_argument('--dataset_name', type=str, default="wikitext", help="Dataset name")
    parser.add_argument('--dataset_version', type=str, default="wikitext-103-raw-v1", help="Dataset version")
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--news_source', type=str, default="Fox News")
    parser.add_argument('--verbose', action="store_true")
    parser.add_argument('--no_output_saving', action="store_false")

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_attributes = { 
        "pipeline":"fill-mask", 
        "top_k":200
    }

    model = Models(args.model_name)

    model = load_model(model, model_attributes, pre_trained = True)
    model = model.to(device)
    tsv_file = "data/offenseval-gr-training-v1.tsv"
    df = pd.read_csv(tsv_file, sep='\t', names=['id', 'tweet', 'subtask_a'])

    df = df.iloc[:5000]
    df_list = df['tweet'].tolist()[1:]


    dataset = TextDataset(df_list)
    dataloader = DataLoader(dataset, batch_size=8)

    optimizer = AdamW(model.parameters(), lr=1e-5)

    model.train()
    for epoch in range(args.epochs):  # Number of training epochs
        for i, batch in enumerate(dataloader):
            # Get the inputs and move them to the GPU
            inputs = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # Forward pass and calculate the loss
            outputs = model(inputs, attention_mask=attention_mask, labels=inputs)
            loss = outputs.loss

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            optimizer.zero_grad() 

            # Print loss every 50 batches
            if (i+1) % 50 == 0:
                print(f'Epoch [{epoch+1}/{args.epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item()}')


        print(f"Epoch {epoch + 1} Loss: {loss.item()}")
        model.save_pretrained(f"{args.output_directory}/checkpoint_epoch_{epoch + 1}")
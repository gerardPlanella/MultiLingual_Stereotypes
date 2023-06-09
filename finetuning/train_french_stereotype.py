import torch
from transformers import XLMRobertaForMaskedLM, XLMRobertaTokenizer, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from datasets import load_dataset, Dataset
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

class TextDataset(Dataset):
    def __init__(self, text):
        self.text = text
        self.tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
        # Compute the maximum length
        self.max_length = max(len(self.tokenizer.encode(t)) for t in text)

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
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--no_output_saving', action="store_false")

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Read the CSV file into a DataFrame
    df = pd.read_csv('data/crows_pairs_FR.csv', sep='\t', header=None, names=['Paragraph1', 'Paragraph2', 'Label', 'Bias_Type'])

    model_attributes = { 
        "pipeline":"fill-mask", 
        "top_k":200
    }

    model = Models(args.model_name)

    model = load_model(model, model_attributes, 'base')
    model = model.to(device)
    csv_file = "data/crows_pairs_FR.csv"
    df = pd.read_csv(csv_file, sep='\t', header=None, names=['Paragraph_1', 'Paragraph_2', 'Label', 'Bias_Type'])

    df = df[df['Label'] == 'stereo']
    df_list = df['Paragraph_1'].tolist()

    dataset = TextDataset(df_list)
    dataloader = DataLoader(dataset, batch_size=8)

    optimizer = AdamW(model.parameters(), lr=1e-5)

    model.train()
    for epoch in range(args.epochs):  # Number of training epochs
        for batch in dataloader:
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

        print(f"Epoch {epoch + 1} Loss: {loss.item()}")
        model.save_pretrained(f"{args.output_directory}/checkpoint_epoch_{epoch + 1}")
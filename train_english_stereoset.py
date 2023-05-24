import torch
from transformers import XLMRobertaForMaskedLM, XLMRobertaTokenizer, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from datasets import load_dataset, Dataset
# from data import preprocessing_fine_tuning
from model import load_model, Models
import argparse
import pandas as pd
import sys
import csv
import math

import logging
logging.basicConfig(level=logging.INFO)

# csv.field_size_limit(sys.maxsize//4)

def log_loss_callback(eval_args, metrics, **kwargs):
    if eval_args.step % 10 == 0:
        print(f"Step: {eval_args.step}, Loss: {metrics['loss']:.4f}")

def tokenize_function(examples, max_length):
    tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
    return tokenizer(examples, return_special_tokens_mask=True, padding='max_length', truncation=True, max_length=max_length)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Multilingual Model Stereotype Analysis.')
    parser.add_argument('--output_directory', type=str, default="./xlm-roberta-finetuned/stereoset", help="Output directory for trained model.")
    parser.add_argument('--model_name', type=str, default="xlm-roberta-base", help="Model to be fine-tuned")
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=8)

    args = parser.parse_args()

    model_attributes = { 
        "pipeline":"fill-mask", 
        "top_k":200
    }
    model = Models(args.model_name)

    model = load_model(model, model_attributes, 'base')

    tokenizer = XLMRobertaTokenizer.from_pretrained(args.model_name)

    dataset = load_dataset('stereoset', 'intrasentence')

    # Filter the dataset
    def is_race_bias(instance):
        return instance['bias_type'] == 'race' and 1 in instance['sentences']['gold_label']

    race_bias_dataset = dataset['validation'].filter(is_race_bias)

    # Extract the target sentences
    def get_target_sentence(instance):
        gold_labels = instance['sentences']['gold_label']
        stereotype_index = gold_labels.index(1)
        return {'text': instance['sentences']['sentence'][stereotype_index]}

    race_bias_sentences = race_bias_dataset.map(get_target_sentence, remove_columns=['id', 'target', 'bias_type', 'context', 'sentences'])

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

    list_sentences = []
    for instance in race_bias_sentences:
        list_sentences.append(instance['text'])

    max_length = max(len(s.split()) for s in list_sentences)
    tokenized_sentences = tokenize_function(list_sentences, max_length)
    dataset = Dataset.from_dict(tokenized_sentences)
    # Calculate the number of training steps per epoch

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)


    print("Training")
    training_args = TrainingArguments(
        output_dir=args.output_directory,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        logging_steps=10,  # logs loss and other metrics every 100 steps
        logging_dir='./logs',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset
                )

    trainer.train()

    print("Saving models")
    model.save_pretrained(args.output_directory)
    tokenizer.save_pretrained(args.output_directory)




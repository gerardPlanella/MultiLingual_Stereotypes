import torch
from transformers import XLMRobertaForMaskedLM, XLMRobertaTokenizer, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from datasets import load_dataset, Dataset
from data import tokenize_function, preprocessing_fine_tuning
from model import load_model, Models
import argparse
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Multilingual Model Stereotype Analysis.')
    parser.add_argument('--output_dir', type=str, default="./xlm-roberta-finetuned", help="Output directory for trained model.")
    parser.add_argument('--model_name', type=str, default="xlm-roberta-base", help="Model to be fine-tuned")
    parser.add_argument('--dataset_name', type=str, default="wikitext", help="Dataset name")
    parser.add_argument('--dataset_version', type=str, default="wikitext-103-raw-v1", help="Dataset version")
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=8)

    parser.add_argument('--verbose', action="store_true")
    parser.add_argument('--no_output_saving', action="store_false")

    args = parser.parse_args()

    model_attributes = { 
        "pipeline":"fill-mask", 
        "top_k":200
    }
    model = Models(args.model_name)

    model = load_model(model, model_attributes, pre_trained = True)
    # Freeze all layers
    # for param in model.parameters():
    #     param.requires_grad = False

    # # Unfreeze the dense layer of the LM head (which is the intermediate layer before the output layer)
    # for param in model.lm_head.dense.parameters():
    #     param.requires_grad = True

    tokenizer = XLMRobertaTokenizer.from_pretrained(args.model_name)
    csv_file = "path/to/all-the-news.csv"
    df = pd.read_csv(csv_file)
    fox_news_df = df[df['publication'] == 'Fox News'].sample(n=5000, random_state=42)
    fox_news_articles = fox_news_df['content'].tolist()

    tokenized_articles = tokenize_function(fox_news_articles)
    fox_news_dataset = Dataset.from_dict(tokenized_articles)

    # data_collator, tokenized_dataset = preprocessing_fine_tuning(args.dataset_name, args.dataset_version, tokenizer)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)


    # Calculate the number of training steps per epoch
    num_train_samples = len(fox_news_dataset["train"])
    steps_per_epoch = num_train_samples // args.batch_size

    print("Training")
    training_args = TrainingArguments(
    output_dir=args.output_dir,
    output_dir=args.output_dir,
    overwrite_output_dir=True,
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch_size,
    save_steps=steps_per_epoch,
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch_size,
    save_steps=steps_per_epoch,
    )   

    trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=fox_news_dataset,
    )

    trainer.train()

    print("Saving models")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)




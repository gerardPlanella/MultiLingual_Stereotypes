import torch
from transformers import XLMRobertaForMaskedLM, XLMRobertaTokenizer, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from datasets import load_dataset

def tokenize_function(examples):
    tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
    return tokenizer(examples["text"], return_special_tokens_mask=True)


if __name__ == "__main__":
    model = XLMRobertaForMaskedLM.from_pretrained("xlm-roberta-base")
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the dense layer of the LM head (which is the intermediate layer before the output layer)
    for param in model.lm_head.dense.parameters():
        param.requires_grad = True

    for param in model.parameters():
        if param.requires_grad == True:
            print(param)


    tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
    print("Loading dataset.")
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split = ["train[:5%]", "test[:1%]"])
    tokenized_dataset = dataset[1].map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])
    print(len(tokenized_dataset))

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

    print("Training")
    training_args = TrainingArguments(
    output_dir="./xlm-roberta-finetuned",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=8,
    save_steps=10_000,
    save_total_limit=2,
    )   

    trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset,
    )

    trainer.train()

    print("Saving models")
    model.save_pretrained("./xlm-roberta-finetuned")
    tokenizer.save_pretrained("./xlm-roberta-finetuned")




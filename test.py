from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset

model = AutoModelForSequenceClassification.from_pretrained("PORTULAN/albertina-ptbr", num_labels=2)
tokenizer = AutoTokenizer.from_pretrained("PORTULAN/albertina-ptbr")
dataset = load_dataset("PORTULAN/glue-ptpt", "rte")

def tokenize_function(examples):
    return tokenizer(examples["sentence1"], examples["sentence2"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

training_args = TrainingArguments(output_dir="albertina-ptbr-rte", evaluation_strategy="epoch")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
)

trainer.train()
from datasets import Dataset, DatasetDict, Value, ClassLabel, Features
import evaluate
import numpy as np
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer


def load_dfs(data_dir="data"):
    """Load data from data directory"""
    dfs = {}
    for split in ['train', 'dev', 'test']:
        dfs[split] = pd.read_csv(f"{data_dir}/{split}.tsv", delimiter='\t', header=None, names=['table_id', 'nr_columns', 'columns', 'table_text', 'statement', 'label'])
    return dfs


def concat_table_with_statement(dfs):
    """Concatenate table text with statement"""
    for split in ['train', 'dev', 'test']:
        # Cannot use f"" string since we are dealing with a pandas dataframe.
        dfs[split]['table_w_statement'] = dfs[split]['table_text'] + " [SEP] " + dfs[split]['statement']
    return dfs


def drop_na(dfs):
    """Drop rows with missing values"""
    for split in ['train', 'dev', 'test']:
        cnt_before = len(dfs[split])
        dfs[split].dropna(inplace=True)
        print(f"{split} - Dropping missing values: {cnt_before} >> {len(dfs[split])}")
    return dfs


def to_dataset(dfs):
    """Convert pandas dataframe to dataset"""
    dataset = DatasetDict()
    for split in ['train', 'dev', 'test']:
        df = dfs[split].drop(columns=['table_id', 'nr_columns', 'columns'])
        dataset[split] = Dataset.from_pandas(df)
        dataset[split] = dataset[split].class_encode_column("label")
    return dataset


def tokenize_function(examples):
    return tokenizer(examples["statement"], examples["table_text"], padding="max_length", truncation=True)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


dfs = load_dfs()
# dfs = concat_table_with_statement(dfs)
dfs = drop_na(dfs)
dataset = to_dataset(dfs)

model_name = "bert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

seed_val = 0
train_dataset = tokenized_dataset["train"].shuffle(seed=seed_val)
dev_dataset = tokenized_dataset["dev"].shuffle(seed=seed_val)

metric = evaluate.load("accuracy")  

if __name__ == "__main__":
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    training_args = TrainingArguments(
        output_dir="bert_table", 
        evaluation_strategy="epoch",
        weight_decay=0.01,
        warmup_ratio=0.1)
    print(training_args.device)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    test_results = trainer.predict(tokenized_dataset["test"])
    print(test_results.metrics)

    print()
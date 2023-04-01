from datasets import Dataset, DatasetDict
import evaluate
import numpy as np
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer

from train import compute_metrics, tokenized_dataset


model_path = "bert_table-3/checkpoint-16500"

model = AutoModelForSequenceClassification.from_pretrained(model_path)

test_args = TrainingArguments(output_dir="testing", evaluation_strategy="epoch")
print(test_args.device)

metric = evaluate.load("accuracy")  

trainer = Trainer(model = model, 
    args = test_args, 
    compute_metrics = compute_metrics
)

test_results = trainer.predict(tokenized_dataset["test"])
print(test_results.metrics)

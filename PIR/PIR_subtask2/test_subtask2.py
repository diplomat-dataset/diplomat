from datasets import load_from_disk, DatasetDict, Dataset
import torch
from transformers import set_seed
import numpy as np
import argparse
import wandb
parser = argparse.ArgumentParser()
parser.add_argument("--model_checkpoint", type=str, default="bert-base-uncased")
parser.add_argument("--seed",type=int,default=42)
parser.add_argument("--batch_size",type=int,default=16)
parser.add_argument("--learning_rate",type=float,default=5e-5)
parser.add_argument("--num_train_epochs",type=int,default=3)
parser.add_argument("--weight_decay",type=float,default=0.01)


args = parser.parse_args()
seed = args.seed
model_checkpoint = args.model_checkpoint
batch_size = args.batch_size
learning_rate = args.learning_rate
num_train_epochs = args.num_train_epochs
weight_decay = args.weight_decay

wandb.init(project=f"Subtask2",name=f"{model_checkpoint}_{seed}")
wandb.config.update(args)

import random
from transformers import TrainingArguments, Trainer
import evaluate

def setup_seed(seed):
    '''set seed for reproducibility'''
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    set_seed(seed)
setup_seed(seed)
from datasets import load_from_disk
dataset = load_from_disk("final_with_split_IDAR_second_subtask_dataset")

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
tokenizer.model_max_length = 512



def preprocess_function(examples):
    '''tokenize function'''
    context = [[" ".join(examples['text'])]*5]
    choice = examples['choice']

    choice = [
        [c] for c in choice
    ]

    context = sum(context, [])
    choice = sum(choice, [])

    tokenized_examples = tokenizer(context, choice, truncation=True)
    return tokenized_examples
def gpt_preprocess_function(examples):
    '''tokenize function of gpt model'''
    context = " ".join(examples['text'])
    choice = examples['choice']
    choice = " ".join(examples['choice'])
    tokenized_examples = tokenizer(context, choice, truncation=True)
    return tokenized_examples
if 'gpt' in model_checkpoint:
    tokenizer.pad_token = tokenizer.eos_token
    tokenized_dataset = dataset.map(gpt_preprocess_function)
else:
    tokenized_dataset = dataset.map(preprocess_function)

from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from typing import Optional, Union
import torch


@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]

        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        #print(flattened_features)
        flattened_features = sum(flattened_features, [])

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
            )

        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch
import evaluate

accuracy = evaluate.load("accuracy")
import numpy as np


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)
from transformers import AutoModelForMultipleChoice, TrainingArguments, Trainer, AutoModelForSequenceClassification, DataCollatorWithPadding

model_name = model_checkpoint.split("/")[-1]
if 'gpt' in model_checkpoint:
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)
    model.config.pad_token_id = model.config.eos_token_id
else:
    model = AutoModelForMultipleChoice.from_pretrained(model_checkpoint)
if 'gpt' in model_checkpoint:
    data_collator = DataCollatorWithPadding(tokenizer)
else:
    data_collator = DataCollatorForMultipleChoice(tokenizer=tokenizer)

training_args = TrainingArguments(
    output_dir=f"/scratch/nlp/lihengli/multiple/subtask2/{model_name}/{seed}/",
    evaluation_strategy="steps",
    save_strategy="steps",
    load_best_model_at_end=True,
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_train_epochs,
    weight_decay=weight_decay,
    report_to="wandb",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer),
    compute_metrics=compute_metrics,
)
print("-"*20)

print(f"/scratch/nlp/lihengli/multiple/subtask2/{model_name}/{seed}/checkpoint-19000")
print("-"*20)
trainer.train(f"/scratch/nlp/lihengli/multiple/subtask2/{model_name}/{seed}/checkpoint-18000")
#trainer.train()
result = trainer.predict(tokenized_dataset["test"])
print(result.metrics)
wandb.log(result.metrics)

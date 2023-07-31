"""
Fine-tuning the library's seq2seq models for question answering using the 🤗 Seq2SeqTrainer.
"""

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import torch
import numpy as np
import random

import datasets
import evaluate
import numpy as np
from datasets import load_dataset, Dataset, DatasetDict, load_from_disk
import wandb

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    set_seed,

)


        
   

def flatten(dataset):
    '''
    1. flatten dataset
    2. each question one data

    '''
    text_list = []
    speaker_list = []
    gold_statement_list = []
    questions_list = []
    answer_list = []
    for data in dataset:
        text = data['text']
        speaker = data['speaker']
        gold_statement = data['gold_statement']
        questions = data['questions']
        answer = data['answer']
        length = len(questions)
        num_gold_statement = length / 3
        for idx in range(length):
            # each question one data
            text_list.append(text)
            speaker_list.append(speaker)
            if idx % num_gold_statement == 0:
                gold_statement_list.append(gold_statement[0])
            elif idx % num_gold_statement == 1:
                gold_statement_list.append(gold_statement[1])
            elif idx % num_gold_statement == 2:
                gold_statement_list.append(gold_statement[2])
            elif idx % num_gold_statement == 3:
                gold_statement_list.append(gold_statement[3])
            elif idx % num_gold_statement == 4:
                gold_statement_list.append(gold_statement[4])
            elif idx % num_gold_statement == 5:
                gold_statement_list.append(gold_statement[5])
            elif idx % num_gold_statement == 6:
                gold_statement_list.append(gold_statement[6])
            elif idx % num_gold_statement == 7:
                gold_statement_list.append(gold_statement[7])
            questions_list.append(questions[idx])
            answer_list.append(answer[idx])
    dataset = Dataset.from_dict({'text':text_list,'speaker':speaker_list,
                                 'gold_statement':gold_statement_list,
                                 'questions':questions_list,
                                 'answer':answer_list})
    return dataset



# See all possible arguments in src/transformers/training_args.py
# or by passing the --help flag to this script.
# We now keep distinct sets of args, for a cleaner separation of concerns.
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model_checkpoint', type=str, default='t5-small')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--max_seq_length', type=int, default=512)
parser.add_argument('--max_answer_length', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=24)
parser.add_argument('--num_train_epochs', type=int, default=50)
parser.add_argument('--lr',type=float, default=5.6e-5)
parser.add_argument('--weight_decay',type=float, default=0.001)
args = parser.parse_args()
seed = args.seed
model_checkpoint = args.model_checkpoint
max_seq_length = args.max_seq_length
max_answer_length = args.max_answer_length
lr = args.lr
weight_decay = args.weight_decay
batch_size = args.batch_size
num_train_epochs = args.num_train_epochs
model_name = model_checkpoint.split("/")[-1]

wandb.init(project="with_gold_seq2seqqa", config = {"model name":model_checkpoint, "seed":seed, "batch_size":batch_size, "train epochs":num_train_epochs, "weight decay":weight_decay, "lr":lr},
           name = f"{model_name}_seed_{seed}", resume=True)




def setup_seed(seed):
    '''set seed for reproducibility'''
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(seed)


# Set seed before initializing model.
set_seed(seed)
dataset = load_from_disk('../dataset/CQA_task_dataset')



# See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
# https://huggingface.co/docs/datasets/loading_datasets.html.

# Load pretrained model and tokenizer
#
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

# We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
# on a small vocab and want a smaller embedding size, remove this test.
embedding_size = model.get_input_embeddings().weight.shape[0]
if len(tokenizer) > embedding_size:
    model.resize_token_embeddings(len(tokenizer))

if model.config.decoder_start_token_id is None:
    raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")



# Temporarily set max_answer_length for training.
max_answer_length = max_answer_length

max_seq_length = min(max_seq_length, tokenizer.model_max_length)
print(max_seq_length)

def preprocess_qa(example,):
    ''' build input and target for training'''
    question = example['questions']
    speaker = example['speaker']
    text = example['text']
    answer = example['answer']
    gold_statement = example['gold_statement']

    inputs = "context: "
    for idx in range(len(text)):
        inputs += speaker[idx].strip()
        inputs += ": "
        inputs += text[idx].strip()
        inputs += " "
    inputs += "Implicature: "
    inputs += gold_statement.strip()
    inputs += " "
    inputs += "question: "
    inputs += question.strip()
    targets = answer.lower().strip()
    return inputs, targets

    

def preprocess_function(examples):
    ''' tokenization function'''
    inputs, targets = preprocess_qa(examples)

    model_inputs = tokenizer(inputs, max_length=max_seq_length, padding=True, truncation=True)
    labels = tokenizer(text_target=targets, max_length=5, padding=True, truncation=True)

        

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function)


# Data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
)
from transformers import Seq2SeqTrainingArguments

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    count = 0
    length  = len(decoded_preds)
    if len(decoded_labels) != length:
        print("decode label and pred not equal length")
    for i in range(length):
        if decoded_preds[i].lower().strip() == decoded_labels[i].lower().strip():
            count += 1
    return {"accuracy score": count/length, "correct number":count, "length":length}



args = Seq2SeqTrainingArguments(
    output_dir=f"./test_result/with_gold/{model_name}/{seed}/{model_name}-finetuned-seq2seq",
    evaluation_strategy="steps",
    save_strategy="steps",
    load_best_model_at_end=True,
    learning_rate=lr,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=weight_decay,
    num_train_epochs=num_train_epochs,
    predict_with_generate=True,
    report_to = "wandb",
)

from transformers import DataCollatorForSeq2Seq

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
tokenized_dataset = tokenized_dataset.remove_columns(
        dataset['train'].column_names
)



# Initialize our Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=tokenized_dataset['train'] ,
    eval_dataset=tokenized_dataset['validation'] ,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Training
trainer.train()
result = trainer.predict(tokenized_dataset['test'])
wandb.log(result.metrics)
wandb.finish()


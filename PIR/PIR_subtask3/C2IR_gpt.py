# This file is to test models on C-> IR
# We will obtain inferences in two steps:
# 1. Step (1): Pragmatic Identification. Set prediction into: pi_list. Note that we only collect the classification on pragmatic turn. Each item should be True (prediction correct) or false(prediction error)
# 2. Step (2): Rationale. Set prediction into: r_list. item of r_list : True, False
# 3. Calculate Score: score += 1 <=> r_list[i] == True & pi_list[i] == True

# 1. Pragmatic Identification
from datasets import load_from_disk, DatasetDict, Dataset
import torch
from transformers import set_seed
import numpy as np
import argparse
import os
from torch.utils.data import DataLoader
from datasets import load_from_disk, Dataset, DatasetDict
import torch
from transformers import AutoTokenizer, GPT2DoubleHeadsModel
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--pi_checkpoint", type=str)
parser.add_argument("--r_checkpoint",type=str)
args = parser.parse_args()

pi_checkpoint = args.pi_checkpoint
print(pi_checkpoint)
batch_size = 80
if pi_checkpoint == "microsoft/DialoGPT-medium" :
    batch_size = 8
if pi_checkpoint == "gpt2":
    batch_size = 24
dataset = load_from_disk("final_first_IDAR_dataset")
dataset = dataset.rename_column('text','original_text')

print(dataset)
def add_speaker(example):
    text = example['original_text']
    dialogue = []
    for i in range(len(text)):
        utterance = text[i]
        speaker = example['speaker'][i]
        t = speaker + ": " + utterance
        dialogue.append(t)
    return {"text":dialogue}
dataset =  dataset.map(add_speaker)
dataset = dataset.remove_columns(["speaker"])
# process dataset to each turn a piece
def change_dataset(dataset):
    # flatten dataset
    text = []
    original_text = []
    query_turn = []
    label = []
    query_turn_number = []
    for data in dataset:
        data_text = data['text']
        for idx,line in enumerate(data_text):
            query_turn.append(line)
            text.append(data_text)
            original_text.append(data['original_text'])
            query_turn_number.append(idx)

            
            if idx in data['correct_turn_number']:
                label.append(True)
            else:
                label.append(False)
    return Dataset.from_dict({"text":text,"query_turn":query_turn,"label":label, "query_turn_number":query_turn_number, "original_text":original_text})
train_dataset = change_dataset(dataset['train'])
test_dataset = change_dataset(dataset['test'])
validation_dataset = change_dataset(dataset['validation'])
dataset = DatasetDict({"train":train_dataset,"test":test_dataset,"validation":validation_dataset})
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(pi_checkpoint)

# load model
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

model = AutoModelForSequenceClassification.from_pretrained(pi_checkpoint,num_labels = 2,)
def change_label(example):
    if example['label']:
        return {'labels':1}
    else:
        return {'labels':0}
def combine_text(example):
    # combine different utterances together with tokenizer.sep_token
    text = ""
    context = example['text']
    for turn in context:
        text += turn
        text += tokenizer.sep_token
    return {"text":text}
def tokenize(example):
    return tokenizer(example['text'],example['query_turn'],truncation = True)
def tokenize_gpt2(example):
    text = ""
    for t in example['text']:
        text += t
        text += tokenizer.eos_token
    for t in example['query_turn']:
        text += t  
        text += tokenizer.eos_token
    return tokenizer(text,truncation = True, padding = True)


if  "gpt2" not in pi_checkpoint and "DialoGPT-medium" not in pi_checkpoint:
    dataset = dataset.map(combine_text)

dataset = dataset.map(change_label)
dataset = dataset.remove_columns(['label'])
dataset = dataset.rename_column("labels",'label')
if  "gpt2" not in pi_checkpoint and "DialoGPT-medium" not in pi_checkpoint:
    dataset = dataset.map(tokenize)
else:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id
    dataset = dataset.map(tokenize_gpt2)


from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
training_args = TrainingArguments(
    output_dir=f"/scratch/nlp/lihengli/imar_subtask/{pi_checkpoint}/{pi_checkpoint}_model",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=20,
    weight_decay=0.01,
    evaluation_strategy="steps",
    save_strategy="steps",
    load_best_model_at_end=True,

    
)
import evaluate
accuracy = evaluate.load("accuracy")
def compute_metrics(eval_pred):

    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

print("Start to train the model")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset = dataset['train'],
    eval_dataset = dataset['validation'],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    
)
result = trainer.predict(dataset['test'])
print(result)
predictions = result.predictions
pred = np.argmax(predictions,axis = -1)
label = result.label_ids.tolist()
second_dataset = load_from_disk("../subtask_two/final_with_split_IDAR_second_subtask_dataset")
if len(pred) != len(label):
    raise ValueError
pi_list = []
second_id = 0
for i in range(len(pred)):
    text = dataset['test'][i]['original_text']
    query_turn_number = dataset['test'][i]['query_turn_number']
    if label[i] == 1:
    
#        break

        if pred[i] == label[i]:
            while second_id < len(second_dataset['test']) and second_dataset['test'][second_id]['text'] == text and second_dataset['test'][second_id]['correct_turn_number'] == query_turn_number:
                pi_list.append(True)
                second_id += 1
        else:
            while second_id < len(second_dataset['test']) and second_dataset['test'][second_id]['text'] == text and second_dataset['test'][second_id]['correct_turn_number'] == query_turn_number:
                pi_list.append(False)
                second_id += 1
print(len(pi_list))

# step 2: Rationale

r_checkpoint = args.r_checkpoint
batch_size = 2
from datasets import load_from_disk
dataset = load_from_disk("../subtask_two/final_with_split_IDAR_second_subtask_dataset")

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = GPT2DoubleHeadsModel.from_pretrained("microsoft/DialoGPT-medium")

num_added_tokens = tokenizer.add_special_tokens({"cls_token": "[CLS]"})
embedding_layer = model.resize_token_embeddings(len(tokenizer))
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id

checkpoint = torch.load(r_checkpoint)
#best_epoch = checkpoint['best_epoch']
#checkpoint = torch.load(os.path.join(f"/scratch/nlp/lihengli/IDAR2_torch/gpt2/1/epoch_{best_epoch}"))
#print(f"/scratch/nlp/lihengli/IDAR2_torch/gpt2/1/epoch_{best_epoch}")
print(r_checkpoint)
model.load_state_dict(checkpoint['model_state_dict'])
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using Device : {device} ")

model.to(device)

def build_choice(example):
    text = ""
    speaker = example['speaker']

    for i in range(len(example['text'])):
        text += f"{speaker[i]}: {example['text'][i]}"
        text += " "
    choice = []
    for c in example['choice']:
        choice.append(text + " " + c + " [CLS]")

    return {"text":choice}
def tokenize(example):
    choices = example['text']
    encoded_choices = tokenizer(choices, truncation=True)
    return encoded_choices
def add_cls_position(example):
    encoded_choices = example['input_ids']
    mc_token_ids = [tokens.index(tokenizer.cls_token_id) for tokens in encoded_choices]
    return {"mc_token_ids":mc_token_ids}




dataset = dataset.map(build_choice)
dataset = dataset.map(tokenize)
dataset = dataset.map(add_cls_position)
dataset = dataset.remove_columns(['text', 'speaker', 'correct_turn_number', 'choice'])
from transformers import DataCollatorWithPadding
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
        mc_token_ids = [feature.pop("mc_token_ids") for feature in features]

        batch_size = len(features)
        #print("batch size: ", batch_size)
        # each choice a item of input_ids
        num_choices = len(features[0]["input_ids"])
        #print("num choices : ", num_choices)
        #print(features[0].keys())
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = sum(flattened_features, [])

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
            )

        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        batch["mc_labels"] = torch.tensor(labels, dtype=torch.int64)
        batch['mc_token_ids'] = torch.tensor(mc_token_ids, dtype = torch.int64)
        #print("batch")
        #print(batch)
        #print("input ids shape")
        #print(batch['input_ids'].shape)
        #print("index shape")
        #print(batch['mc_token_ids'].shape)
        return batch
data_collator = DataCollatorForMultipleChoice(tokenizer=tokenizer)
from torch.utils.data import DataLoader
train_loader = DataLoader(dataset['train'],batch_size = batch_size, collate_fn = data_collator)
validation_loader = DataLoader(dataset['validation'],batch_size = batch_size, collate_fn = data_collator)
test_loader = DataLoader(dataset['test'],batch_size = batch_size, collate_fn = data_collator)
import evaluate
accuracy = evaluate.load("accuracy")
from sklearn.metrics import accuracy_score
whole_pred = []
whole_label = []
def test(example, loader):
    global whole_pred
    global whole_label
    model.eval()
    with torch.no_grad():
        count = 0
        whole_score = 0
        for batch in loader:
            batch = {k:v.to(device) for k,v in batch.items()}
            logits = model(**batch).mc_logits
            pred = logits.argmax(axis = -1)
            whole_pred += pred.cpu().tolist()
            whole_label += batch['mc_labels'].cpu().tolist()
            score = accuracy_score(batch['mc_labels'].cpu(), pred.cpu())
            whole_score += score
            count += 1
    return whole_score/count
test_result = test(model,test_loader)
print("test acc : ", test_result)


r_list = []
if len(whole_pred) != len(whole_label):
    print("big error")
for i in range(len(whole_pred)):
    if whole_pred[i] == whole_label[i]:
        r_list.append(True)
    else:
        r_list.append(False)
if len(r_list) != len(pi_list):
    print("BIIIIGGG ERRROR")
print(r_list)
count = 0
r_true = 0
pi_true = 0
for i in range(len(r_list)):
    if r_list[i] and pi_list[i]:
        count += 1
    if r_list[i]:
        r_true += 1
    if pi_list[i]:
        pi_true += 1
print("r_true :  ",r_true)
print("pi_true : ",pi_true)
print("Accuracy Ratio : ",count / len(r_list))


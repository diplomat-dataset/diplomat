import random
from transformers import TrainingArguments, Trainer, set_seed
import evaluate
import numpy as np
import torch
import argparse
import wandb
from tqdm import tqdm
parser = argparse.ArgumentParser()
parser.add_argument("--model_checkpoint", type=str, default="gpt2")
parser.add_argument("--seed",type=int,default=42)
parser.add_argument("--batch_size",type=int,default=8)
parser.add_argument("--learning_rate",type=float,default=0.001)
parser.add_argument("--num_train_epochs",type=int,default=50)
parser.add_argument("--weight_decay",type=float,default=0.01)
parser.add_argument("--resume", type=str, default=None)


args = parser.parse_args()

seed  = args.seed 
batch_size = args.batch_size
lr = args.learning_rate
epochs = args.num_train_epochs
weight_decay = args.weight_decay
model_checkpoint = args.model_checkpoint
wandb.init(project=f"IDAR2_torch",name=f"{model_checkpoint}_{seed}")
wandb.config.update(args)



def setup_seed(seed):
    '''set seed for reproducibility'''
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    set_seed(seed)
setup_seed(seed)
from datasets import load_from_disk, Dataset, DatasetDict
import torch
from transformers import AutoTokenizer, GPT2DoubleHeadsModel

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = GPT2DoubleHeadsModel.from_pretrained(model_checkpoint)

# Add a [CLS] to the vocabulary (we should train it also!)
num_added_tokens = tokenizer.add_special_tokens({"cls_token": "[CLS]"})
# Update the model embeddings with the new vocabulary size
embedding_layer = model.resize_token_embeddings(len(tokenizer))
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id
dataset = load_from_disk("final_with_split_IDAR_second_subtask_dataset")

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
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using Device : {device} ")
model.to(device)
print("*"*10 + "start to train" + "*" * 10)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay = weight_decay)
criterion = torch.nn.CrossEntropyLoss()
def train(model):
    model.train()
    whole_loss = 0
    count = 0
    for batch in train_loader:
        optimizer.zero_grad()
        batch = {k:v.to(device) for k,v in batch.items()}
        result = model(**batch)
        logits = result.mc_logits
        loss = criterion(logits,batch['mc_labels'])
        whole_loss += loss
        loss.backward()
        optimizer.step()
        count += 1
    average_loss = whole_loss/count
    return average_loss
import evaluate
accuracy = evaluate.load("accuracy")
from sklearn.metrics import accuracy_score

def test(example, loader):
    model.eval()
    with torch.no_grad():
        count = 0
        whole_score = 0
        for batch in loader:
            batch = {k:v.to(device) for k,v in batch.items()}
            logits = model(**batch).mc_logits
            pred = logits.argmax(axis = -1)
            score = accuracy_score(batch['mc_labels'].cpu(), pred.cpu())
            whole_score += score
            count += 1
    return whole_score/count
validation_best = 0
best_epoch = 0
start_epoch = 0

if args.resume:
    print("*" * 20)
    print(f"resume : {args.resume}")
    print("*" * 20)
    if seed == 1:
#        best_epoch = 2
#        validation_best = 0.2827868852459016
        checkpoint = torch.load(args.resume)
        best_epoch = checkpoint['best_epoch']
        validation_best = checkpoint['validation_best']
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        print(f"best epoch : {best_epoch}, validation best : {validation_best}")



    elif seed == 19:
#        best_epoch = 21
#        validation_best = 0.2786885245901639
        checkpoint = torch.load(args.resume)
        best_epoch = checkpoint['best_epoch']
        validation_best = checkpoint['validation_best']
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        print(f"best epoch : {best_epoch}, validation best : {validation_best}")
    elif seed == 588:
        checkpoint = torch.load(args.resume)
        best_epoch = checkpoint['best_epoch']
        validation_best = checkpoint['validation_best']
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        print(f"best epoch : {best_epoch}, validation best : {validation_best}")



    else:
        print("Wrong seed!")
    
import os
model_name = model_checkpoint.split("/")[-1]
for epoch in tqdm(range(start_epoch,epochs)):
    loss = train(model)
    validation_acc = test(model, validation_loader)
    print(f"Epoch {epoch} : training loss : {loss}, validation acc : {validation_acc}")
    wandb.log({"training loss":loss, "validation acc":validation_acc})
    
    if validation_acc > validation_best:
        best_epoch = epoch
        validation_best = validation_acc
        file_path = f"/scratch/nlp/lihengli/IDAR2_torch/{model_name}/{seed}/best_epoch_{model_name}_{seed}.txt"
        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path))  
        with open(file_path,"w") as file:
            file.write(f"best epoch : {best_epoch}, validation score : {validation_best}")
    checkpoint_save_path = f"/scratch/nlp/lihengli/IDAR2_torch/{model_name}/{seed}/epoch_{epoch}"
    #print(os.path.dirname(checkpoint_save_path))
    if not os.path.exists(os.path.dirname(checkpoint_save_path)):
        #print("hahha")
        os.makedirs(os.path.dirname(checkpoint_save_path)) 
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_epoch':best_epoch,
            'validation_best':validation_best,
            'loss': loss,
            }, checkpoint_save_path)
checkpoint = torch.load(os.path.join(f"/scratch/nlp/lihengli/IDAR2_torch/{model_name}/{seed}/epoch_{best_epoch}"))
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
test_result = test(model,test_loader)
wandb.log({"test acc":test_result})
print(test_result)

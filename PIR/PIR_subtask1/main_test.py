from datasets import load_from_disk, DatasetDict, Dataset
import torch
from transformers import set_seed
import numpy as np
import argparse
import wandb
parser = argparse.ArgumentParser()
parser.add_argument("--model_checkpoint", type=str, default="bert-base-uncased")
parser.add_argument("--seed",type=int,default=42)
parser.add_argument("--batch_size",type=int,default=24)

args = parser.parse_args()

model_checkpoint = args.model_checkpoint
seed = args.seed
batch_size = args.batch_size
print("-"*20)
print(f"{model_checkpoint}, seed : {seed}, batch size : {batch_size}")
print("-"*20)
#batch_size = 80
#if model_checkpoint == "microsoft/DialoGPT-medium" :
#    batch_size = 8
#if model_checkpoint == "gpt2":
#    batch_size = 24



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
print("model checkpoint : ",model_checkpoint)
wandb.init(project=f"IMAR",name=f"{model_checkpoint}_{seed}")

dataset = load_from_disk("final_first_IDAR_dataset")
print(dataset)
original_dataset = dataset
def add_speaker(example):
    text = example['text']
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
    query_turn = []
    label = []
    for data in dataset:
        data_text = data['text']
        for idx,line in enumerate(data_text):
            query_turn.append(line)
            text.append(data_text)
            if idx in data['correct_turn_number']:
                label.append(True)
            else:
                label.append(False)
    return Dataset.from_dict({"text":text,"query_turn":query_turn,"label":label})
train_dataset = change_dataset(dataset['train'])
test_dataset = change_dataset(dataset['test'])
validation_dataset = change_dataset(dataset['validation'])
dataset = DatasetDict({"train":train_dataset,"test":test_dataset,"validation":validation_dataset})
print(dataset)
# load tokenizer
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, max_length = 512)
tokenizer.model_max_length = 512

# load model
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint,num_labels = 2,)
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


if model_checkpoint != "gpt2" and model_checkpoint != "microsoft/DialoGPT-medium":
    dataset = dataset.map(combine_text)

dataset = dataset.map(change_label)
dataset = dataset.remove_columns(['label'])
dataset = dataset.rename_column("labels",'label')
if model_checkpoint != "gpt2" and model_checkpoint != "microsoft/DialoGPT-medium" :
    dataset = dataset.map(tokenize)
else:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id
    dataset = dataset.map(tokenize_gpt2)


from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)



print("Loading the model")
model_name = model_checkpoint.split("/")[-1]
training_args = TrainingArguments(
    output_dir=f"/scratch/nlp/lihengli/imar_subtask/{model_name}/{seed}/{model_name}_model",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=20,
    weight_decay=0.01,
    evaluation_strategy="steps",
    save_strategy="steps",
    load_best_model_at_end=True,
    report_to = "wandb"

    
)
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
#print(f"/scratch/nlp/lihengli/imar_subtask/microsoft/DialoGPT-medium/{seed}/microsoft/DialoGPT-medium_model/checkpoint-26000")

print("START TRAINING")
print("-"*30)
#print(f"/scratch/nlp/lihengli/imar_subtask/{model_checkpoint}/{seed}/{model_checkpoint}_model/checkpoint-31500")
#trainer.train(f"/scratch/nlp/lihengli/imar_subtask/microsoft/deberta-v3-base/19/microsoft/deberta-v3-base_model/checkpoint-11000")
trainer.train()
result = trainer.predict(dataset['test'])
predictions = result.predictions
pred = np.argmax(predictions,axis = -1)
count = 0
correct = 0
for i in range(len(original_dataset['test'])):
    test_data = original_dataset['test'][i]
    label = [0 for i in range(len(test_data['text']))]
    correct_turn_number = test_data['correct_turn_number']
    for turn in correct_turn_number:
        label[turn] = 1
    length = len(label)
    corres_prediction = pred[count:count + length]
    if (corres_prediction == label).all():
        correct += 1
    count += length
wandb.log({"test loss":result.metrics['test_loss'], "test accuracy":result.metrics['test_accuracy'], "test whole acc":correct/len(original_dataset['test'])})
print(f"test loss: {result.metrics['test_loss']} test accuracy: {result.metrics['test_accuracy']}, test whole acc:{correct/len(original_dataset['test'])}")

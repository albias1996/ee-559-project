from copy import deepcopy
from torch.utils.data import Dataset
from tqdm import tqdm
import json
import jsonlines
import re
import emoji
import os 
from torch.nn.utils import clip_grad_norm_

import torch
import torch.utils.data
from torch import nn, optim
import numpy as np 
from torch.utils.data import DataLoader, RandomSampler, WeightedRandomSampler
from transformers import AdamW
from torch.nn import CrossEntropyLoss

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix

def preprocess_text(text):
    #lowercase the text
    text = text.lower()
    
    #replace all URLs with "[URL]"
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    text = url_pattern.sub('[URL]', text)
    
    #replace emojis with "[EMOJI]"
    text = emoji.replace_emoji(text, replace='[EMOJI]')
    
    #remove newlines and tab characters
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def get_paraphrase_batch(
    model, 
    tokenizer, 
    input_samples, 
    num_paraphrases,
    device,
    repetition_penalty=10.0,
    diversity_penalty=3.0,
    no_repeat_ngram_size=2,
    temperature=0.7,
    max_length=512):

    '''
    Input
      model: paraphraser
      tokenizer: paraphrase tokenizer
      input_samples: a batch (list) of real samples to be paraphrased
      n: number of paraphrases to get for each input sample
      for other parameters, please refer to:
          https://huggingface.co/docs/transformers/en/main_classes/text_generation#transformers.GenerationConfig
    Output: Tuple.
      synthetic_samples: a list of paraphrased samples
    '''

    synthetic_samples = []
    # input_samples_process = preprocess_text(input_samples['body'])
    input_samples_process = [preprocess_text(sample['text']) for sample in input_samples]
    for sample in input_samples_process:
        input_ids = tokenizer.encode(sample, return_tensors="pt", padding=True, max_length=max_length, truncation=True).to(device)
        output_ids = model.generate(
            input_ids,
            max_length=max_length,
            repetition_penalty=repetition_penalty,
            diversity_penalty=diversity_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            temperature=temperature,
            num_beams=6,
            num_beam_groups=3,
            num_return_sequences=num_paraphrases)
        paraphrased_1 = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        paraphrased_2 = tokenizer.decode(output_ids[1], skip_special_tokens=True)
        synthetic_samples.append((paraphrased_1, paraphrased_2))

    return synthetic_samples


def get_paraphrase_dataset(model, tokenizer, device, data_path, batch_size, num_paraphrases):
    '''
    Input
      model: paraphrase model
      tokenizer: paraphrase tokenizer
      data_path: path to the `jsonl` file of training data
      batch_size: number of input samples to be paraphrases in one batch
      n_paraphrase: number of paraphrased sequences for each sample
    Output:
      paraphrase_dataset: a list of all paraphrase samples. Do not include the original training data.
    '''
    paraphrase_dataset = []
    with jsonlines.open(data_path, "r") as reader:

        input_samples = []
        for sample in reader:
            #input_samples.append(preprocess_text(sample['body']))
            input_samples.append(sample)

            if len(input_samples) == batch_size:
                paraphrases = get_paraphrase_batch(model, tokenizer, input_samples, num_paraphrases, device=device)

                #append paraphrased samples with original metadata
                for original_sample, paraphrased_texts in zip(input_samples, paraphrases):
                    for text in paraphrased_texts:
                        new_entry = original_sample.copy()
                        new_entry['text'] = text
                        paraphrase_dataset.append(new_entry)
                #paraphrase_dataset.extend(paraphrases)
                input_samples = []

        #process the last batch
        if len(input_samples) > 0:
            paraphrases = get_paraphrase_batch(model, tokenizer, input_samples, num_paraphrases, device=device)
            for original_sample, paraphrased_texts in zip(input_samples, paraphrases):
                for text in paraphrased_texts:
                    new_entry = original_sample.copy()
                    new_entry['text'] = text
                    paraphrase_dataset.append(new_entry)
            #paraphrase_dataset.extend(paraphrases)
        

    return paraphrase_dataset

class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_len, split_types):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = []
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                entry = json.loads(line)
                #if (entry['split'] in split_types) and (entry['label']=='Misogynistic'):
                if (entry['split'] in split_types):
                    preprocessed_text = preprocess_text(entry['text'])
                    #self.data.append((preprocessed_text, 1 if entry['label'] == 'sexist' else 0))
                    self.data.append((preprocessed_text, 1 if entry['label'] == 'Misogynistic' else 0))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text, label = self.data[index]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }
    
    
def calculate_weights(dataset):
    #compute class weights to handle imbalance more effectively
    label_counts = torch.tensor([sum(1 for _, label in dataset.data if label == i) for i in range(2)])
    weights = (1 / label_counts).float()
    print("Those are the weights:", weights)
    #weights[1] *= 9  # Increase the weight of the minority class as specified
    return weights

def evaluate_test_set(model, checkpoint_path, file_path, tokenizer, max_len, batch_size, criterion, device):

    #checkpoint = torch.load(checkpoint_path)
    if device=='cpu':
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(checkpoint_path) 

    #checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)

    test_dataset = TextDataset(file_path, tokenizer, max_len, split_types=['test'])
    print("the len of test dataset:", len(test_dataset))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=None)
            loss = criterion(outputs.logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(test_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average=None, zero_division=0)
    recall = recall_score(all_labels, all_preds, average=None, zero_division=0)
    f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)

    conf_matrix = confusion_matrix(all_labels, all_preds)

    #print(conf_matrix)

    #print(f'Evaluation part: Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')

    return avg_loss, precision, recall, f1, accuracy, conf_matrix


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=None)
            loss = criterion(outputs.logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds,  average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    return avg_loss, accuracy, precision, recall, f1



def train(model, restore_ckpt, file_path, tokenizer, max_len, batch_size, epochs, lr, weight_decay, use_weights, device, experiment_dir):

    if restore_ckpt is not None:
        assert restore_ckpt.endswith(".pth")
        print("Loading checkpoint...")
        checkpoint = torch.load(restore_ckpt)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        print(f"Done loading checkpoint")

    train_dataset = TextDataset(file_path, tokenizer, max_len, split_types=['train'])
    val_dataset = TextDataset(file_path, tokenizer, max_len, split_types=['dev'])

    if use_weights:
        weights = calculate_weights(train_dataset)
        samples_weights = weights[[label for _, label in train_dataset.data]]
        sampler = WeightedRandomSampler(samples_weights, len(samples_weights), replacement=True)
    else:
        sampler = RandomSampler(train_dataset)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model.to(device)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    if use_weights:
        criterion = CrossEntropyLoss(weight=weights.to(device))
    else:
        criterion = CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        val_loss, accuracy, precision, recall, f1 = evaluate(model, val_loader, criterion, device)

        print(f'Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')

        # Save model and metrics
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': val_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }, os.path.join(experiment_dir, f'model_epoch_{epoch+1}.pth'))

    print('Finished Training')
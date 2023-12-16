import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
import seaborn as sns
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm
from nlp import load_dataset
from transformers import RobertaTokenizer
from torch.utils.data import Dataset, TensorDataset, DataLoader, SequentialSampler, RandomSampler
from torch.nn.utils.rnn import pad_sequence
from transformers import RobertaForSequenceClassification, AdamW


# Check Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Load/Process Datasets
train_data = load_dataset('multi_nli', split='train')
val_data = load_dataset('multi_nli', split='validation_matched')
train_dataset = [item for item in train_data]
val_dataset = [item for item in val_data]
train_df = pd.DataFrame(train_dataset)
val_df = pd.DataFrame(val_dataset)
group_labels = pd.read_csv('metadata_preset.csv')
train_group_labels = group_labels[group_labels['split'] == 0]['sentence2_has_negation']
val_group_labels = group_labels[group_labels['split'] == 1]['sentence2_has_negation'].reset_index()['sentence2_has_negation']
train_df = train_df.merge(train_group_labels, left_index=True, right_index=True)
val_df = val_df.merge(val_group_labels, left_index=True, right_index=True)
train_df = train_df.rename(columns={"premise": "sentence1", "hypothesis": "sentence2", "label": "gold_label", "sentence2_has_negation":"negation"})
val_df = val_df.rename(columns={"premise": "sentence1", "hypothesis": "sentence2", "label": "gold_label", "sentence2_has_negation":"negation"})
label_map = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}
train_df['gold_label'] = train_df['gold_label'].map(label_map)
val_df['gold_label'] = val_df['gold_label'].map(label_map)
val_df['class'] = val_df['gold_label'].astype(str) + "_" + val_df['negation'].astype(str)
train_df['class'] = train_df['gold_label'].astype(str) + "_" + train_df['negation'].astype(str)
train_df = train_df.dropna()
val_df = val_df.dropna()
train_df['sentence1'] = train_df['sentence1'].astype(str)
train_df['sentence2'] = train_df['sentence2'].astype(str)
val_df['sentence1'] = val_df['sentence1'].astype(str)
val_df['sentence2'] = val_df['sentence2'].astype(str)
train_df = train_df[(train_df['sentence1'].str.split().str.len() > 0) & (train_df['sentence2'].str.split().str.len() > 0)]
val_df = val_df[(val_df['sentence1'].str.split().str.len() > 0) & (val_df['sentence2'].str.split().str.len() > 0)]


# Define Dataset Classes
class MNLIDataRoberta(Dataset):
  def __init__(self, train_df, val_df):
    self.label_dict = {'entailment': 0, 'contradiction': 1, 'neutral': 2}

    # Add negation label
    self.negation_dict = {'entailment_0': 0, 'entailment_1': 1, 'contradiction_0': 2, 'contradiction_1': 3, 'neutral_0': 4, 'neutral_1': 5}
    self.train_df = train_df
    self.val_df = val_df
    self.base_path = '/content/'
    self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base', do_lower_case=True)
    self.train_data = None
    self.val_data = None
    self.init_data()

  def init_data(self):
    self.train_data = self.load_data(self.train_df)
    self.val_data = self.load_data(self.val_df)

  def load_data(self, df):
    MAX_LEN = 512
    token_ids = []
    mask_ids = []
    y = []
    n = []

    premise_list = df['sentence1'].to_list()
    hypothesis_list = df['sentence2'].to_list()
    label_list = df['gold_label'].to_list()
    negation_list = df['class'].to_list()

    for (premise, hypothesis, label, negation) in tqdm(zip(premise_list, hypothesis_list, label_list, negation_list)):
      premise_id = self.tokenizer.encode(premise, add_special_tokens = False)
      hypothesis_id = self.tokenizer.encode(hypothesis, add_special_tokens = False)
      pair_token_ids = [self.tokenizer.bos_token_id] + premise_id + [self.tokenizer.sep_token_id] + hypothesis_id + [self.tokenizer.eos_token_id]

      premise_len = len(premise_id)
      hypothesis_len = len(hypothesis_id)

      attention_mask_ids = torch.tensor([1] * (premise_len + hypothesis_len + 3))  # mask padded values

      token_ids.append(torch.tensor(pair_token_ids))
      mask_ids.append(attention_mask_ids)
      y.append(self.label_dict[label])
      n.append(self.negation_dict[negation])
    
    token_ids = pad_sequence(token_ids, batch_first=True)
    mask_ids = pad_sequence(mask_ids, batch_first=True)
    y = torch.tensor(y)
    n = torch.tensor(n)
    dataset = TensorDataset(token_ids, mask_ids, y, n)
    print(len(dataset))
    return dataset

  def get_data_loaders(self, batch_size=32, shuffle=True):
    train_loader = DataLoader(
      self.train_data,
      shuffle=shuffle,
      batch_size=batch_size
    )

    val_loader = DataLoader(
      self.val_data,
      shuffle=shuffle,
      batch_size=batch_size
    )

    return train_loader, val_loader

# Define Dataset/Model
mnli_dataset = MNLIDataRoberta(train_df[:], val_df[:])
train_loader, val_loader = mnli_dataset.get_data_loaders(batch_size=8)

model = RobertaForSequenceClassification.from_pretrained("Roberta-base-uncased", num_labels=3)
model.to(device)

# Weight Decay
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'gamma', 'beta']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.0}
]

# Define Optimizer
optimizer = AdamW(optimizer_grouped_parameters, lr=2e-6, correct_bias=False)


# Parameter Count
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')


# Accuracy Metric
def multi_acc(y_pred, y_test):
  acc = (torch.log_softmax(y_pred, dim=1).argmax(dim=1) == y_test).sum().float() / float(y_test.size(0))
  return acc

# Train Loop
EPOCHS = 5

negation_dict = {0: 'entailment_0',
 1: 'entailment_1',
 2: 'contradiction_0',
 3: 'contradiction_1',
 4: 'neutral_0',
 5: 'neutral_1'}

def train(model, train_loader, val_loader, optimizer):  
  train_loss_values = []
  val_loss_values = []
  for epoch in range(EPOCHS):
    model.train()
    total_train_loss = 0
    total_train_acc  = 0
    for batch_idx, (pair_token_ids, mask_ids, y, n) in enumerate(tqdm(train_loader, desc="Train")):
      optimizer.zero_grad()
      pair_token_ids = pair_token_ids.to(device)
      mask_ids = mask_ids.to(device)
      labels = y.to(device)
      negation = n.to(device)
      loss, prediction = model(pair_token_ids,
                             attention_mask=mask_ids, 
                             labels=labels).values()

      acc = multi_acc(prediction, labels)

      loss.backward()
      optimizer.step()
      
      total_train_loss += loss.item()
      total_train_acc  += acc.item()

    train_acc  = total_train_acc/len(train_loader)
    train_loss = total_train_loss/len(train_loader)
    model.eval()
    total_val_acc  = 0
    total_val_loss = 0

    class_correct_predictions = defaultdict(int)
    class_total_predictions = defaultdict(int)

    with torch.no_grad():
      for batch_idx, (pair_token_ids, mask_ids, y, n) in enumerate(tqdm(val_loader, desc="Val")):
        optimizer.zero_grad()
        pair_token_ids = pair_token_ids.to(device)
        mask_ids = mask_ids.to(device)
        labels = y.to(device)
        negation = n.to(device)

        loss, prediction = model(pair_token_ids,
                             attention_mask=mask_ids, 
                             labels=labels).values()

        predictions = torch.log_softmax(prediction, dim=1).argmax(dim=1)
        # Compute accuracy per negation class
        for i in range(len(labels)):
          class_identifier = negation_dict[int(negation[i].item())]
          class_correct_predictions[class_identifier] += int(predictions[i] == labels[i])
          class_total_predictions[class_identifier] += 1

        acc = multi_acc(prediction, labels)

        total_val_loss += loss.item()
        total_val_acc  += acc.item()

    val_acc  = total_val_acc/len(val_loader)
    val_loss = total_val_loss/len(val_loader)
    
    print(f'Epoch {epoch+1}: train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} | val_loss: {val_loss:.4f} val_acc: {val_acc:.4f}')
    train_loss_values.append(train_loss)
    val_loss_values.append(val_loss)

    values = []
    for class_id, correct_count in sorted(class_correct_predictions.items()):
      accuracy = correct_count / class_total_predictions[class_id]
      values.append(accuracy)
      print(f"--- Class {class_id} accuracy: {accuracy:.4f}")
    
    plt.figure()
    ax = sns.heatmap(np.vstack((values[::2], values[1::2])), annot=True, fmt='.2%', cmap='Blues', xticklabels=['contradiction', 'entailment', 'neutral'], yticklabels=['no', 'yes'])
    plt.title('RoBERTa - MNLI Class Accuracies', fontsize=15, fontweight='bold')
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Contains Negation Word (Spurious)', fontsize=12)
    plt.savefig(f'plots/RoBERTa_{epoch}.png')
    print('\n')

  plt.figure()
  plt.plot(train_loss_values)
  plt.title('RoBERTa - Train Loss', fontsize=15, fontweight='bold')
  plt.xlabel('Epoch', fontsize=12)
  plt.ylabel('Loss', fontsize=12)
  plt.savefig(f'plots/RoBERTa_TrainLoss.png')

  plt.figure()
  plt.plot(val_loss_values)
  plt.title('RoBERTa - Validation Loss', fontsize=15, fontweight='bold')
  plt.xlabel('Epoch', fontsize=12)
  plt.ylabel('Loss', fontsize=12)
  plt.savefig(f'plots/RoBERTa_ValLoss.png')

  model.save_pretrained("RoBERTa", from_pt=True)

train(model, train_loader, val_loader, optimizer)
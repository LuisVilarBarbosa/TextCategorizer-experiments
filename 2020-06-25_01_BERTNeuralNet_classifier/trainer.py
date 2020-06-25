# Developed on May 13, 15, 20, 22, 25 and 26, 2020
# and June 04, 05, 08, 09, 12, 16, 17, 18, 19, 22 and 25, 2020.
# Based on https://medium.com/swlh/painless-fine-tuning-of-bert-in-pytorch-b91c14912caa
# (accessed on May 12, 2020),
# https://github.com/kabirahuja2431/FineTuneBERT/blob/master/src/main.py
# (accessed on May 15, 2020),
# https://github.com/huggingface/transformers/blob/v2.5.1/examples/run_glue.py
# (accessed on June 04, 2020) and
# https://github.com/LuisVilarBarbosa/TextCategorizer-experiments/tree/master/2020-03-09_01_BERT_by_sentence.

import json
import time
from itertools import zip_longest
from sys import argv

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics, utils
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import BertModel, BertTokenizer

BATCH_SIZE = 32
BERT_MODEL = 'bert-base-multilingual-cased'
FREEZE_BERT = True
LEARNING_RATE = 1e-3
MAX_EPOCHS = 30
MAX_LENGTH = 512
RANDOM_STATE = 42

class BERTTokenizedDataset(Dataset):
    def __init__(self, corpus, labels, model=BERT_MODEL, cache_dir=BERT_MODEL):
        super(BERTTokenizedDataset, self).__init__()
        self.corpus = corpus
        self.labels = labels
        assert len(self.corpus) == len(self.labels)
        self.tokenizer = BertTokenizer.from_pretrained(model, cache_dir=cache_dir)
        self.classes_ = set(self.labels)
        self.label_to_int = dict((l, i) for i, l in enumerate(self.classes_))
        self.int_to_label = dict((i, l) for i, l in enumerate(self.classes_))

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, index):
        text = self.corpus[index]
        label = self.label_to_int[self.labels[index]]
        data = self.tokenizer.encode_plus(text, add_special_tokens=True, max_length=MAX_LENGTH,
                                          pad_to_max_length=True, return_tensors=None,
                                          return_token_type_ids=False, return_attention_mask=True)
        input_ids = torch.tensor(data['input_ids'])
        attention_mask = torch.tensor(data['attention_mask'])
        return input_ids, attention_mask, label

class BERTNeuralNet(nn.Module):
    def __init__(self, num_classes, freeze_bert, model=BERT_MODEL, cache_dir=BERT_MODEL):
        super(BERTNeuralNet, self).__init__()
        self.num_classes = num_classes
        self.bert_layer = BertModel.from_pretrained(model, cache_dir=cache_dir)
        if freeze_bert:
            for p in self.bert_layer.parameters():
                p.requires_grad = False
        self.additional_layers = nn.Sequential(
            nn.Linear(768, 768),
            nn.Linear(768, self.num_classes),
        )

    def forward(self, input_ids, attention_mask):
        last_hidden_state, _pooler_output = self.bert_layer(input_ids, attention_mask=attention_mask)
        cls_logits = last_hidden_state[:, 0, :]
        logits = self.additional_layers(cls_logits)
        return logits

def predict_proba_to_dicts(clf_classes_, y_predict_proba):
    y_predict_proba_shape = (len(y_predict_proba), len(y_predict_proba[0]))
    assert all(len(probs) == y_predict_proba_shape[1] for probs in y_predict_proba)
    assert len(clf_classes_) == y_predict_proba_shape[1]
    my_y_predict_proba = [dict(zip_longest(clf_classes_, probs)) for probs in y_predict_proba]
    return my_y_predict_proba

def dicts_to_predict(dicts, y_true=None, n_accepted_probs=1):
    assert (y_true is None and n_accepted_probs == 1) \
        or (y_true is not None and n_accepted_probs >= 1 and len(dicts) == len(y_true))
    sorted_probs = [sorted(d.items(), key=lambda item: -item[1]) for d in dicts]
    accepted_classes = [[item[0] for item in l[0:n_accepted_probs]] for l in sorted_probs]
    if y_true is None:
        y_pred = [cs[0] for cs in accepted_classes]
    else:
        y_pred = [t if t in cs else cs[0] for cs, t in zip_longest(accepted_classes, y_true)]
    return y_pred

def dump_json(obj, path):
    f = open(path, 'w')
    json.dump(obj, f)
    f.close()

def load_json(path):
    f = open(path, 'r')
    obj = json.load(f)
    f.close()
    return obj

def get_local_time_str():
    return time.strftime('%Y-%m-%d %H:%M:%S %z %Z', time.localtime())

def predictions_to_data_frame(predictions_dict, n_accepted_probs):
    predictions = predictions_dict.copy()
    y_true = predictions.pop('y_true')
    data = dict()
    for clf, y_predict_proba in predictions.items():
        y_pred = dicts_to_predict(y_predict_proba, y_true, n_accepted_probs)
        report = metrics.classification_report(y_true, y_pred, output_dict=True)
        for label in report.keys():
            for metric in report[label].keys():
                col = '%s %s %s' % (metric, clf, label)
                data[col] = report[label][metric]
    df = pd.DataFrame([data])
    return df

def generate_report(execution_info, predictions_dict, excel_file='report.xlsx'):
    try:
        df1 = pd.read_excel(excel_file)
    except FileNotFoundError:
        df1 = pd.DataFrame()
    exec_info = execution_info.copy()
    assert exec_info['Accepted probabilities'].shape == (1,)
    for n_accepted_probs in range(1, exec_info['Accepted probabilities'][0] + 1):
        exec_info['Accepted probabilities'] = n_accepted_probs
        predictions_df = predictions_to_data_frame(predictions_dict, n_accepted_probs)
        df2 = pd.concat(objs=[exec_info, predictions_df], axis=1, sort=False)
        df1 = pd.concat(objs=[df1, df2], axis=0, sort=False)
    df1.to_excel(excel_file, index=False)
    return df1

def get_accuracy_from_logits(logits, labels):
    probs = torch.softmax(logits, dim=1)
    predictions = torch.argmax(probs, dim=1)
    acc = (predictions == labels).float().mean()
    return acc, probs

def evaluate(net, criterion, dataloader, device):
    net = net.to(device)
    net.eval()
    acc_sum, loss_sum, count = 0, 0, 0
    y_test, y_predict_proba = [], []
    with torch.no_grad():
        base_description = 'Evaluating'
        tq = tqdm(iterable=dataloader, desc=base_description, unit='batch')
        for batch in tq:
            batch = tuple(t.to(device) for t in batch)
            input_ids, attention_mask, labels = batch
            logits = net(input_ids, attention_mask)
            loss_sum += criterion(logits, labels).item()
            acc, probs = get_accuracy_from_logits(logits, labels)
            acc_sum += acc
            count += 1
            y_test.extend(labels.tolist())
            y_predict_proba.extend(probs.tolist())
            loss_avg = loss_sum / count
            acc_avg = acc_sum / count
            tq.set_description('%s (Loss=%.5f Acc=%.5f)' % (base_description, loss_avg, acc_avg))
        tq.close()
    y_test = [dataloader.dataset.int_to_label[i] for i in y_test]
    return acc_sum / count, loss_sum / count, y_test, y_predict_proba

def train_and_save_net(net, criterion, optimizer, train_loader, val_loader, max_epochs, device):
    net = net.to(device)
    best_acc = 0
    predictions = {}
    for epoch in tqdm(iterable=range(1, max_epochs + 1), desc='Epoch', unit='epoch'):
        net.train()
        acc_sum, loss_sum, count = 0, 0, 0
        base_description = 'Training'
        tq = tqdm(iterable=train_loader, desc=base_description, unit='batch')
        for batch in tq:
            batch = tuple(t.to(device) for t in batch)
            input_ids, attention_mask, labels = batch
            optimizer.zero_grad()
            logits = net(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            acc, _probs = get_accuracy_from_logits(logits, labels)
            loss_sum += loss.item()
            acc_sum += acc
            count += 1
            loss_avg = loss_sum / count
            acc_avg = acc_sum / count
            tq.set_description('%s (Loss=%.5f Acc=%.5f)' % (base_description, loss_avg, acc_avg))
        tq.close()
        print()
        val_acc, val_loss, y_test, y_predict_proba = evaluate(net, criterion, val_loader, device)
        print(f'\n\nEpoch {epoch} complete! Validation accuracy: {val_acc}, validation loss: {val_loss}')
        if val_acc > best_acc:
            print(f'Best validation accuracy improved from {best_acc} to {val_acc}, saving data...')
            best_acc = val_acc
            dicts = predict_proba_to_dicts(val_loader.dataset.classes_, y_predict_proba)
            predictions[net.__class__.__name__] = dicts
            save_path = f'train_checkpoint_epoch_{epoch}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'validation_loss': val_loss,
                'validation_accuracy': val_acc,
                }, save_path)
            print(f'Checkpoint saved to "{save_path}".')
    predictions['y_true'] = y_test
    return predictions

def train(excel_file, text_column, labels_column, train_test_idxs_file, n_accepted_probs, output_file):
    execution_info = pd.DataFrame()
    execution_info['Start date'] = [get_local_time_str()]
    torch.manual_seed(RANDOM_STATE)
    device = torch.device(f'cuda:{torch.cuda.current_device()}' \
                          if torch.cuda.is_available() \
                          else 'cpu')
    device_str = f'{device.type}:{device.index} ({torch.cuda.get_device_name(device.index)})' \
                 if device.type == 'cuda' \
                 else device.type
    print(f'Device: {device_str}')
    df = pd.read_excel(excel_file)
    df = df.fillna('NaN')
    corpus = df[text_column].tolist()
    labels = df[labels_column].tolist()
    train_test_idxs = load_json(train_test_idxs_file)
    train_idxs = train_test_idxs['train_idxs']
    test_idxs = train_test_idxs['test_idxs']
    corpus_train = utils.safe_indexing(corpus, train_idxs)
    corpus_test = utils.safe_indexing(corpus, test_idxs)
    y_train = utils.safe_indexing(labels, train_idxs)
    y_test = utils.safe_indexing(labels, test_idxs)
    train_set = BERTTokenizedDataset(corpus_train, y_train)
    val_set = BERTTokenizedDataset(corpus_test, y_test)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, num_workers=1)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, num_workers=1)
    assert train_loader.dataset.classes_ == val_loader.dataset.classes_
    net = BERTNeuralNet(len(val_loader.dataset.classes_), freeze_bert=FREEZE_BERT)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    predictions = train_and_save_net(net, criterion, optimizer, train_loader, val_loader, MAX_EPOCHS, device)
    dump_json(predictions, 'predictions.json')
    execution_info['End date'] = [get_local_time_str()]
    execution_info['Excel file'] = [excel_file]
    execution_info['Text column'] = [text_column]
    execution_info['Label column'] = [labels_column]
    execution_info['Accepted probabilities'] = [n_accepted_probs]
    execution_info['Loss function'] = [criterion.__class__.__name__]
    execution_info['Optimizer'] = [optimizer.__class__.__name__]
    execution_info['Max epochs'] = [MAX_EPOCHS]
    execution_info['Device'] = [device_str]
    execution_info['Base model'] = [BERT_MODEL]
    execution_info['Batch size'] = [BATCH_SIZE]
    execution_info['Learning rate'] = [LEARNING_RATE]
    generate_report(execution_info, predictions, output_file)

def main():
    n_args = len(argv)
    if n_args == 7:
        excel_file = argv[1]
        text_column = argv[2]
        labels_column = argv[3]
        train_test_idxs_file = argv[4]
        n_accepted_probs = int(argv[5])
        output_file = argv[6]
        train(excel_file, text_column, labels_column, train_test_idxs_file, n_accepted_probs, output_file)
    else:
        print('Usage: python3 %s <input_Excel_file> <texts_column> <labels_column> <train_test_idxs_JSON> <accepted_probabilities> <output_XLSX_file>' % (argv[0]))
        print('Example: python3 %s 20newsgroups.xlsx data target idxs_news.json 3 report.xlsx' % (argv[0]))

if __name__ == '__main__':
    main()

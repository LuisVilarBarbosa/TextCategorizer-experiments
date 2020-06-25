# Developed on June 18, 22 and 25, 2020.

import json
import pickle
import time
from itertools import zip_longest
from multiprocessing import cpu_count
from sys import argv

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn import dummy, ensemble, linear_model, metrics, svm, utils
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import BertModel, BertTokenizer

BATCH_SIZE = 32
BERT_MODEL = 'bert-base-multilingual-cased'
FREEZE_BERT = True
MAX_LENGTH = 512
PICKLE_PROTOCOL = 4
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

class FeatureExtractor:
    def __init__(self, device, net):
        self.device = device
        self.net = net.to(self.device)

    def extract_features(self, dataloader, output_config_file='embeddings.pkl', output_embeddings_file='embeddings.dat'):
        X = None
        for i, batch in enumerate(tqdm(iterable=dataloader, desc='Obtaining embeddings', unit='doc')):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, attention_mask, _labels = batch
            with torch.no_grad():
                embeddings = self.net(input_ids, attention_mask).cpu().numpy()
            if X is None:
                kwargs = {
                    'filename': output_embeddings_file,
                    'dtype': np.float32,
                    'mode': 'w+',
                    'shape': (len(dataloader.dataset), embeddings.shape[1])
                }
                X = np.memmap(**kwargs)
                kwargs['mode'] = 'r'
                dump_pickle({'numpy.memmap_kwargs': kwargs}, output_config_file)
            X[i:i+embeddings.shape[0], :] = embeddings
        return X


# Based on http://www.erogol.com/predict-probabilities-sklearn-linearsvc/
# (posted on November 14, 2014, and accessed on July 29, 2019),
# except the handling of exception 'np.AxisError' which is based on the information given in
# https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC.decision_function
# (accessed on August 11, 2019).
class LinearSVC(svm.LinearSVC):

    def __platt_func(self, x):
        return 1/(1+np.exp(-x))

    def predict_proba(self, X):
        f = np.vectorize(self.__platt_func)
        raw_predictions = self.decision_function(X)
        platt_predictions = f(raw_predictions)
        try:
            probs = platt_predictions / platt_predictions.sum(axis=1)[:, None]
        except np.AxisError:
            probs = np.asarray(list(map(lambda prob1: [1-prob1, prob1], platt_predictions)))
        return probs

def predict_proba_to_dicts(clf_classes_, y_predict_proba):
    assert len(clf_classes_) == y_predict_proba.shape[1]
    my_clf_classes_ = clf_classes_.tolist()
    my_y_predict_proba = y_predict_proba.tolist()
    my_y_predict_proba = [dict(zip_longest(my_clf_classes_, probs)) for probs in my_y_predict_proba]
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

def dump_pickle(obj, path):
    output_file = open(path, 'wb')
    pickle.dump(obj, output_file, PICKLE_PROTOCOL)
    output_file.close()

def load_pickle(path):
    input_file = open(path, 'rb')
    data = pickle.load(input_file)
    input_file.close()
    return data

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

def train(excel_file, text_column, labels_column, train_test_idxs_file, n_jobs, model_file, n_accepted_probs, output_file):
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
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, num_workers=n_jobs-1)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, num_workers=n_jobs-1)
    assert train_loader.dataset.classes_ == val_loader.dataset.classes_
    net = BERTNeuralNet(len(val_loader.dataset.classes_), freeze_bert=FREEZE_BERT)
    net.load_state_dict(torch.load(model_file, map_location=device)['model_state_dict'])
    net.additional_layers = nn.Sequential(*list(net.additional_layers.children())[0:-1])
    ft = FeatureExtractor(device, net)
    X_train = ft.extract_features(train_loader, 'X_train.pkl', 'X_train.dat')
    X_test = ft.extract_features(val_loader, 'X_test.pkl', 'X_test.dat')
    clfs = [
        ensemble.RandomForestClassifier(n_estimators=100, n_jobs=n_jobs, random_state=RANDOM_STATE),
        LinearSVC(random_state=RANDOM_STATE),
        dummy.DummyClassifier(strategy='stratified', random_state=RANDOM_STATE, constant=None),
        linear_model.SGDClassifier(loss='modified_huber', max_iter=1000, tol=1e-3, n_jobs=n_jobs, random_state=RANDOM_STATE)
    ]
    predictions = {'y_true': y_test}
    for clf in tqdm(iterable=clfs, desc='Fitting classifiers', unit='clf'):
        clf.fit(X_train, y_train)
        dump_pickle(clf, '%s.pkl' % (clf.__class__.__name__))
    for clf in tqdm(iterable=clfs, desc='Obtaining probabilities', unit='clf'):
        y_predict_proba = clf.predict_proba(X_test)
        dicts = predict_proba_to_dicts(clf.classes_, y_predict_proba)
        predictions[clf.__class__.__name__] = dicts
    dump_json(predictions, 'predictions.json')
    execution_info['End date'] = [get_local_time_str()]
    execution_info['Excel file'] = [excel_file]
    execution_info['Text column'] = [text_column]
    execution_info['Label column'] = [labels_column]
    execution_info['Accepted probabilities'] = [n_accepted_probs]
    execution_info['Device'] = [device_str]
    execution_info['Base model'] = [model_file]
    execution_info['Batch size'] = [BATCH_SIZE]
    generate_report(execution_info, predictions, output_file)

def main():
    n_args = len(argv)
    if n_args == 9:
        excel_file = argv[1]
        text_column = argv[2]
        labels_column = argv[3]
        train_test_idxs_file = argv[4]
        n_jobs = int(argv[5])
        model_file = argv[6]
        n_accepted_probs = int(argv[7])
        output_file = argv[8]
        if n_jobs < 0:
            n_jobs = cpu_count() + 1 + n_jobs
        train(excel_file, text_column, labels_column, train_test_idxs_file, n_jobs, model_file, n_accepted_probs, output_file)
    else:
        print('Usage: python3 %s <input_Excel_file> <texts_column> <labels_column> <train_test_idxs_JSON> <n_jobs> <model_file> <accepted_probabilities> <output_XLSX_file>' % (argv[0]))
        print('Example: python3 %s 20newsgroups.xlsx data target idxs_news.json -1 train_checkpoint_epoch_10.pt 3 report.xlsx' % (argv[0]))

if __name__ == '__main__':
    main()

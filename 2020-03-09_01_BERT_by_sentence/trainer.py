# Developed on February 04, 05, 18, 19, 26 and 27
# and March 02, 04, 05 and 09, 2020.

import json
import pickle
import time
from itertools import zip_longest
from string import punctuation
from sys import argv

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn import dummy, ensemble, linear_model, metrics, svm, utils
from tqdm import tqdm
from transformers import BertModel, BertTokenizer

PICKLE_PROTOCOL = 4
RANDOM_STATE = 42

def generate_statistics_chart(data_frame, filename):
    sns.boxplot(data=data_frame, orient='v', fliersize=1)
    plt.savefig(filename, dpi=600)
    plt.close()

class Preprocessor:
    def __init__(self, language='portuguese', nltk_stop_words_package='portuguese', model='bert-base-multilingual-cased'):
        nltk.download('punkt', quiet=False)
        self.language = language
        nltk.download(info_or_id='stopwords', quiet=False)
        self.stop_words = nltk.corpus.stopwords.words(nltk_stop_words_package)
        self.tokenizer = BertTokenizer.from_pretrained(model, cache_dir=model)

    def preprocess(self, corpus):
        sents_by_doc = ([], [])
        tokens_by_sent = ([], [])
        tokens_by_doc = ([], [])
        new_corpus = []
        tokens_to_ignore = set(punctuation).union(self.stop_words)
        for text in tqdm(corpus, desc='Preprocessing', unit='doc'):
            sentences = nltk.sent_tokenize(text, self.language)
            sentences = [self.tokenizer.tokenize(sent) for sent in sentences]
            sents_by_doc[0].append(len(sentences))
            temp_tokens_by_sent = [len(token_list) for token_list in sentences]
            tokens_by_sent[0].extend(temp_tokens_by_sent)
            tokens_by_doc[0].append(sum(temp_tokens_by_sent))
            new_sentences = []
            for sent in sentences:
                temp_tokens = []
                new_sent = []
                for token in sent:
                    if not token.startswith('##'):
                        if temp_tokens and self.tokenizer.convert_tokens_to_string(temp_tokens) not in tokens_to_ignore:
                            new_sent.extend(temp_tokens)
                        temp_tokens = []
                    temp_tokens.append(token.lower())
                if len(temp_tokens) > 0 and self.tokenizer.convert_tokens_to_string(temp_tokens) not in tokens_to_ignore:
                    new_sent.extend(temp_tokens)
                new_sentences.append(new_sent)
            sentences = new_sentences
            sents_by_doc[1].append(len(sentences))
            temp_tokens_by_sent = [len(token_list) for token_list in sentences]
            tokens_by_sent[1].extend(temp_tokens_by_sent)
            tokens_by_doc[1].append(sum(temp_tokens_by_sent))
            sentences = [token_list for token_list in sentences if len(token_list) > 0]
            new_corpus.append(sentences)
        filenames = {
            'Sentences by document (NLTK)': sents_by_doc,
            'Tokens by sentence (BERT)': tokens_by_sent,
            'Tokens by document (BERT)': tokens_by_doc,
        }
        for filename, data in tqdm(filenames.items(), desc='Generating charts', unit='chart'):
            df = pd.DataFrame()
            df[f'{filename}\n(before cleaning)'] = data[0]
            df[f'{filename}\n(after cleaning)'] = data[1]
            generate_statistics_chart(df, filename)
        return new_corpus


class FeatureExtractor:
    def __init__(self, model='bert-base-multilingual-cased'):
        self.tokenizer = BertTokenizer.from_pretrained(model)
        self.model = BertModel.from_pretrained(model) # WordPiece embeddings

    def extract_features(self, corpus, output_config_file='embeddings.pkl', output_embeddings_file='embeddings.dat'):
        X = None
        for i, text in enumerate(tqdm(iterable=corpus, desc='Obtaining embeddings', unit='doc')):
            sentences_embeddings = []
            for sent in text:
                sent = sent[0:510]
                input_ids = torch.tensor([self.tokenizer.encode(sent, add_special_tokens=True)])
                with torch.no_grad():
                    last_hidden_states = self.model.forward(input_ids=input_ids)[0]
                    sentence_embeddings = last_hidden_states[0, 0, :].numpy()
                sentences_embeddings.append(sentence_embeddings)
            text_embeddings = np.average(sentences_embeddings, axis=0)
            if X is None:
                kwargs = {
                    'filename': output_embeddings_file,
                    'dtype': np.float32,
                    'mode': 'w+',
                    'shape': (len(corpus), text_embeddings.shape[0])
                }
                X = np.memmap(**kwargs)
                kwargs['mode'] = 'r'
                dump_pickle({'numpy.memmap_kwargs': kwargs}, output_config_file)
            X[i, :] = text_embeddings
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

def train(excel_file, text_column, labels_column, train_test_idxs_file, n_jobs, n_accepted_probs, output_file):
    execution_info = pd.DataFrame()
    execution_info['Start date'] = [get_local_time_str()]
    df = pd.read_excel(excel_file)
    df = df.fillna('NaN')
    preprocessor = Preprocessor()
    corpus = preprocessor.preprocess(df[text_column])
    dump_json(corpus, 'preprocessed_corpus_BERT.json')
    labels = df[labels_column].tolist()
    train_test_idxs = load_json(train_test_idxs_file)
    train_idxs = train_test_idxs['train_idxs']
    test_idxs = train_test_idxs['test_idxs']
    corpus_train = utils.safe_indexing(corpus, train_idxs)
    corpus_test = utils.safe_indexing(corpus, test_idxs)
    y_train = utils.safe_indexing(labels, train_idxs)
    y_test = utils.safe_indexing(labels, test_idxs)
    ft = FeatureExtractor()
    X_train = ft.extract_features(corpus_train, 'X_train_BERT.pkl', 'X_train_BERT.dat')
    X_test = ft.extract_features(corpus_test, 'X_test_BERT.pkl', 'X_test_BERT.dat')
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
    execution_info['Excel file'] = excel_file
    execution_info['Text column'] = text_column
    execution_info['Label column'] = labels_column
    execution_info['n_jobs'] = n_jobs
    execution_info['Accepted probabilities'] = n_accepted_probs
    generate_report(execution_info, predictions, output_file)

def main():
    n_args = len(argv)
    if n_args == 8:
        excel_file = argv[1]
        text_column = argv[2]
        labels_column = argv[3]
        train_test_idxs_file = argv[4]
        n_jobs = int(argv[5])
        n_accepted_probs = int(argv[6])
        output_file = argv[7]
        train(excel_file, text_column, labels_column, train_test_idxs_file, n_jobs, n_accepted_probs, output_file)
    else:
        print('Usage: python3 %s <input_Excel_file> <texts_column> <labels_column> <train_test_idxs_JSON> <n_jobs> <accepted_probabilities> <output_XLSX_file>' % (argv[0]))
        print('Example: python3 %s 20newsgroups.xlsx data target idxs_news.json -1 3 report.xlsx' % (argv[0]))

if __name__ == '__main__':
    main()

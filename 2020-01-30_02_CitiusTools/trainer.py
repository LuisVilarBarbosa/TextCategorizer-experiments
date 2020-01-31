# Developed on January 28, 29 and 30, 2020.

import json
import os
import pickle
import subprocess
import tarfile
import time
from io import StringIO
from itertools import zip_longest
from string import punctuation
from sys import argv

import nltk
import numpy as np
import pandas as pd
import requests
from sklearn import dummy, ensemble, feature_extraction, linear_model, metrics, svm, utils
from tqdm import tqdm

PICKLE_PROTOCOL = 4
RANDOM_STATE = 42

def download_file(url, desc=None):
    local_filename = url.split('/')[-1]
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            tq = tqdm(desc=desc, total=int(r.headers['Content-Length']), unit='B', unit_scale=True)
            for chunk in r.iter_content(chunk_size=10485760):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)
                    tq.update(len(chunk))
            tq.close()
    return local_filename

class Preprocessor:
    SPLIT_TEXT_TOKEN = 'SUPPOSEDLY_UNIQUE_TOKEN'

    def __init__(self, language_code='pt', nltk_stop_words_package='portuguese'):
        if not os.path.exists('CitiusTools'):
            if not os.path.exists('CitiusTools.tar.gz'):
                url = 'http://gramatica.usc.es/pln/tools/CitiusTools/CitiusTools.tar.gz'
                download_file(url, desc='Downloading CitiusTools')
            tar = tarfile.open('CitiusTools.tar.gz')
            tar.extractall()
            tar.close()
        self.language_code = language_code
        nltk.download(info_or_id='stopwords', quiet=False)
        self.stop_words = nltk.corpus.stopwords.words(nltk_stop_words_package)

    def preprocess(self, corpus):
        corpus = self._preprocess_with_CitiusTools(corpus)
        corpus = [[t for t in token_list if not t in punctuation] for token_list in tqdm(corpus, desc='Removing punctuation', unit='doc')]
        corpus = [[t for t in token_list if not t in self.stop_words] for token_list in tqdm(corpus, desc='Removing stop words', unit='doc')]
        corpus_str = [' '.join(token_list) for token_list in tqdm(iterable=corpus, desc='Joining tokens', unit='doc')]
        return corpus, corpus_str

    def _preprocess_with_CitiusTools(self, corpus):
        pipeline = [
            'CitiusTools/{LING}/sentences-{LING}_exe.perl'.format(LING=self.language_code),
            'CitiusTools/{LING}/tokens-{LING}_exe.perl'.format(LING=self.language_code),
            'CitiusTools/{LING}/splitter-{LING}_exe.perl'.format(LING=self.language_code),
            'CitiusTools/{LING}/ner-{LING}_exe.perl'.format(LING=self.language_code),
            'CitiusTools/{LING}/tagger-{LING}_exe.perl'.format(LING=self.language_code),
            'CitiusTools/{LING}/nec-{LING}_exe.perl'.format(LING=self.language_code)
        ]
        assert all([Preprocessor.SPLIT_TEXT_TOKEN not in text for text in corpus])
        step = len(corpus) * 10000000 // sum([len(c) for c in corpus])
        chunked_corpus = [corpus[i:i+step] for i in range(0, len(corpus), step)] # Dividing in chunks to reduce CitiusTools' memory footprint.
        print('The following progress bar may appear frozen for a while.')
        tq = tqdm(desc='Preprocessing with CitiusTools', total=len(corpus), unit='doc')
        new_corpus = []
        for chunk in chunked_corpus:
            chunk = '\n{0}\n'.format(Preprocessor.SPLIT_TEXT_TOKEN).join(chunk)
            out = chunk.encode()
            for f in pipeline:
                with subprocess.Popen(['perl', f], bufsize=-1, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as p:
                    out, _err = p.communicate(out)
            partial_corpus = Preprocessor._parse_CitiusTools_output(StringIO(out.decode()))
            new_corpus.extend(partial_corpus)
            tq.update(len(partial_corpus))
        tq.close()
        return new_corpus

    @staticmethod
    def _parse_CitiusTools_output(stream):
        corpus = []
        lemmas = []
        while True:
            line = stream.readline()
            line_parts = line.split(' ')
            if len(line_parts) == 3:
                token, lemma, _pos = line_parts
                if token != Preprocessor.SPLIT_TEXT_TOKEN:
                    lemmas.append(lemma.lower())
                else:
                    corpus.append(lemmas)
                    lemmas = []
            elif line == '\n':
                pass
            elif line == '':
                corpus.append(lemmas)
                break
            else:
                raise RuntimeError('Invalid state reached.')
        return corpus

class FeatureExtractor:
    def __init__(self, training_mode, vectorizer_file='vectorizer.pkl'):
        self.training_mode = training_mode
        self.vectorizer_file = vectorizer_file
        self.vectorizer = feature_extraction.text.TfidfVectorizer(token_pattern=r'\S+') if self.training_mode else load_pickle(self.vectorizer_file)

    def extract_features(self, corpus):
        corpus = tqdm(iterable=corpus, desc='Extracting features', unit='doc')
        if self.training_mode:
            X = self.vectorizer.fit_transform(corpus)
            dump_pickle(self.vectorizer, self.vectorizer_file)
            return X
        return self.vectorizer.transform(corpus)

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
    _, corpus_str = Preprocessor().preprocess(df[text_column])
    labels = df[labels_column].tolist()
    train_test_idxs = load_json(train_test_idxs_file)
    train_idxs = train_test_idxs['train_idxs']
    test_idxs = train_test_idxs['test_idxs']
    corpus_train = utils.safe_indexing(corpus_str, train_idxs)
    corpus_test = utils.safe_indexing(corpus_str, test_idxs)
    y_train = utils.safe_indexing(labels, train_idxs)
    y_test = utils.safe_indexing(labels, test_idxs)
    X_train = FeatureExtractor(training_mode=True).extract_features(corpus_train)
    X_test = FeatureExtractor(training_mode=False).extract_features(corpus_test)
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

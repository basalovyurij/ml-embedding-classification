from itertools import groupby

import numpy as np
import torch
from annoy import AnnoyIndex
import html
from functools import lru_cache
import nltk
import pandas as pd
import pymorphy2
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer, models
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from typing import List

from torch import Tensor
from torch.utils.data import TensorDataset
from transformers import BertForSequenceClassification, Trainer, TrainingArguments

try:
    stopwords.words('russian')
except LookupError:
    nltk.download('stopwords')


try:
    nltk.sent_tokenize('test', language='russian')
except LookupError:
    nltk.download('punkt')


STOPWORDS = set(stopwords.words('russian'))
MORPH = pymorphy2.MorphAnalyzer()


@lru_cache(maxsize=1000000)
def normalize_word(w: str) -> str:
    sim_letters = [
        ('A', 'А'),
        ('a', 'а'),
        ('O', 'О'),
        ('o', 'о'),
        ('E', 'Е'),
        ('e', 'е'),
        ('T', 'Т'),
        ('H', 'Н'),
        ('P', 'Р'),
        ('p', 'р'),
        ('M', 'М'),
        ('X', 'Х'),
        ('x', 'х'),
        ('K', 'К'),
        ('B', 'В')
    ]

    for a, b in sim_letters:
        w = w.replace(a, b)

    norm_w = MORPH.parse(w.lower())[0].normal_form
    return norm_w


@lru_cache(maxsize=100000)
def get_words(sent: str) -> List[str]:
    s = html.unescape(sent).lower().strip()
    r = [normalize_word(w) for w in nltk.word_tokenize(s, language='russian', preserve_line=True)]
    r = list(filter(lambda x: x not in STOPWORDS and not x.isnumeric(), r))
    return r





df_train = pd.read_csv('train.csv', sep='\t')
df_test = pd.read_csv('test.csv', sep='\t')

x_train = df_train['text'].to_list()
x_test = df_test['text'].to_list()

le = preprocessing.LabelEncoder()
le.fit(df_train['label'].to_list() + df_test['label'].to_list())
y_train = le.transform(df_train['label'].to_list())
y_test = le.transform(df_test['label'].to_list())
print('load data')


def sklearn():
    tfidf_transformer = TfidfVectorizer(max_df=0.1, tokenizer=get_words, max_features=10000, ngram_range=(1, 1))
    x_train_tfidf = tfidf_transformer.fit_transform(x_train)
    x_test_tfidf = tfidf_transformer.transform(x_test)
    print('tfidf')

    models = [
        ('Dtree', DecisionTreeClassifier()),
        ('SVC  ', LinearSVC()),
    ]

    for name, model in models:
        model.fit(x_train_tfidf, y_train)
        y_pred = model.predict(x_test_tfidf)
        print('Test %s - accuracy: %0.3f ' % (name, accuracy_score(y_test, y_pred)))


def bert_classifier():
    model = BertForSequenceClassification.from_pretrained(
        "DeepPavlov/distilrubert-small-cased-conversational",
        num_labels=len(le.classes_))

    training_args = TrainingArguments(
        output_dir='./results',  # output directory
        num_train_epochs=3,  # total # of training epochs
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=64,  # batch size for evaluation
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir='./logs',  # directory for storing logs
    )

    # x_train1, x_eval, y_train1, y_eval = train_test_split(x_train[:10000], y_train[:10000], test_size=0.1, random_state=42)
    # train_dataset = TensorDataset(Tensor(x_train1), Tensor(y_train1))
    # eval_dataset = TensorDataset(Tensor(x_eval), Tensor(y_eval))
    #
    # def compute_metrics(pred):
    #     labels = pred.label_ids
    #     preds = pred.predictions.argmax(-1)
    #     acc = accuracy_score(labels, preds)
    #     return { 'accuracy': acc }
    #
    # trainer = Trainer(
    #     model=model,  # the instantiated Transformers model to be trained
    #     args=training_args,  # training arguments, defined above
    #     train_dataset=train_dataset,  # training dataset
    #     eval_dataset=eval_dataset,  # evaluation dataset
    #     compute_metrics=compute_metrics
    # )

    train_dataset = TensorDataset(Tensor(x_train[:10000]), Tensor(y_train[:10000]))
    eval_dataset = TensorDataset(Tensor(x_test), Tensor(y_test))

    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        acc = accuracy_score(labels, preds)
        return {'accuracy': acc}

    trainer = Trainer(
        model=model,  # the instantiated Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=eval_dataset,  # evaluation dataset
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.evaluate()


def ann_classifier():
    model = SentenceTransformer('DeepPavlov/distilrubert-small-cased-conversational')
    x_train_vectors = model.encode(x_train, show_progress_bar=True)
    x_test_vectors = model.encode(x_test, show_progress_bar=True)
    print('encode')

    # Compute PCA on the train embeddings matrix
    # pca = PCA(n_components=128)
    # pca.fit(x_train_vectors)
    # pca_comp = np.asarray(pca.components_)

    # We add a dense layer to the model, so that it will produce directly embeddings with the new size
    # dense = models.Dense(in_features=model.get_sentence_embedding_dimension(), out_features=128, bias=False,
    #                      activation_function=torch.nn.Identity())
    # dense.linear.weight = torch.nn.Parameter(torch.tensor(pca_comp))
    # model.add_module('dense', dense)

    index = AnnoyIndex(768, 'angular')
    for i in range(len(x_train_vectors)):
        index.add_item(i, x_train_vectors[i])
    index.build(10)

    y_pred = []
    for i in range(len(x_test_vectors)):
        res = index.get_nns_by_vector(x_test_vectors[i], 5)
        val = list(sorted([(key, len(list(group))) for key, group in groupby([y_train[j] for j in res], lambda x: x)], key=lambda x: -x[1]))[0][0]
        y_pred.append(val)
    print('Test ANN - accuracy: %0.3f ' % (accuracy_score(y_test, y_pred)))


sklearn()
# bert_classifier()
ann_classifier()

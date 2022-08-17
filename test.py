from itertools import groupby

from annoy import AnnoyIndex
import html
from functools import lru_cache
import nltk
import pandas as pd
import pymorphy2
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, precision_score, recall_score, balanced_accuracy_score
from sklearn.svm import LinearSVC
from typing import List


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

    model = LinearSVC()
    model.fit(x_train_tfidf, y_train)
    y_pred = model.predict(x_test_tfidf)
    print('Test SVC - accuracy: %0.3f, f1: %0.3f, recall: %0.3f, precision: %0.3f' % (
        balanced_accuracy_score(y_test, y_pred),
        f1_score(y_test, y_pred, average='macro'),
        recall_score(y_test, y_pred, average='macro'),
        precision_score(y_test, y_pred, average='macro')))


def ann_classifier():
    model = SentenceTransformer('DeepPavlov/distilrubert-small-cased-conversational')
    x_train_vectors = model.encode(x_train, show_progress_bar=True)
    x_test_vectors = model.encode(x_test, show_progress_bar=True)
    print('encode')

    index = AnnoyIndex(768, metric='angular')
    for i in range(len(x_train_vectors)):
        index.add_item(i, x_train_vectors[i])
    index.build(n_trees=500)

    y_pred = []
    for i in range(len(x_test_vectors)):
        res = index.get_nns_by_vector(x_test_vectors[i], 2)
        val = list(sorted([(key, len(list(group))) for key, group in groupby([y_train[j] for j in res], lambda x: x)], key=lambda x: -x[1]))[0][0]
        y_pred.append(val)
    print('Test ANN - accuracy: %0.3f, f1: %0.3f, recall: %0.3f, precision: %0.3f' % (
        balanced_accuracy_score(y_test, y_pred),
        f1_score(y_test, y_pred, average='macro'),
        recall_score(y_test, y_pred, average='macro'),
        precision_score(y_test, y_pred, average='macro')))


sklearn()
ann_classifier()

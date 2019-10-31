from projroot import get_labeled_dataset_path;
from sklearn.preprocessing import LabelEncoder;
from sklearn.feature_extraction.text import CountVectorizer;
from sklearn.model_selection import train_test_split;
from sklearn.naive_bayes import MultinomialNB;
from sklearn.ensemble import RandomForestClassifier;
from sklearn.neighbors import KNeighborsClassifier;
from sklearn.linear_model import LogisticRegression;
from sklearn.svm import SVC;
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline, FeatureUnion;
from sklearn.base import TransformerMixin, BaseEstimator;

import spacy;
import nltk;
import math, sys;

import numpy as np;

from sklearn.metrics import accuracy_score;
from sklearn.metrics import precision_recall_fscore_support;

from semantics.action_item_classifier_NLP import pos_tag_broader, pos_tag_granular, get_broad_chunks, get_chunks;

nlp = spacy.load("en_core_web_sm");

def read(file=None):
    lines = [];
    with open(file) as f:
        lines = f.readlines();

    labels = [];
    data = [];

    for line in lines:
        splits = line.split("\t");
        if len(splits) != 2:
            continue;

        label = splits[0];
        text = splits[1];
        labels.append(label);
        data.append(text);

    return labels,data;

def evaluateAccuracy(y_test, y_pred):
    return accuracy_score(y_true=y_test, y_pred=y_pred);

def evaluatefmeasure(y_test, y_pred):
    return precision_recall_fscore_support(y_true=y_test, y_pred=y_pred);

def pos_tag(doc=None):

    return None;

def tranform_labels(labels):
    label_encoder = LabelEncoder();
    y = label_encoder.fit_transform(labels);
    print(label_encoder.classes_);
    return y;

def transform_to_vec(model=None, doc=None):
    sparse_matrix = model.transform([doc]);
    np_arr = sparse_matrix.toarray();

    #Categorical features
    print(np_arr[0].shape);
    return np_arr[0];

def train_and_crossvalidate(labels=None, data=None):
    y = tranform_labels(labels);
    #docs_train, docs_test, y_train, y_test = train_test_split(data, y, test_size=0.2);

    clf_pipeline = Pipeline(
        [
            ("features", FeatureUnion([
                ("vec", CountVectorizer(ngram_range=(1, 2))),
                ("cat", CategoricalTransformer())
            ])),
            ("estimator", MultinomialNB())
        ]
    );


    scores = cross_validate(clf_pipeline, data, y, cv=3, scoring=("accuracy", "f1"));
    print(scores);
    print("Mean accuracy :: ", np.mean(scores["test_accuracy"]));
    print("Mean f1 :: ", np.mean(scores["test_f1"]));
    '''cv = CountVectorizer(min_df=3);
    cv.fit(docs_train);

    #X_train = cv.transform(docs_train);
    #X_test = cv.transform(docs_test);

    X_train = np.zeros((len(docs_train), len(cv.vocabulary_)));
    X_test = np.zeros((len(docs_test), len(cv.vocabulary_)));

    for index, doc in enumerate(docs_train):
        X_train[index] = transform_to_vec(model=cv, doc=doc);

    for index, doc in enumerate(docs_test):
        X_test[index] = transform_to_vec(model=cv, doc=doc);

    print(X_train.shape);
    print(X_test.shape);

    clf = MultinomialNB();
    #clf = RandomForestClassifier();
    #clf = LogisticRegression();
    #clf = KNeighborsClassifier();
    #clf = SVC();
    clf.fit(X_train, y_train);

    y_pred = clf.predict(X_test);
    y_pred_prob = clf.predict_proba(X_test);

    print(accuracy_score(y_test, y_pred));
    print(precision_recall_fscore_support(y_test, y_pred))'''
    return None;

class CategoricalTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        return None;

    def fit(self, X, y=None):
        return self;

    def transform(self, X):
        np_categorical = np.zeros((len(X), 4), dtype=np.int32);
        for index, sen in enumerate(X):
            sen = sen.rstrip("\n");

            noun_count = 0;
            verb_count = 0;
            other_count = 0;

            doc = nlp(sen);
            tags = pos_tag_broader(doc);
            tagged_sen = pos_tag_granular(doc);

            chunks = get_chunks(tagged_sen);
            broad_chunks = get_broad_chunks(tags);

            hasLingAction = False;

            for chunk in chunks:
                if type(chunk) is nltk.tree.Tree:
                    label = chunk.label();
                    if "MODAL-" in label:
                        hasLingAction = True;

            for chunk in broad_chunks:
                if type(chunk) is nltk.tree.Tree:
                    label = chunk.label();
                    if "INTJ-PHRASE" in label:
                        hasLingAction = True;

            for text, tag in tags:
                if "PROPN" in tag or "NOUN" in tag:
                    noun_count += 1;
                elif "VERB" in tag:
                    verb_count += 1;
                else:
                    other_count += 1;

            if hasLingAction:
                np_categorical[index][0] = 1;
            else:
                np_categorical[index][0] = 0;

            np_categorical[index][1] = noun_count;
            np_categorical[index][2] = verb_count;
            np_categorical[index][3] = other_count;

            #print(doc.text, np_categorical[index]);

        return np_categorical;

if __name__=="__main__":
    file = sys.argv[1];
    labels, data = read(file);
    train_and_crossvalidate(labels=labels, data=data);
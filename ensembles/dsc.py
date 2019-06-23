from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import math
import warnings
import random
from utils import minority_majority_split, minority_majority_name


class DeterministicSamplingClassifier:

    def __init__(self, base_classifier=KNeighborsClassifier(), number_of_classifiers=10, number_of_chunks=10, balance_ratio=0.45, oversampling=RandomOverSampler() ,undersampling=RandomUnderSampler(sampling_strategy='majority')):
        self.base_classifier = base_classifier
        self.number_of_classifiers = number_of_classifiers
        self.number_of_chunks = number_of_chunks
        self.undersampling = undersampling
        self.oversampling = oversampling
        self.balance_ratio = balance_ratio

        self.clf = None
        self.stored_X = []
        self.stored_y = []
        self.number_of_features = None

        self.minority_name = None
        self.majority_name = None
        self.classes = None
        self.label_encoder = None

    def partial_fit(self, X, y, classes=None):

        # ________________________________________
        # Initial preperation

        if classes is None and self.classes is None:
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(y)
            self.classes = self.label_encoder.classes
        elif self.classes is None:
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(classes)
            self.classes = classes

        if classes[0] is "positive":
            self.minority_name = self.label_encoder.transform(classes[0])
            self.majority_name = self.label_encoder.transform(classes[1])
        elif classes[1] is "positive":
            self.minority_name = self.label_encoder.transform(classes[1])
            self.majority_name = self.label_encoder.transform(classes[0])

        y = self.label_encoder.transform(y)

        if self.minority_name is None or self.majority_name is None:
            self.minority_name, self.majority_name = minority_majority_name(y)
            self.number_of_features = len(X[0])

        # ________________________________________
        # Get stored data

        new_X, new_y = [], []

        for tmp_X, tmp_y in zip(self.stored_X, self.stored_y):
            new_X.extend(tmp_X)
            new_y.extend(tmp_y)

        new_X.extend(X)
        new_y.extend(y)

        new_X = np.array(new_X)
        new_y = np.array(new_y)

        # ________________________________________
        # Undersample and store new data

        und_X, und_y = self.undersampling.fit_resample(X, y)

        self.stored_X.append(und_X)
        self.stored_y.append(und_y)

        if len(self.stored_X) > self.number_of_chunks:
                del self.stored_X[0]
                del self.stored_y[0]

        # ________________________________________
        # Oversample when below ratio

        minority, majority = minority_majority_split(new_X, new_y, self.minority_name, self.majority_name)
        ratio = len(minority)/len(majority)

        if ratio < self.balance_ratio:
            new_X, new_y = self.oversampling.fit_resample(new_X, new_y)

        # ________________________________________
        # Train classifier

        self.clf = self.base_classifier.fit(new_X, new_y)


    def predict(self, X):
        return self.clf.predict(X)


    def predict_proba(self, X):
        return self.clf.predict_proba(X)

import os
from os import listdir
from os.path import isfile, join
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ensembles import KMeanClustering
from ensembles import LearnppCDS
from ensembles import LearnppNIE
from ensembles import REA
from ensembles import OUSE
from ensembles import DeterministicSamplingClassifier
from sklearn.neural_network import MLPClassifier

from utils import ranking
from utils import evaluation
from utils import streamTools
from utils import ploting
from scipy.io import arff

from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import SMOTE

from sklearn.utils import shuffle
import numpy as np
import pandas as pd

from joblib import Parallel, delayed
import time

import warnings
import traceback
warnings.simplefilter("ignore")

def load_data(stream_name):
    data, meta = arff.loadarff("streams/%s.arff" % stream_name)
    classes = meta[meta.names()[-1]][1]
    return data, classes

def evaluate_method(classifier, stream_name, method_name, initial_size, step_size):
    print(stream_name, method_name)
    try:
        start = time.time()
        data, meta = arff.loadarff("streams/%s.arff" % stream_name)
        if data is None:
            print("Empty data")
            raise Exception

        classes = meta[meta.names()[-1]][1]
        evl = evaluation.Evaluation(classifier=classifier, stream_name="%s" % stream_name, method_name=method_name, tqdm=False)
        evl.test_and_train(data=data, classes=classes, initial_size=initial_size, step_size=step_size)
        evl.compute_metrics()
        evl.save_to_csv_metrics()
        print("End", stream_name, method_name, time.time()-start)
    except Exception as ex:
        print(str(ex))
        traceback.print_exc()
        print("Exception in ", stream_name, method_name)

cores = open('/proc/cpuinfo').read().count('processor\t:')


methods = [
           DeterministicSamplingClassifier(oversampling=SMOTE() ,undersampling=NearMiss()),
           DeterministicSamplingClassifier(),
           KMeanClustering(),
           LearnppCDS(),
           LearnppNIE(),
           REA(),
           OUSE(),
           MLPClassifier(),
          ]

names = [
         "DSC-S",
         "DSC-R",
         "KMeanClustering",
         "LearnppCDS",
         "LearnppNIE",
         "REA",
         "OUSE",
         "MLPClassifier",
        ]


step_sizes = [
              250,
              500,
              100,
              100,
              50,
              250,
              250,
              500,
              100,
              100,
              100,
              150,
              150,
              50,
              50,
              50,
              52,
              30,
              30,
              30,
              100,
              100,
              100,
              70,
              60,
              60
              ]

initial_sizes = [
                 500,
                 1000,
                 200,
                 200,
                 100,
                 500,
                 500,
                 1000,
                 200,
                 200,
                 200,
                 300,
                 300,
                 100,
                 100,
                 100,
                 52,
                 60,
                 60,
                 60,
                 200,
                 200,
                 200,
                 140,
                 120,
                 120
                ]

streams = []                                       # step init
streams += ["real/abalone-17_vs_7-8-9-10"]         # 250 500
streams += ["real/electricity-normalized"]         # 500 1000
streams += ["real/jm1"]                            # 100 200
streams += ["real/kc1"]                            # 100 200
streams += ["real/kc2"]                            # 50 100
streams += ["real/kr-vs-k-three_vs_eleven"]        # 250 500
streams += ["real/kr-vs-k-zero-one_vs_draw"]       # 250 500
streams += ["real/page-blocks0"]                   # 500 1000
streams += ["real/segment0"]                       # 100 200
streams += ["real/shuttle-c0-vs-c4"]               # 100 200
streams += ["real/vehicle0"]                       # 100 200
streams += ["real/yeast1"]                         # 150 300
streams += ["real/yeast3"]                         # 150 300
streams += ["real/wisconsin"]                      # 50 100
streams += ["real/australian"]                     # 50 100
streams += ["real/pima"]                           # 50 100
streams += ["real/heart"]                          # 52 52
streams += ["real/glass0"]                         # 30 60
streams += ["real/glass-0-1-2-3_vs_4-5-6"]         # 30 60
streams += ["real/glass1"]                         # 30 60
streams += ["real/yeast-0-2-5-7-9_vs_3-6-8"]       # 100 200
streams += ["real/vowel0"]                         # 100 200
streams += ["real/yeast-0-2-5-6_vs_3-7-8-9"]       # 100 200
streams += ["real/yeast-0-3-5-9_vs_7-8"]           # 70 140
streams += ["real/yeast-2_vs_4"]                   # 60 120
streams += ["real/yeast-0-5-6-7-9_vs_4"]           # 60 120


print("Start")
start = time.time()

Parallel(n_jobs=-1)(
    delayed(evaluate_method)(classifier, stream_name, name, initial_size, step_size)
        for classifier, name in zip(methods,names) for stream_name,initial_size,step_size in zip(streams,initial_sizes,step_sizes))


end = time.time()
print("End %f" % (end-start))

import sys
import os
from os import listdir
from os.path import isfile, join
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import significant

streams = []

directory = "gen/b30f5/"
mypath = "streams/%s" % directory
streams += ["%s%s" % (directory, os.path.splitext(f)[0]) for f in listdir(mypath) if isfile(join(mypath, f))]
directory = "gen/b20f5/"
mypath = "streams/%s" % directory
streams += ["%s%s" % (directory, os.path.splitext(f)[0]) for f in listdir(mypath) if isfile(join(mypath, f))]
directory = "gen/b10f5/"
mypath = "streams/%s" % directory
streams += ["%s%s" % (directory, os.path.splitext(f)[0]) for f in listdir(mypath) if isfile(join(mypath, f))]

directory = "gen/b30f10/"
mypath = "streams/%s" % directory
streams += ["%s%s" % (directory, os.path.splitext(f)[0]) for f in listdir(mypath) if isfile(join(mypath, f))]
directory = "gen/b20f10/"
mypath = "streams/%s" % directory
streams += ["%s%s" % (directory, os.path.splitext(f)[0]) for f in listdir(mypath) if isfile(join(mypath, f))]
directory = "gen/b10f10/"
mypath = "streams/%s" % directory
streams += ["%s%s" % (directory, os.path.splitext(f)[0]) for f in listdir(mypath) if isfile(join(mypath, f))]

# directory = "gen/features/"
# mypath = "streams/%s" % directory
# streams += ["%s%s" % (directory, os.path.splitext(f)[0]) for f in listdir(mypath) if isfile(join(mypath, f))]

# directory = "gen/balance/"
# mypath = "streams/%s" % directory
# streams += ["%s%s" % (directory, os.path.splitext(f)[0]) for f in listdir(mypath) if isfile(join(mypath, f))]


# directory = "real/"
# mypath = "streams/%s" % directory
# streams += ["%s%s" % (directory, os.path.splitext(f)[0]) for f in listdir(mypath) if isfile(join(mypath, f))]

streams.sort()

methods = [
         "DSC-R",
         "DSC-S",
         "KMeanClustering",
         "LearnppCDS",
         "LearnppNIE",
         "REA",
         "OUSE",
         "MLPClassifier",
        ]

sgf = significant.Significant(methods, streams, metrics=["auc", "gmean", "f1_score"])
sgf.test()

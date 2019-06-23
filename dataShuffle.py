import os
from os import listdir
from os.path import isfile, join
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.utils import shuffle
import numpy as np
import pandas as pd
from scipy.io import arff

directory = "real/keel/"
mypath = "streams/%s" % directory
streams = ["%s%s" % (directory, os.path.splitext(f)[0]) for f in listdir(mypath) if isfile(join(mypath, f))]

streams.sort()

for stream_name in streams:
    print(stream_name)

    data, meta = arff.loadarff("streams/%s.dat" % stream_name)
    if data is None:
        print("Empty data")
        raise Exception

    data = pd.DataFrame(data, dtype="int")
    data["Class"] = data["Class"].str.decode('ascii')
    data = data.sample(frac=1).reset_index(drop=True)
    # header = '@relation abalone19\n@attribute Sex {M, F, I}'
    data.to_csv("streams/"+stream_name+".arff", index=False, header=False)
    with open("streams/"+stream_name+".dat", "r") as file_dat:
        header = ""
        for line in file_dat:
            header += line
            if "@data" in line:
                break

        with open("streams/"+stream_name+".arff", "r+") as file:
            content = file.read()
            file.seek(0, 0)
            file.write(header + content)

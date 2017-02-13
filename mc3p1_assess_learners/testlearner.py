"""
Test a learner.  (c) 2015 Tucker Balch
"""

import numpy as np
import math
import LinRegLearner as lrl
import RTLearner as rt
import pandas as pd
import sys
import BagLearner as bl


#!!!!DEBUG Statement, set to FALSE when submitting
debug=True
def log(s):
    if debug:
        print s

if __name__=="__main__":
    if len(sys.argv) != 2:
        print "Usage: python testlearner.py <filename>"
        sys.exit(1)
    inf = open(sys.argv[1]) #open the data file we want to apply test learner too
    data = np.array([map(float,s.strip().split(',')) for s in inf.readlines()])

    # compute how much of the data is training and testing
    train_rows = math.floor(0.6* data.shape[0])
    test_rows = data.shape[0] - train_rows

    # separate out training and testing data
    trainX = data[:train_rows,0:-1]
    trainY = data[:train_rows,-1]
    testX = data[train_rows:,0:-1]
    testY = data[train_rows:,-1]

    print testX.shape
    print testY.shape

    #Code for RTLearner_test
    learner = rt.RTLearner(leaf_size=1, verbose=False)  # constructor
    learner.addEvidence(trainX, trainY)

    print "In sample results for RTLearner"
    predY = learner.query(trainX)
    rmse = math.sqrt(((trainY - predY) ** 2).sum() / trainY.shape[0])
    print "RMSE: ", rmse
    c = np.corrcoef(predY, y=trainY)
    print "corr: ", c[0, 1]

    #Get out of sample results
    print "Out of sample results for RTlearner"
    predY = learner.query(testX)  # query
    rmse = math.sqrt(((testY - predY) ** 2).sum() / testY.shape[0])
    print "RMSE: ", rmse
    c = np.corrcoef(predY, y=testY)
    print "corr: ", c[0, 1]

    #code for BagLearner
    learner = bl.BagLearner(learner = rt.RTLearner, kwargs = {"leaf_size":1}, bags = 20, boost = False, verbose = False)
    learner.addEvidence(trainX, trainY)


    print "In sample results for BagLearner"
    predY = learner.query(trainX)
    rmse = math.sqrt(((trainY - predY) ** 2).sum() / trainY.shape[0])
    print "RMSE: ", rmse
    c = np.corrcoef(predY, y=trainY)
    print "corr: ", c[0, 1]

    #Get out of sample results
    print "Out of sample results for Baglearner"
    predY = learner.query(testX)  # query
    rmse = math.sqrt(((testY - predY) ** 2).sum() / testY.shape[0])
    print "RMSE: ", rmse
    c = np.corrcoef(predY, y=testY)
    print "corr: ", c[0, 1]

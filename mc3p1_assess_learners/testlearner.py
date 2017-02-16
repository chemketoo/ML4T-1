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
    np.random.shuffle(data)


    # compute how much of the data is training and testing
    train_rows = math.floor(0.6* data.shape[0])
    test_rows = data.shape[0] - train_rows

    # separate out training and testing data
    trainX = data[:train_rows,0:-1]
    trainY = data[:train_rows,-1]
    testX = data[train_rows:,0:-1]
    testY = data[train_rows:,-1]

    log(testX.shape)
    log(testY.shape)

    '''
    #run RTLearner for cases where leaf size varies from 1 to 140
    results_in_sample = []
    results_out_sample = []  # list with result for leaf size, RMSE value in sample, RMSE value out of sample
    leaf_max=2 #any value from 2 above
    test_count_max=2 #any value from 2 above
    for i in range(1,leaf_max): #i is leaf size

        RMSE_in_sample=[]
        RMSE_out_sample=[]
    #Code for RTLearner_test
        test_count=1
        while test_count <test_count_max:

            learner = rt.RTLearner(leaf_size=i, verbose=False)  # constructor
            learner.addEvidence(trainX, trainY)

            #print "In sample results for RTLearner"
            predY = learner.query(trainX)
            rmse = math.sqrt(((trainY - predY) ** 2).sum() / trainY.shape[0])
            #print "RMSE: ", rmse
            #c = np.corrcoef(predY, y=trainY)
            #print "corr: ", c[0, 1]
            RMSE_in_sample.append(rmse)

            #Get out of sample results
            #print "Out of sample results for RTlearner"
            predY = learner.query(testX)  # query
            rmse = math.sqrt(((testY - predY) ** 2).sum() / testY.shape[0])
            #print "RMSE: ", rmse
            #c = np.corrcoef(predY, y=testY)
            #print "corr: ", c[0, 1]
            RMSE_out_sample.append(rmse)
            test_count+=1
        result_in_sample=np.mean(RMSE_in_sample)
        result_out_sample= np.mean(RMSE_out_sample)
        results_in_sample.append(result_in_sample)
        results_out_sample.append(result_out_sample)
    results={'In sample':pd.Series(results_in_sample,index=range(1,leaf_max)),
             'Out sample':pd.Series(results_out_sample,index=range(1,leaf_max))}
    results_df=pd.DataFrame(results)
    print results_df
    #results_df.to_csv('sample_results.csv')


    #print "RTLEarner analysis complete, check the file for CSVs!"
    '''


    results_in_sample = []
    results_out_sample = []  # list with result for leaf size, RMSE value in sample, RMSE value out of sample

    test_count_max = 20 # any value from 2 above, total number of bagging tests to do per datapoint
    bags_max=20 #bags go from 1 to bags_max
    for i in range(1,bags_max):

        RMSE_in_sample = []
        RMSE_out_sample = []
            # Code for RTLearner_test
        test_count = 1
        while test_count <= test_count_max:

            learner = bl.BagLearner(learner = rt.RTLearner, kwargs = {"leaf_size":20}, bags = bags_max, boost = False, verbose = False)
            learner.addEvidence(trainX, trainY)


            #log("In sample results for BagLearner")
            predY = learner.query(trainX)
            rmse = math.sqrt(((trainY - predY) ** 2).sum() / trainY.shape[0])
            #log("RMSE: ")
            #log(rmse)
            c = np.corrcoef(predY, y=trainY)
            #log("corr: ")
            #log(c[0, 1])
            RMSE_in_sample.append(rmse)

            #Get out of sample results
            #log("Out of sample results for Baglearner")
            predY = learner.query(testX)  # query
            rmse = math.sqrt(((testY - predY) ** 2).sum() / testY.shape[0])
            #log("RMSE: ")
            #log(rmse)
            c = np.corrcoef(predY, y=testY)
            #log("corr: ")
            #log(c[0, 1])
            RMSE_out_sample.append(rmse)
            test_count+=1
        log("RMSE_out_sample")
        log(RMSE_out_sample)

        result_in_sample = np.mean(RMSE_in_sample)
        result_out_sample = np.mean(RMSE_out_sample)
        results_in_sample.append(result_in_sample)
        results_out_sample.append(result_out_sample)
        log("results_out_sample")
        log(results_out_sample)
    results = {'In sample': pd.Series(results_in_sample, index=range(1, bags_max)),
               'Out sample': pd.Series(results_out_sample, index=range(1, bags_max))}
    results_df = pd.DataFrame(results)
    log(results_df)
    results_df.to_csv('bag_learner_results.csv')


    log("BagLearner analysis complete, check the file for CSVs!")
    print "All Complete check for CSVs!"



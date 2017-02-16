
#Your API should support exactly the following:
#import BagLearner as bl
#learner = bl.BagLearner(learner = rt.RTLearner, kwargs = {"leaf_size":1}, bags = 20, boost = False, verbose = False)
#learner.addEvidence(Xtrain, Ytrain)
#Y = learner.query(Xtest)

import RTLearner as rt
import LinRegLearner as lrl

import numpy as np
import pandas as pd
import random

#!!!!DEBUG Statement, set to FALSE when submitting
debug=False
def log(s):
    if debug:
        print s

#!!!initially do with bags =1, then reset to 20 once 1 works
class BagLearner(object):
    def __init__(self, learner = rt.RTLearner, kwargs = {"leaf_size":1}, bags = 20, boost = False, verbose = False):
    #!!! initialize BagLearner

        self.learner=learner
        self.kwargs=kwargs
        self.bags=bags

        if self.learner==rt.RTLearner:
                self.instancelist = [self.learner(leaf_size=self.kwargs["leaf_size"], verbose=False) for i in range(bags)]

        if self.learner==lrl.LinRegLearner:
            self.instancelist = [self.learner(verbose=False) for i in range(bags)]

    #define function to take inputs and produce a decision tree

    def addEvidence(self, trainX, trainY):
        #join the data so can track the predictions as we manipulate
        data=np.concatenate((trainX,trainY[:, None]),axis=1)
        data=pd.DataFrame(data)


        for i in range(len(self.instancelist)):
            # sample dataframe
            data = data.sample(frac=1,replace=True)
            trainX = data.iloc[:, 0:-1]
            trainY = data.iloc[:, -1]
            #add randomly selected data with replacement to the instance
            self.instancelist[i].addEvidence(trainX,trainY)


        # training step

        #test printtree


        #sample data w/replacement and build train dataframe
        #run learner on train dataframe
        #return list of learners


    # !!!use tree to classify new data points (first classify one datapoint then apply to all)
    def query(self, points):
        query_list=[]
        for i in range(len(self.instancelist)):
            Y=self.instancelist[i].query(points)
            query_list.append(Y)
        df=pd.DataFrame(query_list)
        return df.mean(axis=0)

    def author(self):
        return 'nbuckley7'  #  Georgia Tech username.



"""
A FAKE Random Tree Learner.  (c) 2016 Tucker Balch
This is just a linear regression learner implemented named RTLearner so
it is used as a template or placeholder for use by testbest4.  You should
replace this code with your own RTLearner.
"""

import numpy as np
import pandas as pd
import random

#!!!!DEBUG Statement, set to FALSE when submitting
debug=False
def log(s):
    if debug:
        print s

class RTLearner(object):
    def __init__(self, verbose=False, leaf_size=1):
    #!!! initialize tree just wit max leaf_size

        self.leaf_size=leaf_size
        self.structure=['Factor','Split/Y','Left','Right']
        self.df=pd.DataFrame(columns=self.structure)









    #!!!bring in the data
    def addEvidence(self, trainX, trainY):
        #join the data so can track the predictions as we manipulate
        data=np.concatenate((trainX,trainY[:, None]),axis=1)
        data=pd.DataFrame(data)

        class decisionnode(object):
            def __init__(self, col=None, value=None, results=None,fb=None,tb=None):
                self.col = col  # col that splits on
                self.value = value  # split value
                self.results = results
                self.tb = tb
                self.fb = fb


            def build_tree(self, data):

                if len(data)<=1 or len(np.unique(data.iloc[:, -1]))<= 1:
                    return decisionnode(results=np.mean(data.iloc[:,-1]))

                random_vals=0 #initialize to 0

                # randomly pick a random feature i of data to split on

                i = random.randrange(0, stop=data.shape[1] - 1)
                # randomly pick two unique values at that feature for the split val (average)
                if len(np.unique(data.iloc[:, i])) > 1:
                    random_vals = np.random.choice(np.unique(data.iloc[:, i]), 2, replace=False)
                # if only one unique value use that
                elif len(np.unique(data.iloc[:, i])) == 1:
                    random_vals = data.iloc[0, i]
                # get mean of those values
                split_ix=i
                split_val= np.mean(random_vals)
                if len(data)>1 and len(np.unique(data.iloc[:, -1]))>1:

                    fb = self.build_tree(data=data[data.iloc[:, split_ix] <= split_val])
                    tb = self.build_tree(data=data[data.iloc[:, split_ix] > split_val])
                    return decisionnode(col=split_ix,value=split_val,fb=fb,tb=tb)
                else:
                    log("YOU SHOULD NOT REACH HERE")



        tree=decisionnode()
        tree=tree.build_tree(data)

        def printtree(tree, indent=''):
            # Is this a leaf node?
            if tree.results != None:
                print(str(tree.results))
            else:
                print(str(tree.col) + ':' + str(tree.value) + '? ')
                # Print the branches
                print(indent + 'T->'),
                printtree(tree.tb, indent + '  ')
                print(indent + 'F->'),
                printtree(tree.fb, indent + '  ')
        printtree(tree)


        """
        @summary: Add training data to learner
        @param dataX: X values of data to add
        @param dataY: the Y training values


        # slap on 1s column so linear regression finds a constant term
        newdataX = np.ones([dataX.shape[0], dataX.shape[1] + 1])
        newdataX[:, 0:dataX.shape[1]] = dataX

        # build and save the model
        self.model_coefs, residuals, rank, s = np.linalg.lstsq(newdataX, dataY)
        """
    #!!!use tree to classify new data points (first classify one datapoint then apply to all)
    def query(self, points):
        """
        @summary: Estimate a set of test points given the model we built.
        @param points: should be a numpy array with each row corresponding to a specific query.
        @returns the estimated values according to the saved model.
        """
        # get the linear result
        ret_val = (self.model_coefs[:-1] * points).sum(axis=1) + self.model_coefs[-1]
        # add some random noise
        ret_val = ret_val + 0.09 * np.random.normal(size=ret_val.shape[0])
        return ret_val


if __name__ == "__main__":
    print "get me a shrubbery"

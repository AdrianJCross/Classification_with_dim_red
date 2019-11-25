#
# mpp.py - maximum posterior probability (MPP)
#
#    Supervised parametric learning assuming Gaussian pdf
#    with 3 cases of discriminant functions
#
#    Sample code for the Machine Learning class at UTK
#
# Hairong Qi, hqi@utk.edu
#

import numpy as np
import argparse
import sys
import util

def parse_cmdline():
    """ parse command line """
    parser = argparse.ArgumentParser(description=" MPP Demo ")
    parser.add_argument("train", metavar="Training", help="training data file")
    parser.add_argument("test", metavar="Testing", help="test data file")
    parser.add_argument(
        "case", type=int, choices=range(1, 4), help="choose from 1, 2, 3")
    return parser.parse_args()

def accuracy_score(y, y_model):
    """ return accuracy score """
    assert len(y) == len(y_model)
    return np.count_nonzero(y==y_model)/len(y)

class mpp:
    def __init__(self, case=1):
        # init prior probability, equal distribution
        # self.classn = len(self.classes)
        # self.pw = np.full(self.classn, 1/self.classn)

        # self.covs, self.means, self.covavg, self.varavg = \
        #     self.train(self.train_data, self.classes)
        self.case_ = case
        self.pw_ = None


    def fit(self, Tr, y):
        # derive the model 
        self.covs_, self.means_ = {}, {}
        self.covsum_ = None

        self.classes_ = np.unique(y)     # get unique labels as dictionary items
        self.classn_ = len(self.classes_)

        for c in self.classes_:
            arr = Tr[y == c]
            self.covs_[c] = np.cov(np.transpose(arr))
            self.means_[c] = np.mean(arr, axis=0)  # mean along rows
            if self.covsum_ is None:
                self.covsum_ = self.covs_[c]
            else:
                self.covsum_ += self.covs_[c]

        # used by case II
        self.covavg_ = self.covsum_ / self.classn_

        # used by case I
        self.varavg_ = np.sum(np.diagonal(self.covavg_)) / len(self.classes_)

    def predict(self, T):
        # eval all data 
        y = []
        disc = np.zeros(self.classn_)
        nr, _ = T.shape

        if self.pw_ is None:
            self.pw_ = np.full(self.classn_, 1 / self.classn_)

        for i in range(nr):
            for c in self.classes_:
                if self.case_ == 1:
                    edist2 = util.euc2(self.means_[c], T[i])
                    disc[c] = -edist2 / (2 * self.varavg_) + np.log(self.pw_[c])
                elif self.case_ == 2: 
                    mdist2 = util.mah2(self.means_[c], T[i], self.covavg_)
                    disc[c] = -mdist2 / 2 + np.log(self.pw_[c])
                elif self.case_ == 3:
                    mdist2 = util.mah2(self.means_[c], T[i], self.covs_[c])
                    disc[c] = -mdist2 / 2 - np.log(np.linalg.det(self.covs_[c])) / 2 \
                                   + np.log(self.pw_[c])
                else:
                    print("Can only handle case numbers 1, 2, 3.")
                    sys.exit(1)
            y.append(disc.argmax())
            
        return y

def load_data(f):
    """ Assume data format:
    feature1 feature 2 ... label 
    """

    # process training data
    data = np.genfromtxt(f)
    # return all feature columns except last
    X = data[:, :-1]
    y = data[:, -1].astype(int)

    return X, y


def main():
    args = parse_cmdline()
    Xtrain, ytrain = load_data(args.train)
    Xtest, ytest = load_data(args.test)
    model = mpp(args.case)
    model.fit(Xtrain, ytrain)
    y_model = model.predict(Xtest)
    accuracy = accuracy_score(ytest, y_model)
    print('accuracy = ', accuracy)

if __name__ == "__main__":
    main()
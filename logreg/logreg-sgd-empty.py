#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import sys

import math
import matplotlib.pyplot as plt
import numpy
import pandas
import sklearn.metrics
import sklearn.model_selection
import sklearn.linear_model
import sklearn.preprocessing


def load_train_test_data(train_ratio=.5):
    data = pandas.read_csv('./HTRU_3.csv', header=None, names=['x%i' % (i) for i in range(8)] + ['y'])
    X = numpy.asarray(data[['x%i' % (i) for i in range(8)]])
    X = numpy.hstack((numpy.ones((X.shape[0],1)), X))
    y = numpy.asarray(data['y'])

    return sklearn.model_selection.train_test_split(X, y, test_size = 1 - train_ratio, random_state=0)


def scale_features(X_train, X_test, low=0, upp=1):
    minmax_scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(low, upp)).fit(numpy.vstack((X_train, X_test)))
    X_train_scale = minmax_scaler.transform(X_train)
    X_test_scale = minmax_scaler.transform(X_test)
    return X_train_scale, X_test_scale

def sigmoid(v):
    return 1 / (1 + numpy.exp(-v))

def cross_entropy(y, y_hat):
    loss = 0
    for i in range(len(y)):
        loss += -(y[i]*math.log(y_hat[i]) + (1-y[i])*math.log(1-y_hat[i]))
    return loss


def logreg_sgd(X, y, alpha = .0001, iters = 2000, eps=1e-6,lmbda=1e-2):
    # TODO: compute theta
    print("--------------compute theta---------------------")
    n, d = X.shape
    print("N=",n,"D=",d)
    operatorOne = numpy.ones(d)
    theta = numpy.zeros(d)
    pretheta=numpy.zeros(d)
    predited_y=numpy.zeros(n)
    for i in range(iters):
        if i<n:
            print("Xi=",X[i])
            pretheta = theta
            print("i=",i,"--pretheta=",pretheta )
            predited_y[i] = sigmoid(X[i].T.dot(theta))
            print("y=",y[i],"predited_y=",predited_y[i],"y-predited_y=",y[i]-predited_y[i])
            error = X[i].T.dot(y[i]-predited_y[i])
            #print("error=",error)
            theta = pretheta + (alpha*error)-lmbda*theta
            #print("This round theta=",theta)
        else:
            i-=n
            
        #if ((theta-pretheta).T.dot(operatorOne)<=eps):
        #   print("theta<=eps!!!")
         #  break    
    print("--------------compute theta end---------------------")    
    return theta


def predict_prob(X, theta):
    return 1./(1+numpy.exp(-numpy.dot(X, theta)))


def plot_roc_curve(y_test, y_prob):
    # TODO: compute tpr and fpr of different thresholds
    tpr = []
    fpr = []
     
    for h in range(10):
        trehold=(h+1)/10
        TP=0
        FP=0
        TN=0
        FN=0
        print("trehold=",trehold)
        final_y_prob=numpy.copy(y_prob)
        for i in range(len(y_prob)):
            print("y_prob=",y_prob)
            if(final_y_prob[i]>=trehold):
                final_y_prob[i]=1
            else:
                final_y_prob[i]=0
        print("final_y_prob=---------",final_y_prob)      

        for i in range(len(y_prob)):
            if ((y_test[i])==1):
                if final_y_prob[i]==y_test[i]:
                    TP+=1
                else:
                    FN+=1
            else:
                if final_y_prob[i]==y_test[i]:
                    TN+=1
                else:
                    FP+=1
        print("TP=",TP,"FP=",FP,"TN=",TN,"FN=",FN)
        tpr.append(TP/(TP+FN))
        fpr.append(FP/(FP+TN))
    print("TPR=",tpr,"FPR=",fpr)
    plt.plot(fpr, tpr)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig("roc_curve.png")


def main(argv):
    X_train, X_test, y_train, y_test = load_train_test_data(train_ratio=.5)
    X_train_scale, X_test_scale = scale_features(X_train, X_test, 0, 1)

    theta = logreg_sgd(X_train_scale, y_train)
    print(theta)
    y_prob = predict_prob(X_train_scale, theta)
    print("Logreg train accuracy: %f" % (sklearn.metrics.accuracy_score(y_train, y_prob > .5)))
    print("Logreg train precision: %f" % (sklearn.metrics.precision_score(y_train, y_prob > .5)))
    print("Logreg train recall: %f" % (sklearn.metrics.recall_score(y_train, y_prob > .5)))
    y_prob = predict_prob(X_test_scale, theta)
    print("Logreg test accuracy: %f" % (sklearn.metrics.accuracy_score(y_test, y_prob > .5)))
    print("Logreg test precision: %f" % (sklearn.metrics.precision_score(y_test, y_prob > .5)))
    print("Logreg test recall: %f" % (sklearn.metrics.recall_score(y_test, y_prob > .5)))
    plot_roc_curve(y_test.flatten(), y_prob.flatten())


if __name__ == "__main__":
    main(sys.argv)



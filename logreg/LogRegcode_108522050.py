#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import sys

import random
import math
import matplotlib.pyplot as plt
import numpy
import pandas
import sklearn.metrics
import sklearn.model_selection
import sklearn.linear_model
import sklearn.preprocessing


def load_train_test_data(train_ratio=.5):
    data = pandas.read_csv('./HTRU_2.csv', header=None, names=['x%i' % (i) for i in range(8)] + ['y'])
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


def logreg_sgd(X, y, alpha = .0001, iters = 2000, eps=1e-4,lmbda=0.001):
    # TODO: compute theta
    print("--------------compute theta---------------------")
    n, d = X.shape
    print("N=",n,"D=",d)
    theta = numpy.zeros(d)
    pretheta=numpy.zeros(d)
    predited_y= None
    for i in range(iters):
        print("i=",i)
        temp=0
        for temp in range(n):
            #temp = random.randint(0, n - 1)
            #print("temp=",temp)
            pretheta = theta
            predited_y = sigmoid(X[temp].dot(theta))
            error = X[temp].T*(y[temp]-predited_y)-lmbda*theta
            theta = pretheta + (alpha*error)
        #epsilon = numpy.linalg.norm(theta - pretheta)
        #if epsilon < eps:
         #   return theta  
  
    print("--------------compute theta end---------------------")    
    return theta


def predict_prob(X, theta):
    return 1./(1+numpy.exp(-numpy.dot(X, theta)))


def plot_roc_curve(y_test, y_prob):
    # TODO: compute tpr and fpr of different thresholds
    tpr = []
    fpr = []  
    for trehold in numpy.arange(0, 1, 0.1):
        TP=0
        FP=0
        TN=0
        FN=0
        #print("trehold=",trehold)
        final_y_prob=numpy.copy(y_prob)
        for i in range(len(y_prob)):
            #print("y_prob=",y_prob)
            if(final_y_prob[i]>=trehold):
                final_y_prob[i]=1
                if ((y_test[i])==1):
                    TP+=1
                else:
                    FP+=1
            else:
                final_y_prob[i]=0  
                if ((y_test[i])==0):
                    TN+=1
                else:
                    FN+=1         

        #print("final_y_prob=---------",final_y_prob)  
        #print("TP=",TP,"FP=",FP,"TN=",TN,"FN=",FN)
        tpr.append(TP/(TP+FN)) #Recall
        fpr.append(FP/(FP+TN)) #Precision
    #print("TPR=",tpr,"FPR=",fpr)
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



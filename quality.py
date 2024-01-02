# -*- coding: utf-8 -*-
from __future__ import division
import math
import numpy
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy import stats
from .models import *

def showPersonRespCurve(personResponse, personParams, itemsParams, model='1PL', pointsPerBin=5, minBinWidth=0.5):

    #   it's possible to plot a curve only for 1PL model
    if model != '1PL':
        raise ValueError('Cannot plot characteristic curve for models other than 1PL.')

    #   prepare histogram
    indB = []
    indResponses = []
    for itemID in personResponse:
        indB.append(itemsParams[itemID]['b'])
        indResponses.append(personResponse[itemID])
    hist = adaptHist(indB,indResponses)
    binsB = hist['binsDataAvgX']
    binsResponseObserved = hist['binsDataAvgY']

    #   calculate expected probabilities
    binsResponseExpected = []
    for b in binsB:
        binsResponseExpected.append(prob1PL(personParams['theta'],b))

    #   prepare residuals
    stdResiduals = residuals(binsResponseObserved,binsResponseExpected,hist['binsCount'])

    #   calculate Q1 statistics
    q1,p = getQ1(stdResiduals,1)

    #   visualize
    plt.figure(num=1,figsize=(12,6))
    plt.subplot(121)    #   plot person characteristic curve
    plt.plot(binsB,binsResponseExpected) #   expected probabilities
    plt.plot(binsB,binsResponseObserved,'bo') #   observed probabilities
    plt.plot(indB,indResponses,'bx')   #   individual responses
    plt.ylabel('Probability of correct response')
    plt.xlabel('b')
    plt.title('Person characteristic curve (theta = {:.2f})'.format(personParams['theta']))
    plt.subplot(122)    #   plot standardized residuals
    plt.plot(binsB,stdResiduals,'ro')
    plt.ylabel('Standardized residual')
    plt.xlabel('theta')
    plt.title('Standardized residual plot (Q1={:.2f}, p={:.2f})'.format(q1,p))
    plt.show()

    return

def showItemRespCurve(itemResponse, itemParams, personsParams, model='1PL', pointsPerBin=5,minBinWidth=0.5):

    #   prepare histogram
    indThetas = []
    indResponses = []
    for personID in itemResponse:
        try:
            indThetas.append(personsParams[personID]['theta'])
            indResponses.append(itemResponse[personID])
        except:
            pass
    hist = adaptHist(indThetas,indResponses)
    binsTheta = hist['binsDataAvgX']
    binsResponseObserved = hist['binsDataAvgY']

    #   calculate expected probabilities
    binsResponseExpected = []
    for theta in binsTheta:
        if model == '1PL':
            prob = prob1PL(theta,itemParams['b'])
        elif model == '2PL':
            prob = prob2PL(theta,itemParams['a'],itemParams['b'])
        binsResponseExpected.append(prob)

    #   prepare residuals
    stdResiduals = residuals(binsResponseObserved,binsResponseExpected,hist['binsCount'])

    #   calculate Q1 statistics
    if model == '1PL':
        irtParamNum = 1
    elif model == '2PL':
        irtParamNum = 2
    q1,p = getQ1(stdResiduals,irtParamNum)

    #   prepare ICC figure caption
    if model == '1PL':
        itemParamsString = 'b = {:.2f}'.format(itemParams['b'])
    elif model == '2PL':
        itemParamsString = 'b = {:.2f}, a = {:.2f}'.format(itemParams['b'],itemParams['a'])

    #   visualize
    plt.figure(num=1,figsize=(12,6))
    plt.subplot(121)    #   plot item characteristic curve
    plt.plot(binsTheta,binsResponseExpected) #   expected probabilities
    plt.plot(binsTheta,binsResponseObserved,'bo') #   observed probabilities
    plt.plot(indThetas,indResponses,'bx')   #   individual responses
    plt.ylabel('Probability of correct response')
    plt.xlabel('theta')
    plt.title('Item response curve (' + itemParamsString + ')')
    plt.subplot(122)    #   plot standardized residuals
    plt.plot(binsTheta,stdResiduals,'ro')
    plt.ylabel('Standardized residual')
    plt.xlabel('theta')
    plt.title('Standardized residual plot (Q1={:.2f}, p={:.2f})'.format(q1,p))
    plt.show()

    return

#   distribute data into bins so there is garanteed amount of data points per bin,
#       additionally allows to limit minimum bin width
#   parameters:
#       data - a list of objects (any kind)
#       x - a list of floats, corresponding to objects
def adaptHist(x,y,pointsPerBin=5,minBinWidth=0.5):

    #   check if everything is OK with data
    if len(x) != len(y):
        raise ValueError('Inconsistent data. Different x and data list lengths.')
    if len(x) < pointsPerBin:
        raise ValueError('Not enough data points.')

    binsDataX = []
    binsDataY = []
    binsDataAvgX = []
    binsDataAvgY = []
    binsWidth = []
    binsCount = []

    x,y = [list(i) for i in zip(*sorted(zip(x,y), key=lambda pair: pair[0]))]
    binsDataX.append([])
    binsDataY.append([])
    pointsInCurrentBin = 0
    curBin = 0
    curBinWidth = 0
    for i in range(len(x)):
        #   check if it's time to create another bin
        if pointsInCurrentBin >= pointsPerBin and curBinWidth >= minBinWidth:
            binsWidth.append(curBinWidth)
            binsDataAvgX.append(numpy.mean(binsDataX[curBin]))
            binsDataAvgY.append(numpy.mean(binsDataY[curBin]))
            binsCount.append(pointsInCurrentBin)
            binsDataX.append([])
            binsDataY.append([])
            curBin += 1
            pointsInCurrentBin = 0
        binsDataX[curBin].append(x[i])
        binsDataY[curBin].append(y[i])
        pointsInCurrentBin += 1
        if pointsInCurrentBin <= 1:
            curBinWidth = 0
        else:
            curBinWidth = binsDataX[curBin][-1] - binsDataX[curBin][0]

    #   clean leftovers
    del binsDataX[-1]
    del binsDataY[-1]

    return {'binsDataAvgX':binsDataAvgX,'binsDataAvgY':binsDataAvgY,'binsCount':binsCount,
            'binsWidth':binsWidth,'binsDataX':binsDataX,'binsDataY':binsDataY}

#   calculate std residuals based on binned data
def residuals(observedY,expectedY,counts):
    stdResiduals = []
    for j in range (len(observedY)):
        stdResidual = (observedY[j]-expectedY[j])*math.sqrt(counts[j]/(expectedY[j]*(1-expectedY[j])))
        stdResiduals.append(stdResidual)
    return stdResiduals

#   calculate Q1-statistics based on std residuals
def getQ1(stdResiduals,irtParamNum):
    q1 = 0
    for res in stdResiduals:
        q1 += math.pow(res,2)
    p = 1 - stats.chi2.cdf(q1, len(stdResiduals) - irtParamNum)  #   p-value associated with Q1 statistic (chi-squared)
    return q1,p

#   calculate Q1-statistics for an item
def getItemQ1(itemResponse, itemParams, personsParams, model='1PL', pointsPerBin=5,minBinWidth=0.5):

    #   prepare histogram
    indThetas = []
    indResponses = []
    for personID in itemResponse:
        try:
            indThetas.append(personsParams[personID]['theta'])
            indResponses.append(itemResponse[personID])
        except:
            pass
    hist = adaptHist(indThetas,indResponses)
    binsTheta = hist['binsDataAvgX']
    binsResponseObserved = hist['binsDataAvgY']

    #   calculate expected probabilities
    binsResponseExpected = []
    for theta in binsTheta:
        if model == '1PL':
            prob = prob1PL(theta,itemParams['b'])
        elif model == '2PL':
            prob = prob2PL(theta,itemParams['a'],itemParams['b'])
        binsResponseExpected.append(prob)
        if prob == 0 or prob == 1:
            print('fuck',itemParams)

    #   prepare residuals
    stdResiduals = residuals(binsResponseObserved,binsResponseExpected,hist['binsCount'])

    #   prepare number of IRT parameters
    if model == '1PL':
        irtParamNum = 1
    elif model == '2PL':
        irtParamNum = 2

    return getQ1(stdResiduals,irtParamNum)

#   calculate Q1-statistics for a person
def getPersonQ1(personResponse, personParams, itemsParams, model='1PL', pointsPerBin=5,minBinWidth=0.5):

    #   it's possible to plot a curve only for 1PL model
    if model != '1PL':
        raise ValueError('Cannot do Q1 estimation for models other than 1PL. Sorry about that.')

    #   prepare histogram
    indB = []
    indResponses = []
    for itemID in personResponse:
        indB.append(itemsParams[itemID]['b'])
        indResponses.append(personResponse[itemID])
    hist = adaptHist(indB,indResponses)
    binsB = hist['binsDataAvgX']
    binsResponseObserved = hist['binsDataAvgY']

    #   calculate expected probabilities
    binsResponseExpected = []
    for b in binsB:
        binsResponseExpected.append(prob1PL(personParams['theta'],b))

    #   prepare residuals
    stdResiduals = residuals(binsResponseObserved,binsResponseExpected,hist['binsCount'])

    #   calculate Q1 statistics
    q1,p = getQ1(stdResiduals,1)

    return q1,p

#   show likelihood as a function of person theta
def showLikelihood(theta,personResponses,itemsParams,model):
    minTheta = -5
    maxTheta = 5
    thetaSteps = 100
    thetaArray = numpy.linspace(minTheta,maxTheta,thetaSteps)
    llArray = []
    for theta in thetaArray:
        ll = personIterLikelihood(model,theta,personResponses,itemsParams)
        llArray.append(ll)

    #   visualize
    plt.plot(thetaArray,llArray)
    plt.ylabel('Likelihood')
    plt.xlabel('theta')
    plt.title('Likelihood')
    plt.grid()
    plt.show()

def showTestInfo(itemsParams,itemSet,model):
    minTheta = -5
    maxTheta = 5
    thetaSteps = 100
    thetaArray = numpy.linspace(minTheta,maxTheta,thetaSteps)
    infoArray = []
    semArray = [] # standard error of measurement, or SD
    for theta in thetaArray:
        info = 0
        for itemID in itemSet:
            if model == '1PL':
                p = prob1PL(theta,itemsParams[itemID]['b'])
                info += p*(1-p)
            elif model == '2PL':
                p = prob2PL(theta,itemsParams[itemID]['a'],itemsParams[itemID]['b'])
                info += math.pow(itemsParams[itemID]['a'],2)*p*(1-p)
        sem = 1/math.sqrt(info)
        infoArray.append(info)
        semArray.append(sem)

    # get a list of items difficulties
    itemSetDifficulties = []
    for key in itemSet.keys():
        itemSetDifficulties.append(itemSet[key]['b'])

    #   visualize
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(thetaArray, infoArray, 'g-')
    ax2.plot(thetaArray, semArray, 'b-')

    ax2.scatter(itemSetDifficulties,[1]*len(itemSetDifficulties))

    ax1.set_xlabel(u'Словарный запас респондента, логит')
    ax1.set_ylabel(u'Информация', color='g')
    ax2.set_ylabel(u'Стандартная ошибка, логит', color='b')
    plt.xlim([-5,5])

    plt.title(u'Информация и стандартная ошибка теста')
    plt.grid()
    plt.show()

    
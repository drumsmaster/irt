# -*- coding: utf-8 -*-
from __future__ import division
import math
import numpy
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from irt.models import *
import os

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

def showItemRespCurve(itemResponse, itemParams, personsParams, model='1PL', pointsPerBin=5,minBinWidth=0.5,word=''):

    if len(itemResponse) == 0:
        print('['+word+']'+'There are no responses, cannot show item response curve')
        return
    
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

    # calculate outfit and infit
    qual = getItemQuality(itemResponse, itemParams, personsParams)

    #   prepare ICC figure caption
    if model == '1PL':
        itemParamsString = 'b = {:.2f}'.format(itemParams['b'])
    elif model == '2PL':
        itemParamsString = 'b = {:.2f}, a = {:.2f}'.format(itemParams['b'],itemParams['a'])

    # #   visualize
    # plt.figure(num=1,figsize=(20,6))
    # plt.subplot(121)    #   plot item characteristic curve
    # plt.plot(binsTheta,binsResponseExpected, label='Expected probability') #   expected probabilities
    # plt.errorbar(binsTheta,binsResponseObserved,yerr=hist['binsDataYAvgSEM'],fmt='bs', label='Observed probability') #   observed probabilities with standard error of the mean as errorbars
    # plt.plot(indThetas,indResponses,'k.', label='Individual results')   #   individual responses
    # plt.ylabel('Probability of correct response')
    # plt.xlabel('Ability, logit')
    # plt.title('['+ word + '] Item response curve (' + itemParamsString + ')')
    # plt.legend()
    # plt.subplot(122)    #   plot standardized residuals
    # plt.plot(binsTheta,stdResiduals,'ro')
    # plt.ylabel('Standardized residual')
    # plt.xlabel('Ability, logit')
    # plt.title('Standardized residual plot (Q1={:.2f}, p={:.2f}), outfit={:.2f}, infit={:.2f}'.format(q1,p,qual['outfit'],qual['infit']))
    # # plt.show()
    # base_path = '/Users/grigorygolovin/Library/CloudStorage/OneDrive-Personal/Projects/word stock estimation/Myvocab stats/item_stats_en/'
    # file_name = base_path + word + '.png'
    # print(file_name)
    # os.makedirs(base_path, exist_ok=True)
    # plt.savefig(file_name, bbox_inches='tight')
    # plt.close()
    # # end visualization section

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
    binsDataYAvgSEM = []
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
            binsDataYAvgSEM.append(numpy.std(binsDataY[curBin], ddof=1) / numpy.sqrt(numpy.size(binsDataY[curBin])))
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
            'binsWidth':binsWidth,'binsDataX':binsDataX,'binsDataY':binsDataY,'binsDataYAvgSEM':binsDataYAvgSEM}

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

# calculate item quality based on residuals: outfit and infit
# see 10.1 in https://www.edmeasurementsurveys.com/residual-based-item-fit-statistics.html#redisual-based-item-fit-statistics
# 1PL model only for now
# trim_val: Whenever a squared standardised residual is larger than trim_val, it is set to trim_val
def getItemQuality(itemResponse, itemParams, personsParams, trim_val = 10):

    ui = 0 # unweighted mean fit square, outfit
    vi = 0 # weighted mean fit square, infit
    vi_top = 0 # temparary for infit
    vi_bottom = 0 # temparary for infit

    for personID in itemResponse:
        xni = itemResponse[personID] # observed response of person n on item i
        eni = prob1PL(personsParams[personID]['theta'],itemParams['b']) # expected value of Xni
        wni = eni*(1-eni) # variance of Xni
        zni = (xni-eni)/math.sqrt(wni) # standartized residual statistic
        if zni > trim_val:
            zni = trim_val

        ui += zni*zni
        vi_top += wni*zni*zni
        vi_bottom += wni
    
    ui = ui/len(itemResponse)
    vi = vi_top/vi_bottom

    ui_sd = math.sqrt(2.0/len(itemResponse)) # asymptotic standard error for outfit
    ui_max = 1 + 2*ui_sd # top range for acceptable outfit
    ui_min = 1 - 2*ui_sd # bottom range for acceptable outfit

    return {'outfit':ui,
            'outfit_max_ac':ui_max,
            'outfit_min_ac':ui_min,
            'infit':vi,
            'person_number':len(itemResponse)}

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

    # print(getItemQuality(itemResponse, itemParams, personsParams))

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

def showTestInfo(itemsParams,
                 itemSet,
                 model='1PL',
                 minTheta=-12,
                 maxTheta=12):
    
    # parameter validation
    if model not in ['1PL','2PL']:
        raise ValueError('Model should be 1PL or 2PL')
    
    # start
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
    # ax1.plot(thetaArray, infoArray, 'g-')
    ax2.plot(thetaArray, semArray, 'b-')

    # ax2.scatter(itemSetDifficulties,[1]*len(itemSetDifficulties))
    ax1.hist(itemSetDifficulties, bins=30, alpha=0.6, color="orange", edgecolor="black")
    ax1.set_ylabel("Item number", color="orange")

    ax1.set_xlabel('Ability (logit)')
    # ax1.set_ylabel('Information', color='g')
    ax2.set_ylabel(u'Standard Error of Measurement (logit)', color='b')
    plt.xlim([minTheta,maxTheta])

    plt.title('Item bank quality')
    plt.grid()
    plt.show()

def item_person_map(persons_filepath:str,
                    items_filepath:str,
                    bins:int = 15):
    '''Display item-person map (Wright map)'''

    # Load data
    persons_df = pd.read_csv(persons_filepath,delimiter='\t')
    items_df = pd.read_csv(items_filepath,delimiter='\t')

    # Optional: filter out items with totalResponses == 0
    if 'totalResponses' in items_df.columns:
        items_df = items_df[items_df['totalResponses'] > 0]

    # Extract relevant columns
    item_b = items_df['b'].dropna()
    person_theta = persons_df['theta'].dropna()

    # Create figure and axes
    fig, (ax_left, ax_right) = plt.subplots(
        ncols=2,
        sharey=True,
        figsize=(10, 4),
        gridspec_kw={'width_ratios': [1, 1]}
    )

    # Plot left histogram (Persons) horizontally
    ax_left.hist(
        person_theta,
        bins=bins,
        orientation='horizontal',
        color='skyblue',
        edgecolor='black'
    )
    # Invert x-axis so bars grow to the left
    ax_left.invert_xaxis()

    # Show left y‑axis ticks & labels
    ax_left.tick_params(
        axis='y',
        which='both',
        left=True,
        labelleft=True
    )
    ax_left.set_xlabel('Persons')
    ax_left.set_title('Person Distribution')
    ax_left.set_ylabel('Logit Scale (Ability | Difficulty)')
    ax_left.yaxis.set_visible(True)

    # Plot right histogram (Items) horizontally
    ax_right.hist(
        item_b,
        bins=bins,
        orientation='horizontal',
        color='salmon',
        edgecolor='black'
    )

    # Show y‑axis scale on the right
    ax_right.yaxis.set_visible(False)
    ax_right.yaxis.set_label_position("right")
    ax_right.yaxis.tick_right()
    ax_right.tick_params(
        axis='y',
        which='both',
        right=True,
        left=False,
        labelright=True,
        labelleft=False
    )
    ax_right.set_ylabel('Logit Scale (Ability | Difficulty)')
    ax_right.set_xlabel('Items')
    ax_right.set_title('Item Distribution')

    # Adjust spacing and display
    plt.tight_layout()
    plt.show()

    
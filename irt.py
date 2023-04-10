# -*- coding: utf-8 -*-
from __future__ import division
import math
import numpy
import codecs
from models import *
import copy
import random
#import quality
import pprint
import sys

#   data structures
#       persons = {personID1:personResponses1,...}
#       personResponses = {itemID1:response,...}
#       response = int (0/1)
#       items = {itemID1:itemResponse1,...},...}
#       itemResponse = {personID1:response,...}
#       personsParams = {personID1:{'theta':float},...}
#       itemsParams = {itemID1:itemParams1,...}
#       itemParams = {'b':float,'bSD':float,'a':float,'aSD':float}

#   estimate person latent ability based on a set of items
def getPersonAbility(person, itemsParams, model='1PL',method='EAP',
                     thetaRange=(-5,5), thetaSteps=10,
                     priorMean = 0, priorSigma = 2):

    #   expected a posteriori (EAP) method
    if method == 'EAP':
        thetaArray = numpy.linspace(thetaRange[0], thetaRange[1], thetaSteps)
        wLLArray = numpy.zeros(len(thetaArray)) # weighted likelihood array
        for i in range(thetaSteps):
            theta = thetaArray[i]
            wLLArray[i] = personIterLikelihood(model,theta,person,itemsParams) *\
                gauss(theta,priorMean,priorSigma)
        wLLArraySum = sum(wLLArray)
        if wLLArraySum == 0:
            raise ValueError('getPersonParams: weighted likelihood array is all zeroes')
        theta = sum(thetaArray * wLLArray) / wLLArraySum
        thetaSD = math.sqrt(sum(pow(thetaArray-theta,2) * wLLArray) / wLLArraySum)
        return theta,thetaSD

    #
    if method == 'MAP':
        pass

#   estimate item parameters based on a set of persons responses
def getItemParams(item, personsParams, method='EAP', model='1PL',
                  itemRange={'a':(0, 2), 'b':(-5, 5)}, itemSteps={'a':5, 'b':10},
                  priorMean={'a':1,'b':0}, priorSigma={'a':1,'b':2}):

    #   expected a posteriori (EAP) method
    if method == 'EAP':
        bArray = numpy.linspace(itemRange['b'][0], itemRange['b'][1], itemSteps['b'])
        if model == '1PL':
            wLLArray = numpy.zeros(itemSteps['b'])
            for i in range(itemSteps['b']):
                b = bArray[i]
                wLLArray[i] = itemIterLikelihood(model,{'b':b},item,personsParams) *\
                    gauss(b, priorMean['b'], priorSigma['b'])
            wLLArraySum = sum(wLLArray)
            if wLLArraySum == 0:
                raise ValueError('getItem params: weighted likelihood array is all zeroes')
            b = sum(bArray * wLLArray) / wLLArraySum
            bSD = math.sqrt(sum(pow(bArray-b,2) * wLLArray) / wLLArraySum)
            return {'b':b,'bSD':bSD,'a':1.,'aSD':0.}
        if model == '2PL':
            bArray.shape = (len(bArray),1)
            aArray = numpy.linspace(itemRange['a'][0], itemRange['a'][1], itemSteps['a'])
            aArray.shape = (len(aArray),1)
            aArray.shape = (1,len(aArray))
            wLLArray = numpy.zeros((itemSteps['b'],itemSteps['a']))
            for bi in range(itemSteps['b']):
                for ai in range(itemSteps['a']):
                    b = bArray[bi][0]
                    a = aArray[0][ai]
                    wLLArray[bi][ai] = itemIterLikelihood(model,{'a':a,'b':b},item,personsParams) *\
                                       gauss(b, priorMean['b'], priorSigma['b']) *\
                                       gauss(a, priorMean['a'], priorSigma['a'])
            wLLArraySum = sum(sum(wLLArray))
            b = sum(sum(bArray * wLLArray)) / wLLArraySum
            a = sum(sum(aArray * wLLArray)) / wLLArraySum
            bSD = math.sqrt( sum(sum(pow(bArray-b,2) * wLLArray)) / wLLArraySum )
            aSD = math.sqrt( sum(sum(pow(aArray-a,2) * wLLArray)) / wLLArraySum )
            return {'a':a,'aSD':aSD,'b':b,'bSD':bSD}

#   read a table into two dictionaries: persons and items
def readTable(filename,correctSymbol='1',incorrectSymbol='0',linesNum=-1):

    #   fill persons dict
    persons = {}
    f = codecs.open(filename,encoding='utf-8')
    lines = f.readlines()
    itemsIDs = lines[0].strip().split('\t')[1:]

    #   determine how much to read
    if linesNum <= -1 or (linesNum > len(lines)-1):
        linesNum = len(lines) - 1

    for line in lines[1:linesNum+1]:
        cells = line.strip().split('\t')
        personId = cells[0]
        responseData = {}
        for i in range(1,len(cells)):
            if cells[i] == correctSymbol:
                response = 1
            elif cells[i] == incorrectSymbol:
                response = 0
            else:
                continue
            responseData[itemsIDs[i-1]] = response
        persons[personId] = responseData
    f.close()

    #   fill items dict
    items = {}
    for itemID in itemsIDs:
        items[itemID] = {}
    for personID in persons.keys():
        for itemID in persons[personID].keys():
            items[itemID][personID] = persons[personID][itemID]

    return persons, items

#   estimate person abilities and item parameters jointly
#   joint maximum likelihood
def getItemPersonJointParams(persons, items, model='1PL', ranges={'a':(0.1,3),'b':(-6,6),'theta':(-6,6)},
                             steps={'theta':20,'a':20,'b':20}, precisions={'theta':0.05,'a':0.05,'b':0.05},
                             priorMean={'a':1.7,'b':0,'theta':0}, priorSigma={'a':1,'b':2,'theta':2},
                             minRanges={'theta':2,'a':1,'b':2},skip=True,itemsInitialGuess={},personsInitialGuess={}):

    #   prepare persons parameters initial guess
    personsParams = {}
    badPersons = [] #   persons with no positive or negative responses
    for personID in persons.keys():
        correctResponses = sum(persons[personID].values())
        totalResponses = len(persons[personID].values())
        if correctResponses == 0:
            theta = ranges['theta'][0]
            badPersons.append(personID)
        elif correctResponses == totalResponses:
            theta = ranges['theta'][1]
            badPersons.append(personID)
        else:
            theta = math.log(correctResponses/(totalResponses-correctResponses))
            if personID in personsInitialGuess.keys():
                theta = personsInitialGuess[personID]['theta']
        personsParams[personID] = {'theta':theta,'thetaSD':priorSigma['theta'],'converged?':False}

    #   delete bad persons
    for personID in badPersons:
        delPerson(personID,persons,items,personsParams)
    print 'Check for persons with negative/positive responses only: {:d} persons deleted'.format(len(badPersons))

    # prepare item parameters initial guess
    itemsParams = {}
    badItems = {}   #   items with no positive or negative responses
    for itemID in items.keys():
        correctResponses = sum(items[itemID].values())
        totalResponses = len(items[itemID].values())
        if correctResponses == totalResponses:
            b = ranges['b'][0]
            badItems[itemID] = {}
        elif correctResponses == 0:
            b = ranges['b'][1]
            badItems[itemID] = {}
        else:
            b = math.log((totalResponses-correctResponses)/correctResponses)
        if itemID in itemsInitialGuess.keys():
            b = itemsInitialGuess[itemID]['b']
        itemsParams[itemID] = {'b':b,'bSD':priorSigma['b'],'a':priorMean['a'],
                               'aSD':priorSigma['a'],'converged?':False,
                               'correctResponses':correctResponses,'totalResponses':totalResponses}

    #   delete bad items
    for itemID in badItems.keys():
        badItems[itemID] = itemsParams[itemID]
        delItem(itemID, persons, items, itemsParams)
    print 'Check for items with negative/positive responses only: {:d} items deleted'.format(len(badItems.keys()))

    # iterate until convergence is met
    iterFull = 1    #   loops with no elements skipped
    iterSub = 1     #   loops with some elements skipped
    while True:
        print 'Iteration # {}/{}'.format(iterFull,iterSub)
        maxThetaDif = 0
        maxBDif = 0
        maxADif = 0
        skipped = 0

        #   assume items parameters are fixed, estimate persons parameters
        for personID in personsParams:
            if personsParams[personID]['converged?'] and skip:
                skipped += 1
                continue
            thetaOld = personsParams[personID]['theta']
            thetaOldSD = personsParams[personID]['thetaSD']
            thetaRange = max([thetaOldSD*6,minRanges['theta']])
            try:
                theta,thetaSD = getPersonAbility(persons[personID], itemsParams, model, method='EAP',
                                                 thetaRange=(thetaOld-thetaRange/2, thetaOld+thetaRange/2),
                                                 thetaSteps=steps['theta'],
                                                 priorMean=thetaOld)
            except ValueError:
                'Person error',personID
            thetaDif = math.fabs(thetaOld - theta)
            if thetaDif > maxThetaDif:
                maxThetaDif = thetaDif
            personsParams[personID]['theta'] = theta
            personsParams[personID]['thetaSD'] = thetaSD
            if thetaDif < precisions['theta']:
                personsParams[personID]['converged?'] = True

        #   assume persons parameters are fixed, estimate items parameters
        for itemID in itemsParams.keys():
            if itemsParams[itemID]['converged?'] and skip:
                skipped += 1
                continue
            itemParamsOld = copy.deepcopy(itemsParams[itemID])
            aRange = max([itemParamsOld['aSD']*6,minRanges['a']])
            bRange = max([itemParamsOld['bSD']*6,minRanges['b']])
            itemRange = {'a':(itemParamsOld['a']-aRange/2,itemParamsOld['a']+aRange/2),
                         'b':(itemParamsOld['b']-bRange/2,itemParamsOld['b']+bRange/2)}
            try:
                upd(itemsParams[itemID],
                    getItemParams(items[itemID], personsParams, method='EAP',
                                  model=model,itemRange=itemRange,itemSteps=steps,
                                  priorMean=itemParamsOld,priorSigma=priorSigma))
            except ValueError:
                'Item error',itemID

            #   update maximum differences
            bDif = math.fabs(itemParamsOld['b'] - itemsParams[itemID]['b'])
            if bDif > maxBDif:
                maxBDif = bDif
            if model == '2PL':
                aDif = math.fabs(itemParamsOld['a'] - itemsParams[itemID]['a'])
                if aDif > maxADif:
                    maxADif = aDif
            elif model == '1PL':
                aDif = 0

            #   if an item params do not change much, exclude it from further calculations
            if aDif < precisions['a'] and bDif < precisions['b']:
                itemsParams[itemID]['converged?'] = True

        #   if everything skipped, reset 'converged' field to iterate all elements again
        if (skipped == len(items) + len(persons)) and skip:
            for personID in persons.keys():
                personsParams[personID]['converged?'] = False
            for itemID in items.keys():
                itemsParams[itemID]['converged?'] = False
            iterFull += 1
            iterSub = 1
        elif (skipped != len(items) + len(persons)) and skip:
            iterSub += 1
        if skip == False:
            iterFull += 1
            iterSub = 1

        #   check convergence
        print 'max Theta dif = {:.2f}\tmax b dif = {:.2f}\tmax a dif = {:.2f}\tSkipped: {:d}/{:d}'.format(
                maxThetaDif,maxBDif,maxADif,skipped,len(items)+len(persons))
        if maxThetaDif < precisions['theta'] and maxBDif < precisions['b']\
                and maxADif < precisions['a'] and skipped == 0:
            status = 'Precision level reached'
            break

    #   rescale
    thetaArray = [personsParams[i]['theta'] for i in personsParams]
    thetaMean = numpy.mean(thetaArray)
    thetaSD = numpy.std(thetaArray)
    print 'theta mean = {:.2f}\ttheta SD = {:.2f}'.format(thetaMean,thetaSD)
    if model == '1PL':
        thetaSD = 1
    for personID in personsParams:
        personsParams[personID]['theta'] = (personsParams[personID]['theta'] - thetaMean)/thetaSD
    for itemID in itemsParams:
        itemsParams[itemID]['b'] = (itemsParams[itemID]['b']-thetaMean)/thetaSD
        itemsParams[itemID]['a'] *= thetaSD
    print 'Rescaled to thetaMean = 0'

    return personsParams,itemsParams,badItems,status

#   update values of dict1 with values from dict2
def upd(dict1,dict2):
    for key in dict2:
        dict1[key] = dict2[key]
    return dict1

#   delete an item
def delItem(itemID, personsResponses, itemsResponses, itemsParams):
    if itemID in itemsParams:
        del itemsParams[itemID]
    if itemID in itemsResponses:
        del itemsResponses[itemID]
    for personID in personsResponses:
        if itemID in personsResponses[personID]:
            del personsResponses[personID][itemID]

#   delete a person
def delPerson(personID, personsResponses, itemsResponses, personsParams):
    if personID in personsParams:
        del personsParams[personID]
    if personID in personsResponses:
        del personsResponses[personID]
    for itemID in itemsResponses:
        if personID in itemsResponses[itemID]:
            del itemsResponses[itemID][personID]

#   simulate person response
def simulateUserResponse(theta,itemParams,model):
    if model == '1PL':
        p = prob1PL(theta,itemParams['b'])
    if model == '2PL':
        p = prob2PL(theta,itemParams['a'],itemParams['b'])
    if random.random() <= p:
        response = 1
    else:
        response = 0
    return response
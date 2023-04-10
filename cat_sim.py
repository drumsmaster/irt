# -*- coding: utf-8 -*-
import irt,quality,models
import codecs
import random
import numpy
import matplotlib.pyplot as plt
import timeit

#   simulates cat process

thetaTrue = -1
thetaEst = 0
maxIterations = 200
precisionGoal = 0.3
model = '1PL'
thetaStepsEAP = 100
itemsParamsFilename = 'items50000_a1.txt'
priorSigma = 2
thetaRand = 0.1

#   read items params
itemsParams = {}
f = codecs.open(itemsParamsFilename, encoding='utf-8')
lines = f.readlines()
for line in lines[1:]:
    values = line.strip().split('\t')
    itemsParams[values[0]] = {'b':float(values[2])}
f.close()

#   estimate person ability based on full set of items
personResponses = {}
for itemID in itemsParams:
    response = irt.simulateUserResponse(thetaTrue,itemsParams[itemID],model)
    personResponses[itemID] = response

# thetaEst,thetaEstSD = irt.getPersonAbility(personResponses,itemsParams,
#                                                model=model,thetaSteps=thetaStepsEAP)
# print 'Person ability estimated in the full item set: theta = {:.2f}, thetaSD = {:.2f}'.format(thetaEst,thetaEstSD)

#   calculate information
itemsInfo = []
for itemID in itemsParams:
    itemsInfo.append({'itemID':itemID,'b':itemsParams[itemID]['b']})
iter = 1
personResponses = {}
administeredItemIDs = []
iterArray = []
thetaEstArray = []
thetaEstSDArray = []
while iter <= maxIterations:

    #   choose a question
    itemsInfo.sort(key=lambda x:abs(x['b']-thetaEst+random.uniform(-thetaRand,thetaRand)))
    newItemID = ''
    for itemInfo in itemsInfo:
        if itemInfo['itemID'] not in administeredItemIDs:
            newItemID = itemInfo['itemID']
            break
    if newItemID == '':
        print 'Error: item pool is empty'
        exit()
    newItemParams = itemsParams[newItemID]
    #print thetaEst,newItemParams['b']

    #   administer the question
    response = irt.simulateUserResponse(thetaTrue,newItemParams,model)
    personResponses[newItemID] = response
    administeredItemIDs.append(newItemID)
    #print thetaEst,newItemParams['b'],newItemID,newItemParams,response

    #   estimate ability
    newTheta,newThetaSD = irt.getPersonAbility(personResponses,itemsParams,model=model,thetaSteps=thetaStepsEAP,
                         priorMean=thetaEst,priorSigma=priorSigma)
    #quality.showLikelihood(newTheta,personResponses,itemsParams,model)
    thetaEst = newTheta

    #   save results for analysis
    iterArray.append(iter)
    thetaEstArray.append(thetaEst)
    thetaEstSDArray.append(newThetaSD)
    #print 'Iteration {:d}. Theta: {:.2f}, thetaSD: {:.2f}'.format(iter,newTheta,newThetaSD)
    if newThetaSD <= precisionGoal:
        print 'Precision goal reached!'
        break

    iter += 1

print 'After {:d} iterations, estimated theta: {:.2f}, SD: {:.2f}'.format(iterArray[-1],newTheta,newThetaSD)

#   visualize
plt.errorbar(iterArray,thetaEstArray,yerr=thetaEstSDArray,fmt='ks')
plt.plot(iterArray,numpy.ones(len(iterArray))*thetaTrue)
plt.ylabel('Estimated theta')
plt.xlabel('Iteration number')
plt.title('CAT')
plt.grid()
plt.show()

#quality.showTestInfo(itemsParams,administeredItemIDs,model)
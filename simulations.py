# -*- coding: utf-8 -*-
import irt,quality,models
import codecs
import random
import numpy
import matplotlib.pyplot as plt
import timeit

#   simulate user response to study quality of item set
#   plots estimated theta vs real ones
itemsParamsFilename = 'items_est.txt' #'items50000_a1.txt'
itemSetFilename = 'items_est.txt'
simNumPerTheta = 100
model = '1PL'
thetaStepsEAP = 50
thetaArray = numpy.linspace(-4,4,10)

#   read items params
itemsParams = {}
f = codecs.open(itemsParamsFilename, encoding='utf-8')
lines = f.readlines()
for line in lines[1:]:
    values = line.strip().split('\t')
    itemsParams[values[0]] = {'b':float(values[1])}
f.close()

#   read item set
itemSet = []
f = codecs.open(itemSetFilename, encoding='utf-8')
lines = f.readlines()
for line in lines[1:]:
    values = line.strip().split('\t')
    itemSet.append(values[0])
f.close()

#   check that all needed item parameters available
for itemID in itemSet:
    if itemID not in itemsParams:
        print 'Item missing: ' + itemID

#   estimate person ability
thetaEstArray = []
thetaSDEstArray = []
for theta in thetaArray:
    tempThetaEstArray = []
    for i in range(0, simNumPerTheta):
        personResponses = {}
        for itemID in itemSet:
            try:
                response = irt.simulateUserResponse(theta,itemsParams[itemID],model)
            except:
                continue
            personResponses[itemID] = response
        thetaEst,thetaEstSD = irt.getPersonAbility(personResponses,itemsParams,thetaRange=(-6,6),
                                                   model=model,thetaSteps=thetaStepsEAP)
        tempThetaEstArray.append(thetaEst)
    thetaEstArray.append(numpy.mean(tempThetaEstArray))
    thetaSDEstArray.append(numpy.std(tempThetaEstArray))

#   visualize
plt.errorbar(thetaArray,thetaEstArray,yerr=thetaSDEstArray,fmt='ks')
plt.plot(thetaArray,thetaArray)
plt.ylabel('Estimated theta')
plt.xlabel('True theta')
plt.title('Test validity')
plt.grid()
plt.show()

quality.showTestInfo(itemsParams,itemSet,model)
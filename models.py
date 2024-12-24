# -*- coding: utf-8 -*-
from __future__ import division
import math

#   probabilities, likelihoods, and other supplementary functions

#   gaussian distribution
def gauss(x,mean,sigma):
    return 1./math.sqrt(2*math.pi)*math.exp(-math.pow((x-mean)/sigma,2)/2)

# def logGauss(x,mean,sigma):
#     return

#   1-parameter logit function
def prob1PL(theta,b):
    d = 1. #   log/ogive coefficient
    try:
        p = 1/(1+math.exp(d*(b-theta)))
    except:
        print(type(b),type(theta))
    return p

#   2-parameter logit function
def prob2PL(theta,a,b):
    d = 1. #   log/ogive coefficient
    return 1/(1+math.exp(d*a*(b-theta)))

def likelihood(model,theta,itemParams,response):

    if model == '1PL':
        p = prob1PL(theta,itemParams['b'])
    elif model == '2PL':
        p = prob2PL(theta,itemParams['a'],itemParams['b'])

    if response == 1:
        ll = p
    else:
        ll = 1 - p

    return ll

def logLL(model,theta,itemParams,response):

    if model == '1PL':
        p = prob1PL(theta,itemParams['b'])
    elif model == '2PL':
        p = prob2PL(theta,itemParams['a'],itemParams['b'])

    if response == 1:
        ll = math.log(p)
    else:
        ll = math.log(1 - p)

    return ll

#   iterate through items in a person's response
def personIterLikelihood(model,theta,personResponses,itemsParams):
    ll = 1.
    for itemID in personResponses.keys():
        ll *= likelihood(model,theta,itemsParams[itemID],personResponses[itemID])
    return ll

#   iterate through persons in an item's response
def itemIterLikelihood(model, itemParams, itemResponses, personsParams):
    ll = 1.
    for personID in itemResponses.keys():
        ll *= likelihood(model, personsParams[personID]['theta'],
                         itemParams, itemResponses[personID])
    return ll

# def person1PLLogLikelihood(theta, personResponse, itemsParams):
#     ll = 0.
#     for itemID in personResponse.keys():
#         response = personResponse[itemID]
#         b = itemsParams[itemID]['b']
#         p = prob1PL(theta,b)
#         ll += response*math.log(p) + (1-response)*math.log(1-p)
#     return ll
#
# def item1PLLogLikelihood(b, itemResponse, personsParams):
#     #   optimize power
#     ll = 0.
#     for personID in itemResponse.keys():
#         response = itemResponse[personID]
#         theta = personsParams[personID]['theta']
#         p = prob1PL(theta,b)
#         ll += response*math.log(p) + (1-response)*math.log(1-p)
#     return ll
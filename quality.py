# -*- coding: utf-8 -*-
from __future__ import division
import math
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from irt.models import *
import os
import pprint
from typing import Tuple, Dict, Any

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

def showItemRespCurve(itemResponse, itemParams, personsParams, base_path, model='1PL', pointsPerBin=5,minBinWidth=0.5,word=''):

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

    #   visualize
    plt.figure(num=1,figsize=(20,6))
    plt.subplot(121)    #   plot item characteristic curve
    plt.plot(binsTheta,binsResponseExpected, label='Expected probability') #   expected probabilities
    plt.errorbar(binsTheta,binsResponseObserved,yerr=hist['binsDataYAvgSEM'],fmt='bs', label='Observed probability') #   observed probabilities with standard error of the mean as errorbars
    plt.plot(indThetas,indResponses,'k.', label='Individual results')   #   individual responses
    plt.ylabel('Probability of correct response')
    plt.xlabel('Ability, logit')
    plt.title('['+ word + '] Item response curve (' + itemParamsString + ')')
    plt.legend()
    plt.subplot(122)    #   plot standardized residuals
    plt.plot(binsTheta,stdResiduals,'ro')
    plt.ylabel('Standardized residual')
    plt.xlabel('Ability, logit')
    plt.title('Standardized residual plot (Q1={:.2f}, p={:.2f}), outfit={:.2f}, infit={:.2f}'.format(q1,p,qual['outfit'],qual['infit']))
    # plt.show()
    file_name = base_path + word + '.png'
    # print(file_name)
    os.makedirs(base_path, exist_ok=True)
    plt.savefig(file_name, bbox_inches='tight')
    plt.close()
    # end visualization section

    return


def adaptHist(x,y,pointsPerBin=5,minBinWidth=0.5):
    '''
    distribute data into bins so there is garanteed amount of data points per bin, 
    additionally allows to limit minimum bin width
    '''

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
            binsDataAvgX.append(np.mean(binsDataX[curBin]))
            binsDataAvgY.append(np.mean(binsDataY[curBin]))
            binsDataYAvgSEM.append(np.std(binsDataY[curBin], ddof=1) / np.sqrt(np.size(binsDataY[curBin])))
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


def residuals(observedY,expectedY,counts):
    '''
    Calculate std residuals based on binned data
    '''
    stdResiduals = []
    for j in range (len(observedY)):
        stdResidual = (observedY[j]-expectedY[j])*math.sqrt(counts[j]/(expectedY[j]*(1-expectedY[j])))
        stdResiduals.append(stdResidual)
    return stdResiduals


def getQ1(stdResiduals,irtParamNum):
    '''
    Calculate Q1-statistics based on std residuals
    '''

    q1 = 0
    for res in stdResiduals:
        q1 += math.pow(res,2)
    p = 1 - stats.chi2.cdf(q1, len(stdResiduals) - irtParamNum)  #   p-value associated with Q1 statistic (chi-squared)
    return q1,p


def getItemQuality(itemResponse, itemParams, personsParams, trim_val = 10):
    '''
    Calculate item quality based on residuals: outfit and infit
    see 10.1 in https://www.edmeasurementsurveys.com/residual-based-item-fit-statistics.html#redisual-based-item-fit-statistics
    1PL model only for now
    
    trim_val: Whenever a squared standardised residual is larger than trim_val, it is set to trim_val
    '''

    if len(itemResponse) == 0 or ('b' not in itemParams):  # no responses or no difficulty. It is impossible to estimate item quality
        return {'outfit':float('nan'),
                'infit':float('nan'),
                'outfit_min_ac':float('nan'),
                'outfit_max_ac':float('nan')}

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


def getPersonQuality(personResponse, personParams, itemsParams, trim_val = 10):
    '''
    Calculate person fit quality based on residuals: outfit and infit
    see 10.1 in https://www.edmeasurementsurveys.com/residual-based-item-fit-statistics.html#redisual-based-item-fit-statistics
    1PL model only for now
    
    trim_val: Whenever a squared standardised residual is larger than trim_val, it is set to trim_val
    '''

    if len(personResponse) == 0:  # no responses. It is impossible to estimate person quality
        return {'outfit':float('nan'),
                'infit':float('nan'),
                'outfit_min_ac':float('nan'),
                'outfit_max_ac':float('nan'),
                'total_responses':0}

    ui = 0 # unweighted mean fit square, outfit
    vi = 0 # weighted mean fit square, infit
    vi_top = 0 # temparary for infit
    vi_bottom = 0 # temparary for infit

    for itemID in personResponse:
        if itemID not in itemsParams:
            continue
        xni = personResponse[itemID] # observed response of item n on person i
        eni = prob1PL(personParams['theta'],itemsParams[itemID]['b']) # expected value of Xni
        wni = eni*(1-eni) # variance of Xni
        zni = (xni-eni)/math.sqrt(wni) # standartized residual statistic
        if zni > trim_val:
            zni = trim_val

        ui += zni*zni
        vi_top += wni*zni*zni
        vi_bottom += wni
    
    ui = ui/len(personResponse)
    vi = vi_top/vi_bottom

    ui_sd = math.sqrt(2.0/len(personResponse)) # asymptotic standard error for outfit
    ui_max = 1 + 2*ui_sd # top range for acceptable outfit
    ui_min = 1 - 2*ui_sd # bottom range for acceptable outfit

    return {'outfit':ui,
            'outfit_max_ac':ui_max,
            'outfit_min_ac':ui_min,
            'infit':vi,
            'total_responses':len(personResponse)}


def getItemQ1(itemResponse, itemParams, personsParams, model='1PL', pointsPerBin=5,minBinWidth=0.5):
    '''
    Calculate Q1-statistics for an item
    '''

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


def getPersonQ1(personResponse, personParams, itemsParams, model='1PL', pointsPerBin=5,minBinWidth=0.5):
    '''
    Calculate Q1-statistics for a person
    '''

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


def showLikelihood(theta,personResponses,itemsParams,model):
    '''
    Show likelihood as a function of person theta
    '''

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


def calculateItemBankInformation(itemsParams,
                 itemSet,
                 model='1PL',
                 minTheta: float = -12,
                 maxTheta: float = 12,
                 thetaSteps: int = 100):
    '''
    Calculate information and Standard Error of Measurement for a set of items
    
    itemsParams: A dictionary with bank items
    itemSet: A list with a subset of items we are interested with (other items in the bank will be ignored)
    model: '1PL' or '2PL'
    minTheta, maxTheta, thetaSteps: specifies theta axis. Information and SEM will be returned for that axis,
    '''
    
    # parameter validation
    if model not in ['1PL','2PL']:
        raise ValueError('Model should be 1PL or 2PL')
    
    # start
    thetaArray = np.linspace(minTheta,maxTheta,thetaSteps)
    infoArray = []
    semArray = [] # standard error of measurement, or SEM
    for theta in thetaArray:
        info = 0
        for itemID in itemSet:
            if itemsParams[itemID]['type'] not in ['yn','mc']:
                continue    # make sure only yn and mc questions contribute
            if (not np.isfinite(itemsParams[itemID]['b'])) | np.isnan(itemsParams[itemID]['b']):
                continue    # there is no valid estimate of item difficulty
            if model == '1PL':
                p = prob1PL(theta,itemsParams[itemID]['b'])
                info += p*(1-p)
            elif model == '2PL':
                p = prob2PL(theta,itemsParams[itemID]['a'],itemsParams[itemID]['b'])
                info += math.pow(itemsParams[itemID]['a'],2)*p*(1-p)
        sem = 1/math.sqrt(info)
        infoArray.append(info)
        semArray.append(sem)

    return (thetaArray,infoArray,semArray)


def showTestInfo(itemsParams,
                 itemSet,
                 model='1PL',
                 minTheta: float = -12,
                 maxTheta: float = 12,
                 thetaSteps: int = 100):
    
    thetaArray,infoArray,semArray = calculateItemBankInformation(itemsParams,itemSet,model,minTheta,maxTheta,thetaSteps)

    # get a list of items difficulties
    itemSetDifficulties = []
    for item in itemSet:
        if itemsParams[item]['type'] not in ['yn','mc']:
                continue    # make sure only yn and mc questions contribute
        itemSetDifficulties.append(itemSet[item]['b'])

    #   visualize
    fig, ax_hist = plt.subplots()

    # Right y-axis (SEM)
    ax_sem = ax_hist.twinx()

    # Second LEFT y-axis (Information)
    ax_info = ax_hist.twinx()
    ax_info.spines["right"].set_visible(False)
    ax_info.spines["left"].set_position(("outward", 60))
    ax_info.yaxis.set_label_position("left")
    ax_info.yaxis.set_ticks_position("left")
    ax_info.set_ylim(0, max(infoArray) * 1.05)

    # ---- Plotting ----

    # Histogram (MAIN left axis, orange)
    ax_hist.hist(
        itemSetDifficulties,
        bins=30,
        alpha=0.6,
        color="orange",
        edgecolor="black"
    )
    ax_hist.set_ylabel("Item number", color="orange")
    ax_hist.tick_params(axis="y", colors="orange")
    ax_hist.set_xlabel("Ability (logit)")

    # Information (SHIFTED left axis, green)
    ax_info.plot(thetaArray, infoArray, color="green")
    ax_info.set_ylabel("Information", color="green")
    ax_info.tick_params(axis="y", colors="green")

    # SEM (RIGHT axis, blue)
    ax_sem.plot(thetaArray, semArray, color="blue")
    ax_sem.set_ylabel("Standard Error of Measurement (logit)", color="blue")
    ax_sem.tick_params(axis="y", colors="blue")

    # ---- Cosmetics ----
    ax_hist.set_xlim(minTheta, maxTheta)
    ax_hist.grid(True, axis="x", alpha=0.3)
    ax_hist.set_title("Item bank quality")

    plt.tight_layout()
    plt.show()




def item_person_map(persons_filepath:str,
                    items_filepath:str,
                    bins:int = 15):
    '''Display item-person map (Wright map)'''

    # Load data
    persons_df = pd.read_csv(persons_filepath)
    items_df = pd.read_csv(items_filepath)

    # Optional: filter out items with totalResponses == 0
    if 'totalResponses' in items_df.columns:
        items_df = items_df[items_df['totalResponses'] > 0]

    # do not show fake words
    items_df = items_df[items_df['type'].isin(['yn','mc'])]

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

def getItemParams(items_filepath:str):
    '''
    Construct items dictionary with their parameters based on item output file of IRT procedure
    '''

    itemsParams = {}
    itemBank = pd.read_csv(items_filepath)
    for row in itemBank.itertuples(index=False):
        itemsParams[row.itemID] = {'b':row.b,
                                   'type': row.type}
    
    return itemsParams


def itemBankQuality(
    items_filepath: str,
    model: str = "1PL",
    sem_threshold: float = 0.35,
    theta_min: float = -12,
    theta_max: float = 12,
    theta_steps: int = 100,
    outfit_range: Tuple[float, float] = (0.7, 1.3),
    infit_range: Tuple[float, float] = (0.7, 1.3),
    fit_percentiles: Tuple[int, ...] = (5, 25, 50, 75, 95),
    extreme_outfit_hi: float = 1.5,
    extreme_infit_hi: float = 1.4,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Bank-level quality summary + optional glanceable printout.
    Keeps code compact: no helper subroutines/wrappers.
    """
    res: Dict[str, Any] = {}

    # ---------------- load + basic counts ----------------
    df0 = pd.read_csv(items_filepath)
    df0 = df0[df0["type"].isin(["yn", "mc", "f"])].copy()
    vc = df0["type"].value_counts()
    res["questions_total"] = int(len(df0))
    res["yn_questions_total"] = int(vc.get("yn", 0))
    res["mc_questions_total"] = int(vc.get("mc", 0))
    res["f_questions_total"] = int(vc.get("f", 0))

    # ---------------- fit subset (yn+mc) ----------------
    df = df0[df0["type"].isin(["yn", "mc"])].copy()
    res["yn_mc_questions_total"] = int(len(df))
    df["infit"] = pd.to_numeric(df.get("infit"), errors="coerce")
    df["outfit"] = pd.to_numeric(df.get("outfit"), errors="coerce")

    # ---------------- SEM usable range ----------------
    itemsParams = getItemParams(items_filepath)
    theta, info, sem = calculateItemBankInformation(
        itemsParams, itemsParams.keys(), model, theta_min, theta_max, theta_steps
    )
    theta = np.asarray(theta, float)
    sem = np.asarray(sem, float)
    ok = sem <= sem_threshold
    if np.any(ok):
        res["sem_theta_threshold_min"] = round(float(theta[ok][0]), 2)
        res["sem_theta_threshold_max"] = round(float(theta[ok][-1]), 2)
        res["sem_range"] = round(float(theta[ok][-1] - theta[ok][0]), 2)
    else:
        res["sem_theta_threshold_min"] = round(float(theta_max), 2)
        res["sem_theta_threshold_max"] = round(float(theta_min), 2)
        res["sem_range"] = round(float(theta_min - theta_max), 2)

    # ---------------- bank-level fit summaries ----------------
    # We'll compute the same set for: all, yn, mc
    for prefix, sub in [("", df), ("yn_", df[df["type"] == "yn"]), ("mc_", df[df["type"] == "mc"])]:
        res[f"{prefix}n"] = int(len(sub))

        for col, (lo, hi), extreme_hi in [
            ("infit", infit_range, extreme_infit_hi),
            ("outfit", outfit_range, extreme_outfit_hi),
        ]:
            s = sub[col].dropna()
            n = int(len(s))
            res[f"{prefix}{col}_n_nonmissing"] = n

            # percentiles + median (median also equals p50 if you include it)
            if n:
                qs = np.percentile(s.to_numpy(), fit_percentiles)
                for p, q in zip(fit_percentiles, qs):
                    res[f"{prefix}{col}_p{int(p)}"] = round(float(q), 3)
                res[f"{prefix}{col}_median"] = round(float(s.median()), 2)
            else:
                for p in fit_percentiles:
                    res[f"{prefix}{col}_p{int(p)}"] = np.nan
                res[f"{prefix}{col}_median"] = np.nan

            # directional misfit rates
            if n:
                under = int((s > hi).sum())
                over = int((s < lo).sum())
                res[f"{prefix}{col}_bad_ratio"] = round(float((under + over) / n), 3)
                res[f"{prefix}{col}_underfit_ratio"] = round(float(under / n), 3)
                res[f"{prefix}{col}_overfit_ratio"] = round(float(over / n), 3)
                res[f"{prefix}{col}_gt_{extreme_hi}_ratio"] = round(float((s > extreme_hi).mean()), 3)
            else:
                res[f"{prefix}{col}_bad_ratio"] = np.nan
                res[f"{prefix}{col}_underfit_ratio"] = np.nan
                res[f"{prefix}{col}_overfit_ratio"] = np.nan
                res[f"{prefix}{col}_gt_{extreme_hi}_ratio"] = np.nan

    # ---------------- optional: neat, glanceable print ----------------
    if verbose:
        def fnum(x, nd=2):
            if x is None or (isinstance(x, float) and np.isnan(x)):
                return "NA"
            if isinstance(x, (int, np.integer)):
                return str(int(x))
            if isinstance(x, (float, np.floating)):
                return f"{float(x):.{nd}f}"
            return str(x)

        def fpct(x):
            if x is None or (isinstance(x, float) and np.isnan(x)):
                return "NA"
            return f"{100.0 * float(x):.1f}%"

        p5, p25, p75, p95 = fit_percentiles[0], fit_percentiles[1], fit_percentiles[-2], fit_percentiles[-1]

        print("\nITEM BANK QUALITY — QUICK GLANCE")
        print("--------------------------------")
        print(f"Total items (yn+mc+f): {res['questions_total']} | yn: {res['yn_questions_total']} | mc: {res['mc_questions_total']} | f: {res['f_questions_total']}")
        print(f"Fit evaluated on (yn+mc): {res['yn_mc_questions_total']}")
        print(f"SEM≤{sem_threshold} usable θ-range: [{fnum(res['sem_theta_threshold_min'])}, {fnum(res['sem_theta_threshold_max'])}] (width {fnum(res['sem_range'])})")

        for label, prefix in [("ALL (yn+mc)", ""), ("YN items", "yn_"), ("MC items", "mc_")]:
            print(f"\n{label} (n={res[prefix+'n']})")
            print("-" * (len(label) + 6))

            print(f"Infit : med {fnum(res[prefix+'infit_median'])} | p{p5} {fnum(res[prefix+f'infit_p{p5}'],3)}  p{p25} {fnum(res[prefix+f'infit_p{p25}'],3)}  p{p75} {fnum(res[prefix+f'infit_p{p75}'],3)}  p{p95} {fnum(res[prefix+f'infit_p{p95}'],3)}")
            print(f"        bad {fpct(res[prefix+'infit_bad_ratio'])} (under {fpct(res[prefix+'infit_underfit_ratio'])}, over {fpct(res[prefix+'infit_overfit_ratio'])}) | >{extreme_infit_hi} {fpct(res[prefix+f'infit_gt_{extreme_infit_hi}_ratio'])}")

            print(f"Outfit: med {fnum(res[prefix+'outfit_median'])} | p{p5} {fnum(res[prefix+f'outfit_p{p5}'],3)}  p{p25} {fnum(res[prefix+f'outfit_p{p25}'],3)}  p{p75} {fnum(res[prefix+f'outfit_p{p75}'],3)}  p{p95} {fnum(res[prefix+f'outfit_p{p95}'],3)}")
            print(f"        bad {fpct(res[prefix+'outfit_bad_ratio'])} (under {fpct(res[prefix+'outfit_underfit_ratio'])}, over {fpct(res[prefix+'outfit_overfit_ratio'])}) | >{extreme_outfit_hi} {fpct(res[prefix+f'outfit_gt_{extreme_outfit_hi}_ratio'])}")

        print(f"\nNotes: 'bad' = outside ranges infit{infit_range}, outfit{outfit_range}; "
              "under = >hi, over = <lo.\n")

    return res


def personSetQuality(persons_filepath:str):
    '''
    Calculate fit quality of person set used for IRT
    '''

    persons = pd.read_csv(persons_filepath)
    results = {}

    # question count
    results['persons_total'] = len(persons)

    # stats on infit/outfit
    results['outfit_median'] = round(float(persons['outfit'].median()),ndigits=2)
    results['infit_median'] = round(float(persons['infit'].median()),ndigits=2)
    results['bad_outfit_items'] = len(persons[(persons['outfit']<0.7) | (persons['outfit']>1.3)])
    results['bad_outfit_items_ratio'] = round(results['bad_outfit_items'] / len(persons),ndigits=2)
    results['bad_infit_items'] = len(persons[(persons['infit']<0.7) | (persons['infit']>1.3)])
    results['bad_infit_items_ratio'] = round(results['bad_infit_items'] / len(persons),ndigits=2)

    return results


##################
# tests
##################

# item_bank_path = '/Users/grigorygolovin/Library/CloudStorage/OneDrive-Personal/Projects/word stock estimation/MyVocab stats/items_en.txt'
# item_params = getItemParams(item_bank_path)

# showTestInfo(item_params,item_params)

# pprint.pprint(itemBankQuality(item_bank_path,sem_threshold=0.35))

# # getItemParams(item_bank_path)

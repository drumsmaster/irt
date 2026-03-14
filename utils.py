##########################
# IRT/CAT
# utilities module
##########################

import io


def readTableIntoDict(filename, idKey, delimiter = '\t'):
    '''Read a table and form a dict from it'''

    resultDict = {}
    f = io.open(filename, mode='r', encoding='utf-8')
    keys = f.readline()[:-1].split(delimiter)
    for line in f.readlines():
        newdict = {}
        values = line[:-1].split(delimiter)
        for i in range(len(keys)):
            if len(values) <= i:
                newdict[keys[i]] = ''
            else:
                newdict[keys[i]] = values[i]
        id = newdict.pop(idKey)
        resultDict[id] = newdict
    f.close()
    return resultDict


def readItemsParams(filename, delimiter = ',', idKey='itemID'):
    '''Read a list of dicts and form a dict of items params'''

    resultDict = readTableIntoDict(filename, idKey, delimiter)

    # make sure b is a number
    for key in resultDict.keys():
        try:
            resultDict[key]['b'] = float(resultDict[key]['b'])
        except:
            resultDict[key]['b'] = float('nan')

    return resultDict


def readPersonsParams(filename, delimiter = ',', idKey='personID'):
    '''Read a list of dicts and form a dict of persons params'''

    resultDict = readTableIntoDict(filename, idKey, delimiter)

    # make sure b is a number
    for key in resultDict.keys():
        try:
            resultDict[key]['theta'] = float(resultDict[key]['theta'])
        except:
            resultDict[key]['theta'] = float('nan')

    return resultDict
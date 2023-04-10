# -*- coding: utf-8 -*-
import irt,quality
import codecs
import timeit
import sys
import pprint

#   perform joint item-person parameters estimation and write down the results

#   parameters
language = 'en'
user_folder = 'Drumsmaster' # Grigory if at work
data_filename = 'C:\\Users\\'+user_folder+'\\SkyDrive\\MyVocab stats\\results '+language+'.txt'       #   file with the data in a table format
testwords_filename = 'C:\\Users\\'+user_folder+'\\Dropbox\\Projects\\word stock estimation\\gae_test\\static\\data\\'+language+'\\test_words_'+language+'.txt'       #   testwords file
persons_filename = 'C:\\Users\\'+user_folder+'\\SkyDrive\\MyVocab stats\\persons_'+language+'.txt'    #   file to write with person data
items_filename = 'C:\\Users\\'+user_folder+'\\SkyDrive\\MyVocab stats\\items_'+language+'.txt'        #   file to write with items data
model = '1PL'
personsNum = -1
precisions={'theta':0.02,'a':0.02,'b':0.02}

#   read data
personsResponses, itemsResponses = irt.readTable(data_filename,linesNum=personsNum)
print 'Response data loaded. Items: {:d}, persons: {:d}'.format(len(itemsResponses),len(personsResponses))

#   estimate parameters
start_time = timeit.default_timer()
print 'Parameter estimation started'
personsParams,itemsParams,badItems,status = irt.getItemPersonJointParams(
        personsResponses,itemsResponses,model=model,
        ranges={'a':(0.1,3),'b':(-6,6),'theta':(-6,6)},
        steps={'theta':20,'a':1,'b':20},
        precisions=precisions,
        priorMean={'a':1,'b':0,'theta':0},
        priorSigma={'a':1,'b':2,'theta':2},
        minRanges={'theta':1,'a':1,'b':1},
        skip=True)
elapsed = timeit.default_timer() - start_time
print 'Parameter estimation stopped. ' + status
print 'Estimation time: {:.0f} s'.format(elapsed)

#   save persons thetas and fit quality (Q1 and p) into file
f = codecs.open(persons_filename,mode='w',encoding='utf-8')
f.write('personID\ttheta\tQ1\tQ1_p\titems\n')  #   header
for personID in personsParams.keys():
    q1,p = quality.getPersonQ1(personsResponses[personID],personsParams[personID],itemsParams,model)
    line = u'{:s}\t{:.2f}\t{:.2f}\t{:.4f}\t{:d}\n'.format(
            personID,personsParams[personID]['theta'],q1,p,len(personsResponses[personID]))
    f.write(line)
f.close()
print 'Person parameters saved'

#   save items params and fit quality (Q1 and p) into file
itemsParams.update(badItems)
f = codecs.open(items_filename,mode='w',encoding='utf-8')
f.write('itemID\ta\tb\tQ1\tQ1_p\ttotalResponses\tratioOfCorrectResponses\n') #   header
for itemID in itemsParams.keys():
    try:
        q1,p = quality.getItemQ1(itemsResponses[itemID],itemsParams[itemID],personsParams,model)
    except:
        print u'Item [{:s}] - impossible to calculate quality'.format(itemID)
        q1 = float('nan')
        p = float('nan')

    if itemsParams[itemID]['totalResponses'] == 0:
        ratioOfCorrectResponses = -1
    else:
        ratioOfCorrectResponses = float(itemsParams[itemID]['correctResponses'])/float(itemsParams[itemID]['totalResponses'])
    line = u'{:s}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.4f}\t{:d}\t{:.4f}\n'.format(
        itemID,itemsParams[itemID]['a'],itemsParams[itemID]['b'],q1,p,itemsParams[itemID]['totalResponses'],ratioOfCorrectResponses)
    f.write(line)
f.close()
print 'Item parameters saved'
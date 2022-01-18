import os
import pickle
from train import NB_Model
import math
import string
import random

def printF(f, mf):
    print("\nF-scores")
    print("For legitimate e-mails: ", end="")
    print(f["legitimate"])
    print("For spam e-mails: ", end="")
    print(f["spam"])
    print("Macro averaged F is " + str(mf))

def printP(p, mp):
    print("Precision")
    print("For legitimate e-mails: " + str(p["legitimate"]))
    print("For spam e-mails: " + str(p["spam"]))
    print("\nMacro averaged precision is " + str(mp))

def printR(r, mr):
    print("\nRecall")
    print("For legitimate e-mails: " + str(r["legitimate"]))
    print("For spam e-mails: " + str(r["spam"]))
    print("\nMacro averaged recall is " + str(mr))

# output of both system is stored in these dictionaries
# format: output[x] = [y,z]
# x is filename, y is label given by the system ("legitimate" or "spam"), z is True if decision is correct, False otherwise
# ex. output["5-1298msg1.txt"] = ["legitimate", True]
# ex. output_MI["5-1298msg1.txt"] = ["spam", False]
output = {}
output_MI = {}

# precision: dictionary from class name to precision
# returns macro averaged precision
def getMacroAvgP(precision):
    macroAvgP = sum(precision.values()) / len(classes)
    return macroAvgP

# tp: dictionary from class name to number of true positives
# fp: dictionary from class name to number of false positives
def getPrecision(tp, fp):
    precision = dict()
    for c in classes:
        precision[c] = tp[c] / (tp[c] + fp[c])
    return precision

# tp: dictionary from class name to number of true positives
# fn: dictionary from class name to number of false negatives
def getRecall(tp, fn):
    recall = dict()
    for c in classes:
        recall[c] = tp[c] / (tp[c] + fn[c])   
    return recall

# fMeasure: dictionary from class name to F scores
def getMacroAvgF(fMeasure):
   return sum(fMeasure.values()) / len(classes)

# given precision and recall, calculates f scores for all classes
def getF(precision, recall):
    fMeasure = {}
    for c in classes:
        f = 2*precision[c]*recall[c] / (precision[c] + recall[c])
        fMeasure[c] = f
    return fMeasure

# given the output of two system, calculates F scores
# returns the absolute value of the difference between these scores
def getS(output1, output2):
    # tp fp tn fn
    st1 = {"legitimate":{"tp":0, "fp":0, "tn":0, "fn":0}, "spam":{"tp":0, "fp":0, "tn":0, "fn":0}}
    st2 = {"legitimate":{"tp":0, "fp":0, "tn":0, "fn":0}, "spam":{"tp":0, "fp":0, "tn":0, "fn":0}}
    for filename in output1:
        a = output1[filename]
        b = output2[filename]
        
        ### For a
        if a[1]:
            if a[0] == "legitimate":
                st1["legitimate"]["tp"] += 1
                st1["spam"]["tn"] += 1
            else:
                st1["spam"]["tp"] += 1
                st1["legitimate"]["tn"] += 1
        else:
            if a[0] == "legitimate":
                st1["legitimate"]["fp"] += 1
                st1["spam"]["fn"] += 1
            else:
                st1["spam"]["fp"] += 1
                st1["legitimate"]["fn"] += 1
        #### For b
        if b[1]:
            if b[0] == "legitimate":
                st2["legitimate"]["tp"] += 1
                st2["spam"]["tn"] += 1
            else:
                st2["spam"]["tp"] += 1
                st2["legitimate"]["tn"] += 1
        else:
            if b[0] == "legitimate":
                st2["legitimate"]["fp"] += 1
                st2["spam"]["fn"] += 1
            else:
                st2["spam"]["fp"] += 1
                st2["legitimate"]["fn"] += 1
        
    # calculate f from st1 and st2
    ### f1
    p1 = {}
    p1["legitimate"] = st1["legitimate"]["tp"] / (st1["legitimate"]["tp"] + st1["legitimate"]["fp"])
    p1["spam"] = st1["spam"]["tp"] / (st1["spam"]["tp"] + st1["spam"]["fp"])
    recall1 = {}
    recall1["legitimate"] = st1["legitimate"]["tp"] / (st1["legitimate"]["tp"] + st1["legitimate"]["fn"])
    recall1["spam"] = st1["spam"]["tp"] / (st1["spam"]["tp"] + st1["spam"]["fn"])
    f1 = {}
    f1["legitimate"] = (2 * p1["legitimate"] * recall1["legitimate"]) / (p1["legitimate"] + recall1["legitimate"])
    f1["spam"] = (2 * p1["spam"] * recall1["spam"]) / (p1["spam"] + recall1["spam"])
    macroAvgF1 = sum(f1.values()) / 2

    ### f2
    p2 = {}
    p2["legitimate"] = st2["legitimate"]["tp"] / (st2["legitimate"]["tp"] + st2["legitimate"]["fp"])
    p2["spam"] = st2["spam"]["tp"] / (st2["spam"]["tp"] + st2["spam"]["fp"])
    recall2 = {}
    recall2["legitimate"] = st2["legitimate"]["tp"] / (st2["legitimate"]["tp"] + st2["legitimate"]["fn"])
    recall2["spam"] = st2["spam"]["tp"] / (st2["spam"]["tp"] + st2["spam"]["fn"])
    f2 = {}
    f2["legitimate"] = (2 * p2["legitimate"] * recall2["legitimate"]) / (p2["legitimate"] + recall2["legitimate"])
    f2["spam"] = (2 * p2["spam"] * recall2["spam"]) / (p2["spam"] + recall2["spam"])
    macroAvgF2 = sum(f2.values()) / 2

    return abs(macroAvgF1 - macroAvgF2)


# model (NB_model) is loaded from the pickle file that train.py or train_mi.py dumped
def getModel(useMutualInfo):
    modelName = ""
    if useMutualInfo:
        modelName = "model_mi.pickle"
    else:
        modelName = "model.pickle"

    infile = open(modelName,"rb")
    model = pickle.load(infile)
    infile.close()
    return model

# for all files in test data, makes a prediction as "legitimate" or "spam" using the model
# fills the variables true positive/false positive/true negative/false negative
def process(useMutualInfo):

    # traverse test set, compare probability of each email being legitimate/spam
    
    probOfLegitimate = math.log(model.probL)
    probOfSpam = math.log(model.probS)

    path = pathLegitimate
    for filename in os.listdir(path):
        file = open(os.path.join(path, filename), 'r', encoding='latin-1')
        fileContent = file.read()

        resultL = probOfLegitimate              # probability of being legitimate accumulates in resultL
        resultS = probOfSpam                    # similar for resultS

        for token in fileContent.split():

            if token in model.condProb:         # check if token has seen in training stage

                probOfWordGivenClass = model.condProb[token]["legitimate"]      # get the probability of this token given legitimate class
                resultL += math.log(probOfWordGivenClass)                       # add its log to resultL
                
                probOfWordGivenClass = model.condProb[token]["spam"]            # get the probability of this token given legitimate class
                resultS += math.log(probOfWordGivenClass)                       # add its log to resultS

            else:
                # for tokens we have not seen before
                # add a probability for these newly encountered tokens, only if we are not using mutual info
                if not useMutualInfo:
                    resultL += math.log(model.alpha/(model.wordCountL + model.alpha * len(model.condProb)))
                    resultS += math.log(model.alpha/(model.wordCountS + model.alpha * len(model.condProb)))

        
        if resultL >= resultS:
            # we classified this document as legitimate
            # it is indeed legitimate

            # check if we are using mutual information
            # if so, use variable with suffix _MI
            if useMutualInfo:
                # increment true positive for legitimate class
                # also increment true negative for spam, since we correctly didn't classify this e-mail as spam
                truePositive_MI["legitimate"] += 1
                trueNegative_MI["spam"] += 1
                output_MI[filename] = ["legitimate", True]      # store outcome in output file

            else:
                truePositive["legitimate"] += 1
                trueNegative["spam"] += 1
                output[filename] = ["legitimate", True]

        elif resultL < resultS:
            # we classified this document as spam
            # but it was legitimate

            if useMutualInfo:
                falseNegative_MI["legitimate"] += 1
                falsePositive_MI["spam"] += 1
                output_MI[filename] = ["spam", False]
            else:
                falseNegative["legitimate"] += 1
                falsePositive["spam"] += 1
                output[filename] = ["spam", False]


    # same process for spam e-mails
    path = pathSpam
    for filename in os.listdir(path):
        file = open(os.path.join(path, filename), 'r', encoding='latin-1')
        fileContent = file.read()

        resultL = probOfLegitimate
        resultS = probOfSpam

        for token in fileContent.split():
            if token in model.condProb:

                probOfWordGivenClass = model.condProb[token]["legitimate"]
                resultL += math.log(probOfWordGivenClass)
                
                probOfWordGivenClass = model.condProb[token]["spam"]
                resultS += math.log(probOfWordGivenClass)

            else:
                if not useMutualInfo:
                    resultL += math.log(model.alpha/(model.wordCountL + model.alpha * len(model.condProb)))
                    resultS += math.log(model.alpha/(model.wordCountS + model.alpha * len(model.condProb)))
        
        if resultS > resultL:
            # we classified this document as spam
            # it is indeed spam
            
            if useMutualInfo:
                truePositive_MI["spam"] += 1
                trueNegative_MI["legitimate"] += 1
                output_MI[filename] = ["spam", True]
            else:
                truePositive["spam"] += 1
                trueNegative["legitimate"] += 1
                output[filename] = ["spam", True]

        elif resultS < resultL:
            # we classified this document as legitimate
            # but it was spam
            
            if useMutualInfo:
                falseNegative_MI["spam"] += 1
                falsePositive_MI["legitimate"] += 1
                output_MI[filename] = ["legitimate", False]
            else:
                falseNegative["spam"] += 1
                falsePositive["legitimate"] += 1
                output[filename] = ["legitimate", False]


# start of the program

pathLegitimate = "./dataset/dataset/test/legitimate"
pathSpam = "./dataset/dataset/test/spam"

# for each file, calculate the class using Model
# compare with the class it actually belongs, store in variables below

truePositive = {"legitimate":0, "spam":0}
falsePositive = {"legitimate":0, "spam":0}
trueNegative = {"legitimate":0, "spam":0}
falseNegative = {"legitimate":0, "spam":0}

# these are for model with mutual information
truePositive_MI = {"legitimate":0, "spam":0}
falsePositive_MI = {"legitimate":0, "spam":0}
trueNegative_MI = {"legitimate":0, "spam":0}
falseNegative_MI = {"legitimate":0, "spam":0}


# run without using mutual information
model = getModel(useMutualInfo = False)
process(useMutualInfo = False)

# run with using mutual information
model = getModel(useMutualInfo = True)
process(useMutualInfo = True)

# class names
classes = ["legitimate", "spam"]

# macro averaged precision

precision = getPrecision(truePositive, falsePositive)
macroAvgP = getMacroAvgP(precision)

precision_MI = getPrecision(truePositive_MI, falsePositive_MI)
macroAvgP_MI = getMacroAvgP(precision_MI)

# recall
recall = getRecall(truePositive, falseNegative)
macroAvgR = sum(recall.values())/2
recall_MI = getRecall(truePositive_MI, falseNegative_MI)
macroAvgR_MI = sum(recall_MI.values())/2

# F measure

fMeasure = getF(precision, recall)
macroAvgF = getMacroAvgF(fMeasure)

fMeasure_MI = getF(precision_MI, recall_MI)
macroAvgF_MI = getMacroAvgF(fMeasure_MI)

print("\nNaive Bayes Without Feature Extraction\n")

# performance values for each class separately
for c in classes:
    print(c + " e-mails")
    print("True positives: " + str(truePositive[c]))
    print("False positives: " + str(falsePositive[c]))
    print("True negatives: " + str(trueNegative[c]))
    print("False negatives: " + str(falseNegative[c]))
    print("\n")

printP(precision, macroAvgP)
printR(recall, macroAvgR)
printF(fMeasure, macroAvgF)

print("\nNaive Bayes With Mutual Information\n")

# performance values for each class separately
for c in classes:
    print(c + " e-mails")
    print("True positives: " + str(truePositive_MI[c]))
    print("False positives: " + str(falsePositive_MI[c]))
    print("True negatives: " + str(trueNegative_MI[c]))
    print("False negatives: " + str(falseNegative_MI[c]))
    print("\n")

printP(precision_MI, macroAvgP_MI)
printR(recall_MI, macroAvgR_MI)
printF(fMeasure_MI, macroAvgF_MI)






    

# randomization test (difference between macro-averaged F-scores) of two versions of classifier

count = 0
s = abs(macroAvgF - macroAvgF_MI)           # original difference between two models
R = 1000                                    # number of iterations

newOutput = {}
newOutput_MI = {}

for i in range(0, R):
    # for each document, shuffle the output of models by 50% chance
    for doc in output:
        a = output[doc]
        b = output_MI[doc]
        randNum = random.randint(0,2)            # produces 0 or 1
        if randNum == 0:
            # shuffle
            newOutput[doc] = b
            newOutput_MI[doc] = a
        else:
            newOutput[doc] = a
            newOutput_MI[doc] = b
    
    # calculate s*
    sStar = getS(newOutput, newOutput_MI)
    if sStar > s:
        count += 1

p = (count + 1) / (R + 1)
print("\nP value is " + str(p))



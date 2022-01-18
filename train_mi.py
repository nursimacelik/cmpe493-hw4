# Spam/non-spam classifier using Multinomial Naive Bayes algorithm and Mutual Information as feature selection
# Same processes as train.py, but with reading the top k words and using only them

import os
import string
import pickle
import math

class DocumentClass():
    def __init__(self, vocabulary, docCount, wordCount, probability = 0):
        self.vocabulary = vocabulary
        self.docCount = docCount
        self.wordCount = wordCount

class NB_Model():
    def __init__(self, condProb = dict(), probL = 0, probS = 0, wordCountL = 0, wordCountS = 0, alpha = 1):
        self.condProb = condProb
        self.probL = probL
        self.probS = probS
        self.wordCountL = wordCountL
        self.wordCountS = wordCountS
        self.alpha = alpha


def get_vocabulary(path, className):
    vocab = dict()
    docCount = 0
    wordCount = 0
    for filename in os.listdir(path):
        docCount += 1
        file = open(os.path.join(path, filename), 'r', encoding='latin-1')
        fileContent = file.read().split()
        for token in fileContent:
            if token not in topWords:
                continue

            wordCount += 1
            if token not in vocab:
                vocab[token] = 1
            else:
                vocab[token] += 1
    return DocumentClass(vocab, docCount, wordCount)

alpha = 1

with open("topWords.pickle","rb") as infile:
    topWords = pickle.load(infile)

# read e-mails
pathLegitimate = "./dataset/dataset/training/legitimate"
pathSpam = "./dataset/dataset/training/spam"

classLegitimate = get_vocabulary(pathLegitimate, "legitimate")
classSpam = get_vocabulary(pathSpam, "spam")

allDocCount = classLegitimate.docCount + classSpam.docCount
probL = classLegitimate.docCount / allDocCount
probS = classSpam.docCount / allDocCount

allVocabulary = dict()

for word in topWords:
    allVocabulary[word] = classLegitimate.vocabulary.get(word, 0) + classSpam.vocabulary.get(word, 0)
    if word not in classLegitimate.vocabulary:
        classLegitimate.vocabulary[word] = 0
    if word not in classSpam.vocabulary:
        classSpam.vocabulary[word] = 0

condProb_MI = dict()       # ex. condProb_MI["Chinese"]["legitimate"] = 0.25

k = len(topWords)

# for each class, calculate probability of each word given that class
for word, frequency in classLegitimate.vocabulary.items():
    
    condProb_MI[word] = dict()
    prob = (frequency + alpha) / (classLegitimate.wordCount + alpha * k)
    condProb_MI[word]["legitimate"] = prob
    

for word, frequency in classSpam.vocabulary.items():

    if word not in condProb_MI:
        condProb_MI[word] = dict()
    prob = (frequency + alpha) / (classSpam.wordCount + alpha * k)
    condProb_MI[word]["spam"] = prob


model_MI = NB_Model(condProb_MI, probL, probS, classLegitimate.wordCount, classSpam.wordCount, alpha)
# dump the model
with open("model_mi.pickle", "wb") as outfile:
    pickle.dump(model_MI, outfile)

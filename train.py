# Spam/non-spam classifier using Multinomial Naive Bayes algorithm

import os
import string
import pickle
import math

class DocumentClass():
    def __init__(self, vocabulary, docCount, wordCount, probability = 0):
        self.vocabulary = vocabulary
        self.docCount = docCount
        self.wordCount = wordCount

# model to be dumped
class NB_Model():
    def __init__(self, condProb = dict(), probL = 0, probS = 0, wordCountL = 0, wordCountS = 0, alpha = 1):
        self.condProb = condProb
        self.probL = probL
        self.probS = probS
        self.wordCountL = wordCountL
        self.wordCountS = wordCountS
        self.alpha = alpha

def get_vocabulary(path, className):
    vocab = dict()          # store frequency for each word (scope is class)
    docCount = 0            # number of all documents that belong to this class
    wordCount = 0           # number of words that belong to this class (a document from this class) (non unique)
    for filename in os.listdir(path):
        docCount += 1
        file = open(os.path.join(path, filename), 'r', encoding='latin-1')
        fileContent = file.read().split()
        for token in fileContent:
            wordCount += 1
            if token not in vocab:
                vocab[token] = 1
            else:
                vocab[token] += 1
        
        # for each token, find number of documents that contain the token
        for token in set(fileContent):
            if token not in docCountWordOccurs[className]:
                docCountWordOccurs[className][token] = 1
            else:
                docCountWordOccurs[className][token] += 1
    # create and return a document class out of these information
    return DocumentClass(vocab, docCount, wordCount)

# start of the program

alpha = 1

# read e-mails
pathLegitimate = "./dataset/dataset/training/legitimate"
pathSpam = "./dataset/dataset/training/spam"

# stores the number of documents each word occurs
docCountWordOccurs = {"legitimate": {}, "spam": {}}
# traverse files and extract vocabulary along other info for both classes
classLegitimate = get_vocabulary(pathLegitimate, "legitimate")
classSpam = get_vocabulary(pathSpam, "spam")

allDocCount = classLegitimate.docCount + classSpam.docCount
probL = classLegitimate.docCount / allDocCount                  # probability of legitimate class
probS = classSpam.docCount / allDocCount                        # probability of spam class

allVocabulary = dict()                                          # dictionary to hold all words and their frequencies

for word in set(classLegitimate.vocabulary) | set(classSpam.vocabulary):
    allVocabulary[word] = classLegitimate.vocabulary.get(word, 0) + classSpam.vocabulary.get(word, 0)
    if word not in classLegitimate.vocabulary:
        classLegitimate.vocabulary[word] = 0
    if word not in classSpam.vocabulary:
        classSpam.vocabulary[word] = 0
print("Size of the vocabulary is " + str(len(allVocabulary)))

condProb = dict()       # ex. condProb["Chinese"]["legitimate"] = 0.25

# for each class, calculate probability of each word given that class
for word, frequency in classLegitimate.vocabulary.items():
    
    condProb[word] = dict()
    prob = (frequency + alpha) / (classLegitimate.wordCount + alpha * len(allVocabulary))
    condProb[word]["legitimate"] = prob
    

for word, frequency in classSpam.vocabulary.items():

    prob = (frequency + alpha) / (classSpam.wordCount + alpha * len(allVocabulary))
    condProb[word]["spam"] = prob

# create the model
model = NB_Model(condProb, probL, probS, classLegitimate.wordCount, classSpam.wordCount, alpha)
# dump it
with open("model.pickle", "wb") as outfile:
    pickle.dump(model, outfile)


# feature selection with mutual information
mutualInfo = dict()
for word in allVocabulary:
    
    n = [[1, 1], [1, 1]]                                # initial values are 1 instead of 0, so that divison error doesn't occur
    
    if word not in docCountWordOccurs["legitimate"]:
        docCountWordOccurs["legitimate"][word] = 0

    if word not in docCountWordOccurs["spam"]:
        docCountWordOccurs["spam"][word] = 0
    
    n[1][1] += docCountWordOccurs["legitimate"][word]   # number of documents from legitimate class that the word occurs
    n[0][1] += classLegitimate.docCount - n[1][1]       # number of documents from legitimate class that the word doesn't occur
    n[1][0] += docCountWordOccurs["spam"][word]         # number of documents from spam class that the word occurs
    n[0][0] += classSpam.docCount - n[1][0]             # number of documents from spam class that the word doesn't occur
    
    miSum = 0
    for i in range(0,2):
        for j in range(0,2):
            k = 1 - i
            l = 1 - j
            temp = (allDocCount * n[i][j]) / ((n[i][j] + n[k][j]) * (n[i][j] + n[i][l]))
            miSum += (n[i][j]/allDocCount)*(math.log2(temp))
    
    mutualInfo[word] = miSum

# get top k elements
k = 100
mutualInfo = sorted(mutualInfo.items(), reverse=True, key=lambda item: item[1])[:k]
topWords = []

for item in mutualInfo:
    topWords.append(item[0])

print("Top 100 words are: ")
print(topWords)

# dump the top k words so that train_mi.py can use it
with open("topWords.pickle", "wb") as outfile:
    pickle.dump(topWords, outfile)

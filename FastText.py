import fasttext
import numpy as np
import string
import json
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer

# # Skipgram model
# model = fasttext.skipgram('data.txt', 'model')
# print(model.words) # list of words in dictionary
#
# # CBOW model
# model = fasttext.cbow('data.txt', 'model')
# print(model.words) # list of words in dictionary

postProcessedTrainPhrases = []
postProcessedTestPhrases = []
testSentences = []

def preprocessData():
    print("Loading and preprocessing data...\n")
    # load training and testing data
    with open('labeled_document2.json') as json_data:
        allTrainData = json.load(json_data)

    trainPhrases, testPhrases, trainLabel, testLabel = train_test_split(allTrainData['Comment'],
                                                                        allTrainData['CommentLabel'], test_size=0.2,
                                                                        random_state=42)

    #     print(testPhrases[0:100])
    punctuation = list(string.punctuation)
    stopWords = stopwords.words('english') + punctuation

    engStemmer = SnowballStemmer('english')
    # postProcessedTrainPhrases = []
    #     for phrase in trainPhrases:
    #         uni_doc = unicode(phrase, errors='replace')
    #         tokens = word_tokenize(uni_doc)
    #         filtered = [word for word in tokens if word not in stop_words]
    #         try:
    #             stemmed = [stemmer.stem(word) for word in filtered]
    #         except UnicodeDecodeError:
    #             print(word)
    #         postProcessedTrainPhrases.append(parsedWords)

    #     for phrase in testPhrases:
    #         uni_doc = unicode(phrase, errors='replace')
    #         tokens = word_tokenize(uni_doc)
    #         filtered = [word for word in tokens if word not in stop_words]
    #         try:
    #             stemmed = [stemmer.stem(word) for word in filtered]
    #         except UnicodeDecodeError:
    #             print(word)
    #         postProcessedTestPhrases.append(parsedWords)
    for phrase in trainPhrases:
        if not isinstance(phrase, str):
            continue
        tokens = word_tokenize(phrase)
        parsedWords = []
        for t in tokens:
            if t not in stopWords:
                parsedWords.append(engStemmer.stem(t))
        postProcessedTrainPhrases.append(parsedWords)

    for phrase in testPhrases:
        if not isinstance(phrase, str):
            continue
        tokens = word_tokenize(phrase)
        parsedWords = []
        for t in tokens:
            if t not in stopWords:
                parsedWords.append(engStemmer.stem(t))
        postProcessedTestPhrases.append(parsedWords)
    return (trainLabel, testLabel)

def outputFile(filename, phrases, labels):
    f = open(filename+".txt", "w+")
    for i in range(len(phrases)):
        sentence = ""
        for j in range(len(phrases[i])):
            sentence += " " + phrases[i][j]
        f.write("__label__" + str(labels[i]) + sentence + "\r\n")
    f.close()

def extractText(phrases):
    for i in range(len(phrases)):
        sentence = ""
        for j in range(len(phrases[i])):
            sentence += " " + phrases[i][j]
        testSentences.append(sentence)

# preprocess data
(trainLabels, testLabels) = preprocessData()

# create training and testing file
# outputFile("training", postProcessedTrainPhrases, trainLabels)
# outputFile("testing", postProcessedTestPhrases, testLabels)

# train the fasttext model
print('Buidling the model...\n')
classifier = fasttext.supervised('training.txt', 'model')
result = classifier.test('testing.txt')

# classify testing data
extractText(postProcessedTestPhrases)
labels = classifier.predict(testSentences)
print('Negative comments found:')
for i in range(len(labels)):
    if int(labels[i][0]) == 0:
        print(testSentences[i])

# evaluate the model
print('\nEvaluating the model...')
count = 0
for i in range(len(testLabels)):
    # print(testLabels[i])
    if testLabels[i] == int(labels[i][0]):
        count += 1
print('Accuracy at 1: ', count/len(testLabels))
print('Precision at 1:', result.precision)
print('Recall at 1:', result.recall)
print('Total number of examples:', result.nexamples)
print('Number of correctly predicted examples:', count, '\n')

# classify sample texts
print('Classifying sample texts...')
texts = ['homework google', 'try it yourself', 'you didnt show any effort']
labels = classifier.predict(texts)
print(texts)
print(labels)


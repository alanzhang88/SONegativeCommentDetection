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


def preprocessData():
    print("Loading and preprocessing data...")
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
            sentence += " " + phrases[i][j];
        f.write("__label__" + str(labels[i]) + sentence)
    f.close()

(trainSenti, testSenti) = preprocessData()
outputFile("training", postProcessedTrainPhrases, trainSenti)
print(postProcessedTrainPhrases)

classifier = fasttext.supervised('train.txt', 'model')
result = classifier.test('test.txt')
print('P@1:', result.precision)
print('R@1:', result.recall)
print('Number of examples:', result.nexamples)

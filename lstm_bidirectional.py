import numpy as np
import pandas as pd

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding, Bidirectional
from keras.layers.core import SpatialDropout1D
from sklearn.model_selection import StratifiedKFold
from keras.datasets import imdb
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer

from gensim import corpora

# def parsePhrases(stopWords, engStemmer, phrases):
#     print "parse the phrases with stopwords and stemmer"
#     processedPhrases = []
#     for phrase in phrases:
#         tokens = word_tokenize(phrase)
#         parsedWords = []
#         for t in tokens:
#             if t not in stopWords:
#                 parsedWords.append(engStemmer.stem(t))
#         processedPhrases.append(parsedWords)
#     return processedPhrases
postProcessedTrainPhrases = []
postProcessedTestPhrases = []

def preprocessData():
    print "Loading and preprocessing data..."
    # load training and testing data
    trainData = pd.read_csv('train.tsv', sep='\t', header=0)
    testData = pd.read_csv('test.tsv', sep='\t', header=0)

    trainPhrases = trainData['Phrase'].values
    testPhrases = testData['Phrase'].values

    stopWords = set(['(', ')', '[', ']', '{', '}',
                     '.', ',',':', ';','"', "'"])
    stopWords.update(set(stopwords.words('english')))
    engStemmer = SnowballStemmer('english')
    #postProcessedTrainPhrases = []
    for phrase in trainPhrases:
        tokens = word_tokenize(phrase)
        parsedWords = []
        for t in tokens:
            if t not in stopWords:
                parsedWords.append(engStemmer.stem(t))
        postProcessedTrainPhrases.append(parsedWords)

    #postProcessedTestPhrases = []
    for phrase in testPhrases:
        tokens = word_tokenize(phrase)
        parsedWords = []
        for t in tokens:
            if t not in stopWords:
                parsedWords.append(engStemmer.stem(t))
        postProcessedTestPhrases.append(parsedWords)
    #postProcessedTestPhrases = parsePhrases(stopWords, engStemmer, testPhrases)
    return trainData['Sentiment'].values


def convertPhrasesToIDs(phrases):
    print "converting the phrases to id to be processed"
    wordIDs = []
    wordIDLens = []
    for phrase in phrases:
        ids = []
        for word in phrase:
            ids.append(toIDMap.token2id[word])
        wordIDs.append(ids)
        wordIDLens.append(len(ids))
    return ( wordIDs, wordIDLens )

def findSequenceLen(wordListLen):
    print "calculate the norm sequence length"
    wordLenMean = np.mean(wordListLen)
    wordLenStd = np.std(wordListLen)
    return np.round(wordLenMean + 3 * wordLenStd).astype(int)

if __name__ == "__main__":
    trainSenti = preprocessData()
    # process training data and testing data
    toIDMap = corpora.Dictionary(np.concatenate((postProcessedTrainPhrases, postProcessedTestPhrases), axis=0))
    allPhraseSize = len(toIDMap.keys())

    (trainWordIDs, trainWordIDLens) = convertPhrasesToIDs(postProcessedTrainPhrases)
    (testWordIDs, testWordIDLens) = convertPhrasesToIDs(postProcessedTestPhrases)

    sequenceLen = findSequenceLen(trainWordIDLens + testWordIDLens)

    print "pad sequence"
    trainingData = sequence.pad_sequences(np.array(trainWordIDs), maxlen=sequenceLen)
    testingData = sequence.pad_sequences(np.array(testWordIDs), maxlen=sequenceLen)

    print "categorize the labels"
    #print len(np.unique(trainSenti))
    trainingDataLabel = np_utils.to_categorical(trainSenti, len(np.unique(trainSenti)))

    cvscores = []
    batchSize = 15606
    end = 156060
    for i in range(0, 10):
        print("Fold:%d" % i)
        tests = trainingData[i * batchSize:(i + 1) * batchSize]
        tests_label = trainingDataLabel[i * batchSize:(i + 1) * batchSize]
        if (i > 0):
            trains = np.concatenate((trainingData[0:(i * batchSize)], trainingData[(i + 1) * batchSize:end]), axis=0)
            trains_label = np.concatenate((trainingDataLabel[0:(i * batchSize)], trainingDataLabel[(i + 1) * batchSize:end]), axis=0)
        else:
            trains = trainingData[(i + 1) * batchSize:end]
            trains_label = trainingDataLabel[(i + 1) * batchSize:end]

        model = Sequential()
        model.add(Embedding(allPhraseSize, 128))
        model.add(SpatialDropout1D(0.1))
        model.add(Bidirectional(LSTM(128)))
        #model.add(Bidirectional(LSTM(128)))
        #model.add(Flatten())
        model.add(Dense(len(np.unique(trainSenti))))
        model.add(Activation('softmax'))

        # model = Sequential()
        # model.add(Embedding(allPhraseSize, 128, dropout=0.2))
        # model.add(LSTM(128, dropout_W=0.2, dropout_U=0.2))
        # model.add(Dense(num_labels))
        # model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(trains,trains_label , epochs=3, batch_size=256, verbose=1)
        # evaluate the model
        scores = model.evaluate(tests, tests_label, verbose=0)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
        cvscores.append(scores[1] * 100)

        if(i==0):
            model_json = model.to_json()
            with open('model_best2.json', 'w') as json_file:
                json_file.write(model_json)

            model.save_weights('model_best2.h5')


    test_pred = model.predict_classes(testingData)



    # test_df['Sentiment'] = test_pred.reshape(-1, 1)
    # header = ['PhraseId', 'Sentiment']
    # test_df.to_csv('./lstm_sentiment3.csv', columns=header, index=False, header=True)
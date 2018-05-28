import numpy as np
import os
import string
import json
from sklearn.model_selection import train_test_split
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
from keras.models import model_from_json
from gensim import corpora
# from imblearn.over_sampling import SMOTE

class LSTMModel():
    def __init__(self, postProcessedTrainPhrases = None, postProcessedTestPhrases = None, save_model = True):
        self.postProcessedTrainPhrases = []
        self.postProcessedTestPhrases = []
        self.model = None
        self.save_model = save_model

    def preprocessData(self,hijackData=None):
        print("Loading and preprocessing data...")
        # load training and testing data
        with open(os.path.dirname(__file__)+'/labeled_document2.json') as json_data:
            allTrainData = json.load(json_data)
        trainPhrases, testPhrases, trainLabel,testLabel = train_test_split(allTrainData['Comment'], allTrainData['CommentLabel'], test_size=0.2, random_state=42)

        if hijackData is not None:
            trainPhrases = hijackData['X_train']
            testPhrases = hijackData['X_test']
            trainLabel = hijackData['y_train']
            testLabel = hijackData['y_test']

    #     print(testPhrases[0:100])
        punctuation = list(string.punctuation)
        stopWords = stopwords.words('english') + punctuation
        engStemmer = SnowballStemmer('english')
        for phrase in trainPhrases:
            if not isinstance(phrase, str):
                continue
            tokens = word_tokenize(phrase)
            parsedWords = []
            for t in tokens:
                if t not in stopWords:
                    parsedWords.append(engStemmer.stem(t))
            self.postProcessedTrainPhrases.append(parsedWords)

        for phrase in testPhrases:
            if not isinstance(phrase, str):
                continue
            tokens = word_tokenize(phrase)
            parsedWords = []
            for t in tokens:
                if t not in stopWords:
                    parsedWords.append(engStemmer.stem(t))
            self.postProcessedTestPhrases.append(parsedWords)
        return (trainLabel,testLabel)


    def convertPhrasesToIDs(self, phrases, toIDMap):
        print ("converting the phrases to id to be processed")
        wordIDs = []
        wordIDLens = []
        for phrase in phrases:
            ids = []
            for word in phrase:
                ids.append(toIDMap.token2id[word])
            wordIDs.append(ids)
            wordIDLens.append(len(ids))
        return ( wordIDs, wordIDLens )

    def findSequenceLen(self, wordListLen):
        print( "calculate the norm sequence length")
        wordLenMean = np.mean(wordListLen)
        wordLenStd = np.std(wordListLen)
        return np.round(wordLenMean + 3 * wordLenStd).astype(int)


    def trainAndTest(self,sample_weight = None,hijackData = None):
        (trainSenti, testSenti) = self.preprocessData(hijackData)
        # process training data and testing data

        toIDMap = corpora.Dictionary(np.concatenate((self.postProcessedTrainPhrases, self.postProcessedTestPhrases), axis=0))
        allPhraseSize = len(toIDMap.keys())

        (trainWordIDs, trainWordIDLens) = self.convertPhrasesToIDs(self.postProcessedTrainPhrases, toIDMap)
        (testWordIDs, testWordIDLens) = self.convertPhrasesToIDs(self.postProcessedTestPhrases, toIDMap)

        sequenceLen = self.findSequenceLen(trainWordIDLens + testWordIDLens)

        print( "pad sequence")
        trainingData = sequence.pad_sequences(np.array(trainWordIDs), maxlen=sequenceLen)
        testingData = sequence.pad_sequences(np.array(testWordIDs), maxlen=sequenceLen)

        # sm = SMOTE(random_state=12, ratio = 1.0)
        # trainingData, trainSenti = sm.fit_sample(trainingData, trainSenti)

        print ("categorize the labels")
        #print len(np.unique(trainSenti))
        trainingDataLabel = np_utils.to_categorical(trainSenti, len(np.unique(trainSenti)))

        self.model = Sequential()
        self.model.add(Embedding(allPhraseSize, 128))
        self.model.add(SpatialDropout1D(0.1))
        self.model.add(Bidirectional(LSTM(128)))
        self.model.add(Dense(len(np.unique(trainSenti))))
        self.model.add(Activation('softmax'))

        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.fit(trainingData,trainingDataLabel , epochs=10, batch_size=256, verbose=2,sample_weight=sample_weight)
        # evaluate the model
        # serialize weights to HDF5
        if self.save_model:
            self.save(os.path.dirname(__file__)+"/LSTM.json",os.path.dirname(__file__)+"/LSTM.h5")

        # testingData, testSenti = sm.fit_sample(testingData, testSenti)
        testingDataLabel = np_utils.to_categorical(testSenti, len(np.unique(testSenti)))
        scores = self.model.evaluate(testingData, testingDataLabel, verbose=0)
        print("%s: %.2f%%" % (self.model.metrics_names[1], scores[1] * 100))

    # predictedRes = model.predict_proba(testingData)

    def save(self,JSONPath,ModelPath):
        model_json = self.model.to_json()

        #changed to absolute path
        with open(JSONPath, "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5, absolute path
        self.model.save_weights(ModelPath)
        print("Saved model to disk")

    def load(self,JSONPath,ModelPath):
        json_file = open(JSONPath, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        # load weights into new model
        self.model.load_weights(ModelPath)
        print("Loaded model from disk")

    def predict(self, comments, hijackData=None):

        if self.model is None:
            self.load(os.path.dirname(__file__)+'/LSTM.json',os.path.dirname(__file__)+"/LSTM.h5")
            self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            self.preprocessData(hijackData)

        self.postProcessedTestPhrases = []
        punctuation = list(string.punctuation)
        stopWords = stopwords.words('english') + punctuation
        engStemmer = SnowballStemmer('english')
        for phrase in comments:
            if not isinstance(phrase, str):
                continue
            tokens = word_tokenize(phrase)
            parsedWords = []
            for t in tokens:
                if t not in stopWords:
                    parsedWords.append(engStemmer.stem(t))
            self.postProcessedTestPhrases.append(parsedWords)

        toIDMap = corpora.Dictionary(np.concatenate((self.postProcessedTrainPhrases, self.postProcessedTestPhrases), axis=0))
        allPhraseSize = len(toIDMap.keys())

        (testWordIDs, testWordIDLens) = self.convertPhrasesToIDs(self.postProcessedTestPhrases, toIDMap)
        (trainWordIDs, trainWordIDLens) = self.convertPhrasesToIDs(self.postProcessedTrainPhrases, toIDMap)

        sequenceLen = self.findSequenceLen(testWordIDLens+trainWordIDLens)

        print( "pad sequence")
        testingData = sequence.pad_sequences(np.array(testWordIDs), maxlen=sequenceLen)

        # loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        
        res = self.model.predict(testingData)

        # res = [(np.array(l)/sum(l)).tolist() for l in predict_res]
        return [(np.array(l)/sum(l)).tolist() for l in res]

def return_new_lstm():
    lstm_new = LSTMModel()
    lstm_new.postProcessedTrainPhrases = []
    lstm_new.postProcessedTestPhrases = []
    return lstm_new

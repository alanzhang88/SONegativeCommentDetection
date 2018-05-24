import numpy as np
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
from imblearn.over_sampling import SMOTE

class LSTM():
    def __init__(self, postProcessedTrainPhrases = None, postProcessedTestPhrases = None):
        self.postProcessedTrainPhrases = []
        self.postProcessedTestPhrases = []


    def preprocessData(self):
        print("Loading and preprocessing data...")
        # load training and testing data
        with open('labeled_document2.json') as json_data:
            allTrainData = json.load(json_data)
        trainPhrases, testPhrases, trainLabel,testLabel = train_test_split(allTrainData['Comment'], allTrainData['CommentLabel'], test_size=0.2, random_state=42)
        
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


    def trainAndTest(self):
        (trainSenti, testSenti) = preprocessData()
        # process training data and testing data

        toIDMap = corpora.Dictionary(np.concatenate((self.postProcessedTrainPhrases, self.postProcessedTestPhrases), axis=0))
        allPhraseSize = len(toIDMap.keys())

        (trainWordIDs, trainWordIDLens) = convertPhrasesToIDs(self.postProcessedTrainPhrases, toIDMap)
        (testWordIDs, testWordIDLens) = convertPhrasesToIDs(self.postProcessedTestPhrases, toIDMap)

        sequenceLen = findSequenceLen(trainWordIDLens + testWordIDLens)

        print( "pad sequence")
        trainingData = sequence.pad_sequences(np.array(trainWordIDs), maxlen=sequenceLen)
        testingData = sequence.pad_sequences(np.array(testWordIDs), maxlen=sequenceLen)

        sm = SMOTE(random_state=12, ratio = 1.0)
        trainingData, trainSenti = sm.fit_sample(trainingData, trainSenti)

        print ("categorize the labels")
        #print len(np.unique(trainSenti))
        trainingDataLabel = np_utils.to_categorical(trainSenti, len(np.unique(trainSenti)))

        model = Sequential()
        model.add(Embedding(allPhraseSize, 128))
        model.add(SpatialDropout1D(0.1))
        model.add(Bidirectional(LSTM(128)))
        model.add(Dense(len(np.unique(trainSenti))))
        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(trainingData,trainingDataLabel , epochs=3, batch_size=256, verbose=1)
        # evaluate the model
        # serialize weights to HDF5
        model_json = model.to_json()
        with open("LSTM.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights("LSTM.h5")
        print("Saved model to disk")

        testingData, testSenti = sm.fit_sample(testingData, testSenti)
        testingDataLabel = np_utils.to_categorical(testSenti, len(np.unique(testSenti)))
        scores = model.evaluate(testingData, testingDataLabel, verbose=0)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

    # predictedRes = model.predict_proba(testingData)

    def predict(self, comments):

        json_file = open('LSTM.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("LSTM.h5")
        print("Loaded model from disk")

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

        toIDMap = corpora.Dictionary(self.postProcessedTestPhrases)
        allPhraseSize = len(toIDMap.keys())

        (testWordIDs, testWordIDLens) = self.convertPhrasesToIDs(self.postProcessedTestPhrases, toIDMap)

        sequenceLen = self.findSequenceLen(testWordIDLens)

        print( "pad sequence")
        testingData = sequence.pad_sequences(np.array(testWordIDs), maxlen=sequenceLen)

        # loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        predict_res = loaded_model.predict_proba(testingData)
        res = [(np.array(l)/sum(l)).tolist() for l in predict_res]
        return res

def return_new_lstm():
    lstm_new = LSTM()
    lstm_new.postProcessedTrainPhrases = []
    lstm_new.postProcessedTestPhrases = []
    return lstm_new

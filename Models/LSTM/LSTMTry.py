import numpy as np
import LSTMUtil
import pickle
import os
import string
import json
from keras.models import model_from_json
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

from gensim import corpora
from imblearn.over_sampling import SMOTE

with open('labeled_document2.json') as json_data:
	allTrainData = json.load(json_data)
trainPhrases, testPhrases, trainLabel,testLabel = train_test_split(allTrainData['Comment'], allTrainData['Score'], test_size=0.1, random_state=42)

postProcessedTrainPhrases = []
postProcessedTestPhrases = []

LSTM_process = LSTMUtil.return_new_lstm()
LSTM_process.trainAndTest()
# LSTM_process = LSTM(postProcessedTrainPhrases, postProcessedTestPhrases)

negcount = 0
poscount = 0
res = LSTM_process.predict(testPhrases)
for i in res:
	if i[0] > i[1]:
		negcount +=1
	else:
		poscount +=1

print("negative: ", negcount)
print("positive: ", poscount)


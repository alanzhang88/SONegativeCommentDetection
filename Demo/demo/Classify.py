import requests
from django.shortcuts import render
from django.http import HttpResponse
import sys, os
import numpy as np
import fasttext
from django.conf import settings as setting

# #import all models
sys.path.append(os.path.join(os.path.dirname(__file__), 'Models','LSTM'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'Models','CNN'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'Models','FastText'))

from LSTMUtil import LSTMModel
from CNNUtil import CNNModel
from FastTextUtil import FastText

class Classify:
    def predict(comments):
        #load models

        # lstm_model = LSTMModel()
        # cnn_model = CNNModel()
        # cnn_model.load_model(os.path.dirname(__file__)+'/Models/CNN/CNNmodel.h5')
        # fasttext_model = FastText()
        lstm_model = setting.LSTM
        cnn_model = setting.CNN
        fasttext_model = setting.FT

        # predict labels of comments
        lstm_label = lstm_model.predict(comments)
        cnn_label = cnn_model.predict(comments)
        fasttext_label = fasttext_model.predict(comments)

        # print log info
        print(lstm_label)
        print(cnn_label)
        print(fasttext_label)

        return lstm_label, cnn_label, fasttext_label

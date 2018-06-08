import os, sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__),'..','CNN'))
sys.path.append(os.path.join(os.path.dirname(__file__),'..','CNN','embeddings'))
sys.path.append(os.path.join(os.path.dirname(__file__),'..','LSTM'))

from CNNUtil import CNNModel
from LSTMUtil import LSTMModel
from data_preprocessing import load_file
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score
import pandas as pd


class AdaBoost:
    def __init__(self,DataPath,FileType,classifierName, iterations=10):
        if FileType == 'csv':
            data = pd.read_csv(DataPath,names=['Comment','Label'])
            data = data.sample(frac=1)
            X = data['Comment'].values
            y = data['Label'].values
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X,y,train_size=0.9)
        else:
            with open(DataPath) as json_data:
                data = json.load(json_data)
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(data['Comment'],data['CommentLabel'],train_size=0.9)
        self.iterations = iterations
        self.N = self.X_train.shape[0]
        self.weights = np.ones(self.N) / self.N
        self.alphas = []
        self.classifiers = []
        self.classifiers_name = classifierName

    def getDataAsDict(self):
        return {'X_train':self.X_train,
                'X_test':self.X_test,
                'y_train':self.y_train,
                'y_test':self.y_test}

    def calErr(self,pred):
        err = 0
        for i in range(self.N):
            if pred[i] != self.y_train[i]:
                err += self.weights[i]
        return err

    def updateWeight(self,pred,alpha):
        upWeightFactor = np.exp(alpha)
        downWeightFactor = np.exp(-alpha)
        self.weights = [self.weights[i]*downWeightFactor if pred[i] == self.y_train[i] else self.weights[i]*upWeightFactor for i in range(self.N)]
        self.weights = self.weights / np.sum(self.weights)

    def train(self):
        if self.classifiers_name == 'cnn':
            for i in range(self.iterations):
                cnnModel = CNNModel(save_model=False,epochs=10)
                cnnModel.data.hijackTrainTestData(X_train=self.X_train,
                                                  X_test=self.X_test,
                                                  y_train=self.y_train,
                                                  y_test=self.y_test)
                cnnModel.build_model(sample_weight=self.weights)
                cnnPred = np.argmax(cnnModel.predict(self.X_train),axis=1)
                cnnErr = self.calErr(cnnPred)
                print('At iteration %d, cnnErr: %f' % (i+1,cnnErr))
                alpha = 0.5 * np.log((1 - cnnErr) / cnnErr)
                self.alphas.append(alpha)
                self.classifiers.append(cnnModel)
                self.updateWeight(cnnPred,alpha)
        else:
            for i in range(self.iterations):
                lstmModel = LSTMModel(save_model=False)
                lstmModel.trainAndTest(sample_weight=self.weights,hijackData=self.getDataAsDict())
                lstmPred = np.argmax(lstmModel.predict(self.X_train),axis=1)
                lstmErr = self.calErr(lstmPred)
                print('At iteration %d, lstmErr: %f' % (i+1,lstmErr))
                alpha = 0.5 * np.log((1 - lstmErr) / lstmErr)
                self.alphas.append(alpha)
                self.classifiers.append(lstmModel)
                self.updateWeight(lstmPred,alpha)


    def predict(self,X):
        sumPred = np.zeros((X.shape[0],2))
        for i in range(len(self.classifiers)):
            sumPred += self.alphas[i] * np.array(self.classifiers[i].predict(X))
        return np.argmax(sumPred,axis=1)

    def eval(self):
        pred = self.predict(self.X_test)
        acc = accuracy_score(self.y_test,pred)
        print('Accuracy: ', acc)


if __name__ == '__main__':
    model = AdaBoost(DataPath='labeled_comments.csv',FileType='csv',classifierName='cnn')
    model.train()
    model.eval()

import tensorflow as tf
import numpy as np
import pandas as pd
from itertools import chain
from sklearn.model_selection import train_test_split

# data = pd.read_csv('labeled_news.csv',header=None)
# y_label = data[0]
# x_data = data.drop(0,axis=1)
#
# tokenizer = tf.keras.preprocessing.text.Tokenizer()
# tokenizer.fit_on_texts(x_data.values[:,0].tolist())
# tokenizer.fit_on_texts(x_data.values[:,1].tolist())
# print(len(tokenizer.word_counts.keys()))
# # print((x_data.values[:,0] + ' ' + x_data.values[:,1]).tolist())
#
# # print(x_data.values[0,0])
# # print(len(x_data.values[0,0]))
# # print([tokenizer.texts_to_sequences(x_data.values[0,0])])
# test = tf.keras.preprocessing.text.text_to_word_sequence(x_data.values[0,0] + ' ' + x_data.values[0,1])
# print(test)
# print(list(chain(*tokenizer.texts_to_sequences(test))))

# add padding token

class DataHandler:
    def __init__(self, filePath, sequence_length, n_class):
        data = pd.read_csv(filePath,header=None)
        y_true = data[0]
        X_data = data.drop(0,axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X_data,y_true,test_size=0.1)
        self.i = 0
        self.sequence_length = sequence_length
        self.X_train = X_train.values
        self.X_test = X_test.values
        self.y_train = y_train.values
        self.y_test = y_test.values
        self.filter = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\''
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(filters=self.filter)
        # self.tokenizer.fit_on_texts((self.X_train[:,0] + ' ' + self.X_train[:,1]).tolist())
        self.tokenizer.fit_on_texts((self.X_train[:,0] + ' ' + self.X_train[:,1] + ' ' + self.X_train[:,2]).tolist())
        self.tokenizer.fit_on_texts((self.X_test[:,0] + ' ' + self.X_test[:,1] + ' ' + self.X_test[:,2]).tolist())
        self.vocab_size = len(self.tokenizer.word_counts.keys()) + 1 # last one used for padding token
        self.n_class = n_class

    def one_hot_encode(self,vec,vals):
        n = vec.shape[0]
        out = np.zeros((n,vals))
        out[range(n),vec-1] = 1
        return out

    def tokenize_strs(self,strs):
        '''
        strs is a list of strings, return a list of indexes of tokenizer sequences
        '''
        res = []
        for s in strs:
            seqs = tf.keras.preprocessing.text.text_to_word_sequence(s,filters=self.filter)
            seqs = list(chain(*self.tokenizer.texts_to_sequences(seqs)))
            # seqs += [self.vocab_size] * (self.sequence_length - len(seqs))
            res.append(seqs)
        # return res
        return tf.keras.preprocessing.sequence.pad_sequences(res,
                                                             maxlen=self.sequence_length,
                                                             padding='post',
                                                             truncating='post',
                                                             value=self.vocab_size-1)

    def next_batch(self,batch_size):
        X = self.X_train[self.i:self.i+batch_size]
        y = self.y_train[self.i:self.i+batch_size]
        self.i = (self.i + batch_size) % len(self.X_train)
        # return (self.tokenize_strs((X[:,0] + ' ' + X[:,1]).tolist()), self.one_hot_encode(y,self.n_class))
        return (self.tokenize_strs((X[:,0] + ' ' + X[:,1] + ' ' + X[:,2]).tolist()), self.one_hot_encode(y,self.n_class))

    def test_batch(self):
        # return self.tokenize_strs((self.X_test[:,0] + ' ' + self.X_test[:,1]).tolist()), self.one_hot_encode(self.y_test,self.n_class)
        return self.tokenize_strs((self.X_test[:,0] + ' ' + self.X_test[:,1] + ' ' + self.X_test[:,2]).tolist()), self.one_hot_encode(self.y_test,self.n_class)


# test = DataHandler(filePath='labeled_news.csv',sequence_length=50,n_class=17)
# print(test.vocab_size)
# print(test.next_batch(5))
# print(test.test_batch())

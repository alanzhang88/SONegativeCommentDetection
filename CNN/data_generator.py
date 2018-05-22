import sys, os
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
sys.path.append(os.path.join(os.path.dirname(__file__), 'embeddings'))
from word2vec_gensim import get_embedding

class DataHandler:
    def __init__(self,
                 embeddingFile='./embeddings/word2vec_vec',
                 trainingFile='./embeddings/processed.csv',
                 maxlength=50,
                 embed_size=100,
                 num_classes=2):
        self.EmbeddingFile = embeddingFile
        self.TrainingFile = trainingFile
        self.maxlen = maxlength
        self.embed_size = embed_size
        self.num_classes = num_classes
        self.tokenizer = Tokenizer()
        data = pd.read_csv(self.TrainingFile,names=['Comment','Label'])
        data = data.sample(frac=1)
        X = data['Comment'].values
        y = data['Label'].values
        self.tokenizer.fit_on_texts(X)
        X = self.tokenizer.texts_to_sequences(X)
        X = pad_sequences(X,maxlen=self.maxlen,padding='post',truncating='post')
        y = to_categorical(y,num_classes=self.num_classes)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X,y,train_size=0.9)
        self.i = 0

    def get_embedding_matrix(self):
        word_index = self.tokenizer.word_index
        nb_words = len(word_index)
        embedding_matrix = np.random.uniform(size=(nb_words+1,self.embed_size))
        embedding_index = get_embedding(self.EmbeddingFile)

        for word,i in word_index.items():
            embedding_vector = embedding_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        return embedding_matrix

    def next_batch(self,batch_size):
        X = self.X_train[self.i:self.i+batch_size]
        y = self.y_train[self.i:self.i+batch_size]
        self.i = (self.i + batch_size) % len(self.X_train)
        return (X,y)

    def get_text_data(self):
        return (self.X_test,self.y_test)
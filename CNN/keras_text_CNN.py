import sys, os
import numpy as np
import pandas as pd
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Embedding, Input

sys.path.append(os.path.join(os.path.dirname(__file__), 'embeddings'))
from word2vec_gensim import get_embedding

EmbeddingFile = './embeddings/word2vec_vec'
TrainingFile = './embeddings/processed.csv'

data = pd.read_csv(TrainingFile,names=['Comment','Label'])
X = data['Comment'].values
y = data['Label'].values

maxlength = 50
embed_size = 100
num_classes = 2

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)
X = tokenizer.texts_to_sequences(X)
X = pad_sequences(X,maxlength=maxlength,padding='post',truncating='post')
y = to_categorical(y,num_classes=num_classes)

word_index = tokenizer.word_index
nb_words = len(word_index)
embedding_matrix = np.random.normal(size=(nb_words,embed_size))
embedding_index = get_embedding(EmbeddingFile)

for word,i in word_index.items():
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

filter_sizes = [2,3,4,5]
num_filters = 20

inp = Input(shape=(maxlength,))
x = Embedding(input_dim=nb_words,output_dim=embed_size,input_length=maxlength,weights=[embedding_matrix],trainable=False)(inp)

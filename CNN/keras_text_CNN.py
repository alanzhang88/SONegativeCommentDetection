import sys, os
import numpy as np
import pandas as pd
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Embedding, Input, Conv2D, MaxPooling2D, Concatenate, Dropout, Flatten, Dense, Reshape
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from sklearn.model_selection import train_test_split

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
X = pad_sequences(X,maxlen=maxlength,padding='post',truncating='post')
y = to_categorical(y,num_classes=num_classes)

word_index = tokenizer.word_index
nb_words = len(word_index)
embedding_matrix = np.random.normal(size=(nb_words+1,embed_size))
embedding_index = get_embedding(EmbeddingFile)

for word,i in word_index.items():
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

filter_sizes = [2,3,4,5,6,7,8]
num_filters = 20
drop_prob = 0.1
lr = 0.0001

inp = Input(shape=(maxlength,))
x = Embedding(input_dim=nb_words+1,output_dim=embed_size,input_length=maxlength,weights=[embedding_matrix],trainable=False)(inp)
x = Reshape((maxlength,embed_size,1))(x)
pooled_output = []
for filter_size in filter_sizes:
    conv = Conv2D(num_filters,kernel_size=(filter_size,embed_size),kernel_initializer='random_normal',activation='relu')(x)
    max_pooled = MaxPooling2D(pool_size=(maxlength - filter_size + 1, 1))(conv)
    pooled_output.append(max_pooled)

z = Concatenate(axis=1)(pooled_output)
z = Flatten()(z)
z = Dropout(rate=drop_prob)(z)
outp = Dense(num_classes,activation='sigmoid')(z)
model = Model(inputs=inp,outputs=outp)
model.compile(optimizer=Adam(lr=lr),loss=categorical_crossentropy,metrics=['accuracy'])

batch_size = 64
epochs = 3

X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.9)
hist = model.fit(X_train,y_train,batch_size=batch_size,epochs=epochs,validation_data=(X_test,y_test),verbose=2)

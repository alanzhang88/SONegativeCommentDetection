from keras.models import Model
from keras.layers import Embedding, Input, Conv2D, MaxPooling2D, Concatenate, Dropout, Flatten, Dense, Reshape,SpatialDropout1D
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from keras.callbacks import Callback
# from data_generator import DataHandler, clean_data
import data_generator
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')
from embedding_config import config
from keras.models import load_model
from saveModel import SaveModel
import numpy as np
import json

# from sklearn.metrics import precision_recall_fscore_support

#For predict purposes
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


EMBEDDING_CONFIGS = config.embedding_configs

class CNNModel:

    def __init__(self, num_filters=256, filter_sizes=[4], drop_prob=0.2, lr=0.001, batch_size=32, epochs=10, max_length=50, num_classes=2, embed_size=100,save_model=True,random_state=None, activation="sigmoid"):
        self.num_filters = num_filters
        self.filter_sizes = filter_sizes
        self.drop_prob = drop_prob
        self.lr = lr
        self.max_length = max_length
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.epochs = epochs
        self.embed_size = embed_size
        self.model = None
        self.save_model = save_model
        self.data = data_generator.DataHandler(embeddingFile=EMBEDDING_CONFIGS[0].embedding_file,
                   trainingFile= config.training_file,
                   maxlength=self.max_length,
                   embed_size=self.embed_size,
                   num_classes=self.num_classes,
                   random_state=random_state)
        self.activation = activation



    def build_model(self):

        embedding_matrix = self.data.get_embedding_matrix()
        inp = Input(shape=(self.max_length,))
        x = Embedding(input_dim=embedding_matrix.shape[0],output_dim=self.embed_size,input_length=self.max_length,weights=[embedding_matrix])(inp)
        x = Reshape((self.max_length,self.embed_size,1))(x)
        pooled_output = []
        for filter_size in self.filter_sizes:
            conv = Conv2D(self.num_filters,kernel_size=(filter_size,self.embed_size),kernel_initializer='random_normal',activation=self.activation)(x)
            max_pooled = MaxPooling2D(pool_size=(self.max_length - filter_size + 1, 1))(conv)
            pooled_output.append(max_pooled)

        z = Concatenate(axis=1)(pooled_output) if len(pooled_output) > 1 else pooled_output[0]
        z = Flatten()(z)
        z = Dropout(rate=self.drop_prob)(z)
        outp = Dense(self.num_classes,kernel_initializer='random_normal',activation='sigmoid')(z)
        self.model = Model(inputs=inp,outputs=outp)
        self.model.compile(optimizer=Adam(lr=self.lr),loss=categorical_crossentropy,metrics=['accuracy',self.TNR, self.precision, self.f1_score, self.TN, self.FP, self.FN, self.TP])
        X_test, y_test = self.data.get_test_data()
        X_train, y_train = self.data.get_train_data()
        savemodel = SaveModel(validation_data=(X_test,y_test),target_name='f1_score',target_val=0.65)
        callbacks = [savemodel] if self.save_model else None
        history = self.model.fit(x=X_train,y=y_train,batch_size=self.batch_size,epochs=self.epochs,verbose=2,validation_data=(X_test,y_test),callbacks=callbacks)
        return history

    
    def TNR(self, y_true, y_pred):
        mat = tf.confusion_matrix(labels=tf.argmax(y_true, 1),predictions=tf.argmax(y_pred, 1),num_classes=self.num_classes)
        return mat[0][0] / (mat[0][0] + mat[0][1])
    
    def precision (self, y_true, y_pred):
        mat = tf.confusion_matrix(labels=tf.argmax(y_true, 1),predictions=tf.argmax(y_pred, 1),num_classes=self.num_classes)
        return mat[0][0] / (mat[0][0] + mat[1][0])

    def f1_score(self, y_true, y_pred):
        recall = self.TNR(y_true, y_pred)
        precision = self.precision(y_true, y_pred)
        f1_score = (2 * recall * precision) /(precision + recall)
        return f1_score 

    
    def TN(self, y_true, y_pred):
        mat = tf.confusion_matrix(labels=tf.argmax(y_true, 1),predictions=tf.argmax(y_pred, 1),num_classes=self.num_classes)
        return mat[0][0]

    def FP(self, y_true, y_pred):
        mat = tf.confusion_matrix(labels=tf.argmax(y_true, 1),predictions=tf.argmax(y_pred, 1),num_classes=self.num_classes)
        return mat[0][1]

    def FN(self, y_true, y_pred):
        mat = tf.confusion_matrix(labels=tf.argmax(y_true, 1),predictions=tf.argmax(y_pred, 1),num_classes=self.num_classes)
        return mat[1][0]

    def TP(self, y_true, y_pred):
        mat = tf.confusion_matrix(labels=tf.argmax(y_true, 1),predictions=tf.argmax(y_pred, 1),num_classes=self.num_classes)
        return mat[1][1]


    def load_model(self, filePath):
        self.model = load_model(filePath, custom_objects={"TNR":self.TNR, "precision":self.precision, "f1_score":self.f1_score, 'TN':self.TN, 'FP':self.FP, 'FN':self.FN, 'TP':self.TP})

    #input: list of string
    def predict(self, commentList):
        #preprocess data
        comments = self.data.process_new_data(commentList)
        res = self.model.predict(comments)
        print (res)
        return [(np.array(l)/sum(l)).tolist() for l in res]
    

if __name__ == "__main__":
    CNN_model = CNNModel()
    # CNN_model.load_model("./CNNmodel.h5")
    CNN_model.build_model()
    CNN_model.load_model("./CNNmodel.h5")
    print (CNN_model.predict(["Google you homeword first"]))
   
   
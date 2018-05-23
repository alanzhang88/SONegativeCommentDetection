from keras.models import Model
from keras.layers import Embedding, Input, Conv2D, MaxPooling2D, Concatenate, Dropout, Flatten, Dense, Reshape,SpatialDropout1D
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from keras.callbacks import Callback
from data_generator import DataHandler
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')
from embedding_config import config


EMBEDDING_CONFIGS = config.embedding_configs


#EmbeddingFile = './embeddings/word2vec_vec'
#EmbeddingFile = './embeddings/glove.twitter.27B/glove.twitter.27B.100d.txt'
TrainingFile = './embeddings/processed.csv'

maxlength = 50
#maxlength = 200
embed_size = 100
num_classes = 2

data = DataHandler(embeddingFile=EMBEDDING_CONFIGS[0].embedding_file,
                   trainingFile=TrainingFile,
                   maxlength=maxlength,
                   embed_size=embed_size,
                   num_classes=num_classes)
# data = DataHandler(embeddingFile=EmbeddingFile,
#                    trainingFile=TrainingFile,
#                    maxlength=maxlength,
#                    embed_size=embed_size,
#                    num_classes=num_classes)

embedding_matrix = data.get_embedding_matrix()

filter_sizes = [4,5,6,7]

num_filters = 32
#num_filters = 512
drop_prob = 0.2
#drop_prob = 0.1
lr = 0.001

class SaveModel(Callback):
    def __init__(self,target_name,target_val,validation_data=()):
        super(Callback,self).__init__()
        self.validation_data = validation_data
        self.target_val = target_val
        self.target_name = target_name

    def on_epoch_end(self,epoch,logs={}):
        res = self.model.test_on_batch(self.validation_data[0],self.validation_data[1])
        last = -1
        for i in range(len(self.model.metrics_names)):
            if self.model.metrics_names[i] == self.target_name and res[i] >= self.target_val and res[i] > last:
                last = res[i]
                print('Saveing model with %s reaching %f' % (self.target_name,res[i]))
                model.save(filepath='./model.h5')

def fpp(y_true,y_pred):
    mat = tf.confusion_matrix(labels=tf.argmax(y_true,1),predictions=tf.argmax(y_pred,1),num_classes=num_classes)
    return mat[0][1] / (mat[0][1] + mat[1][1])

inp = Input(shape=(maxlength,))
x = Embedding(input_dim=embedding_matrix.shape[0],output_dim=embed_size,input_length=maxlength,weights=[embedding_matrix])(inp)

# x = SpatialDropout1D(0.4)(x)

x = Reshape((maxlength,embed_size,1))(x)
pooled_output = []
for filter_size in filter_sizes:
    conv = Conv2D(num_filters,kernel_size=(filter_size,embed_size),kernel_initializer='random_normal',activation='relu')(x)
    max_pooled = MaxPooling2D(pool_size=(maxlength - filter_size + 1, 1))(conv)
    pooled_output.append(max_pooled)

z = Concatenate(axis=1)(pooled_output)
z = Flatten()(z)
z = Dropout(rate=drop_prob)(z)
outp = Dense(num_classes,kernel_initializer='random_normal',activation='sigmoid')(z)
model = Model(inputs=inp,outputs=outp)
model.compile(optimizer=Adam(lr=lr),loss=categorical_crossentropy,metrics=['accuracy',fpp])

batch_size = 128
epochs = 20
steps = 1000
X_test, y_test = data.get_test_data()
X_train, y_train = data.get_train_data()
savemodel = SaveModel(validation_data=(X_test,y_test),target_name='acc',target_val=0.65)

model.fit(x=X_train,y=y_train,batch_size=batch_size,epochs=epochs,verbose=2,validation_data=(X_test,y_test),callbacks=[savemodel])

# for i in range(steps):
#     X_train, y_train = data.next_batch(batch_size)
#     model.train_on_batch(X_train,y_train)
#     if i % 100 == 0:
#         res = model.test_on_batch(X_test,y_test)
#         print('On step %d' % i)
#         print('Accuracy: %f' % res[1])
#         print('FPP: %f \n' % res[2])
#
# res = model.test_on_batch(X_test,y_test)
# print('On step %d' % 1000)
# print('Accuracy: %f' % res[1])
# print('FPP: %f \n' % res[2])

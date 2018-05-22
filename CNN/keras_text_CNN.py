from keras.models import Model
from keras.layers import Embedding, Input, Conv2D, MaxPooling2D, Concatenate, Dropout, Flatten, Dense, Reshape
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from data_generator import DataHandler

EmbeddingFile = './embeddings/word2vec_vec'
TrainingFile = './embeddings/processed.csv'

maxlength = 50
embed_size = 100
num_classes = 2
data = DataHandler(embeddingFile=EmbeddingFile,
                   trainingFile=TrainingFile,
                   maxlength=maxlength,
                   embed_size=embed_size,
                   num_classes=num_classes)

embedding_matrix = data.get_embedding_matrix()

filter_sizes = [4,5,6,7]
num_filters = 32
drop_prob = 0.2
lr = 0.001

inp = Input(shape=(maxlength,))
x = Embedding(input_dim=embedding_matrix.shape[0],output_dim=embed_size,input_length=maxlength,weights=[embedding_matrix],trainable=False)(inp)
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

batch_size = 128
steps = 1000
X_test, y_test = data.get_text_data()

for i in range(steps):
    X_train, y_train = data.next_batch(batch_size)
    model.train_on_batch(X_train,y_train)
    if i % 100 == 0:
        res = model.test_on_batch(X_test,y_test)
        print('On step %d' % i)
        print('Accuracy: %f \n' % res[1])

res = model.test_on_batch(X_test,y_test)
print('On step %d' % 1000)
print('Accuracy: %f \n' % res[1])

from MongodbClient import MyMongoClient
import sys, os
import numpy as np

# #import all models
sys.path.append(os.path.join(os.path.dirname(__file__), 'Models','LSTM'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'Models','CNN'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'Models','FastText'))

from LSTMUtil import LSTM
from CNNUtil import CNNModel
from FastTextUtil import FastText

# from Models.LSTM.LSTMUtil import LSTM
# from Models.CNN.CNNUtil import CNNModel
# from Models.FastText.FastTextUtil import FastText

#weight: 0.85, 0.65, 0.65

model_weights = np.array([0.85, 0.65, 0.65])
normalized_weights = np.array([np.array(item)/np.sum(model_weights) for item in model_weights])

#multiply with list of list  * weight
collectionName = sys.argv[1] if len(sys.argv) > 1 else "PostFirstIter"

client = MyMongoClient()
collection = client.get_collection(collectionName)
it = collection.aggregate([{'$match':{'CommentsLabel':{'$exists':False},'FirstIterCommentsLabel':{'$exists':False},'Score':{'$lte':-5},'CommentCount':{'$gt':0}}},{'$sample':{'size':1000}}])


model_labels = {}
cnn_lables = []
lstm_labels = []
fasttext_labels = []


#load models
lstm_model = LSTM()
cnn_model = CNNModel()
#load CNN model
cnn_model.load_model("Models/CNN/CNNmodel.h5")
fasttext_model = FastText()

count = 0
#build three diff models
for doc in it:
    print('Predicting PostId %d with %d comments' % (doc['Id'],doc['CommentCount']))
    commentsLabel = {}
    all_labels = []

    lstm_set = []
    cnn_set = []
    fasttext_set = []

    lstm_label = lstm_model.predict(doc['Comments'])
    cnn_label = cnn_model.predict(doc['Comments'])
    fasttext_label = fasttext_model.predict(doc['Comments'])

    for i in range(len(cnn_label)):


        lstm_set.append(np.asscalar(np.argmax(lstm_label[i])))
        cnn_set.append(np.asscalar(np.argmax(cnn_label[i])))
        fasttext_set.append(np.asscalar(np.argmax(np.asarray(fasttext_label[i]))))



    for i in range(len(cnn_label)):
        l1 = np.multiply(lstm_label[i],normalized_weights[0])
        l2 = np.multiply(cnn_label[i],normalized_weights[1])
        inter_1= np.add(l1, l2)
        l3 = np.multiply(np.asarray(fasttext_label[i]),normalized_weights[2])
        inter_2 = np.add(inter_1, l3)
        max_label = np.argmax(inter_2)
        # negative comments
        if max_label  == 0:
            all_labels.append(0)
        else:
            all_labels.append(1)

    commentsLabel = {"LSTM": lstm_set, "CNN" : cnn_set, "FastText" : fasttext_set, "All" : all_labels}

    doc['FirstIterCommentsLabel'] = commentsLabel
    collection.save(doc)
    count += 1
    print ("Labeled %d out of 1000" % count)

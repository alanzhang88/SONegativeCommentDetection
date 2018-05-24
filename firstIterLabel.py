from MongodbClient import MyMongoClient
import sys, os
import numpy as np

# #import all models
# sys.path.append(os.path.join(os.path.dirname(__file__), 'Models/LSTM'))
# sys.path.append(os.path.join(os.path.dirname(__file__), 'Models/FastText'))
# sys.path.append(os.path.join(os.path.dirname(__file__), 'Models/CNN'))

# from LSTMUtil import LSTM
# from CNNUtil import CNNModel
# from FastTextUtil import FastText

from Models.LSTM.LSTMUtil import LSTM
from Models.CNN.CNNUtil import CNNModel
from Models.FastText.FastTextUtil import FastText

#weight: 0.85, 0.65, 0.65

model_weights = np.array([0.85, 0.65, 0.65])
normalized_weights = np.array([np.array(item)/np.sum(model_weights) for item in model_weights])

#multiply with list of list  * weight
collectionName = sys.argv[1] if len(sys.argv) > 1 else "PostFirstIter"

client = MyMongoClient()
collection = client.get_collection(collectionName)
it = collection.aggregate([{'$match':{'CommentsLabel':{'$exists':False},'FirstIterCommentsLabel':{'$exists':False},'Score':{'$lte':-5},'CommentCount':{'$gt':0}}},{'$sample':{'size':1000}}])


#load models
lstm_model = LSTM()
cnn_model = CNNModel()
#load CNN model
cnn_model.load_model("Models/CNN/CNNmodel.h5")
fasttext_model = FastText()

#build three diff models
for doc in it:
    print('Predicting PostId %d with %d comments' % (doc['Id'],doc['CommentCount']))
    commentsLabel = []
    lstm_label = lstm_model.predict(doc['Comments'])
    cnn_label = cnn_model.predict(doc['Comments'])
    fasttext_label = fasttext_model.predict(doc['Comments'])

    for i in range(len(cnn_label)):
        l1 = np.multiply(lstm_label[i],normalized_weights[0])
        l2 = np.multiply(cnn_label[i],normalized_weights[1])
        inter_1= np.add(l1, l2)
        l3 = np.multiply(np.asarray(fasttext_label[i]),normalized_weights[2])
        inter_2 = np.add(inter_1, l3)
        max_label = np.amax(inter_2)
        label = inter_2.tolist().index(max_label)
        # negative comments
        if label  == 0:
            commentsLabel.append(0)
        else:
            commentsLabel.append(0)

    doc['FirstIterCommentsLabel'] = commentsLabel
    print (commentsLabel)
    collection.save(doc)

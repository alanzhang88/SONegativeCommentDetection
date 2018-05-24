from MongodbClient import MyMongoClient
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), 'Models/LSTM'))
#import all models
sys.path.append(os.path.join(os.path.dirname(__file__), 'Models/CNN'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'Models/FastText'))
from LSTMUtil import LSTM
from CNNUtil import CNNModel


#weight: 0.8, 0.65, 
#normalize weight

#multiple with list of list  * weight

collectionName = sys.argv[1] if len(sys.argv) > 1 else "PostFirstIter"

client = MyMongoClient()
collection = client.get_collection(collectionName)

it = collection.aggregate([{'$match':{'CommentsLabel':{'$exists':False},'FirstIterCommentsLabel':{'$exists':False},'Score':{'$lte':-5},'CommentCount':{'$gt':0}}},{'$sample':{'size':1000}}])

for doc in it:
    print('Predicting PostId %d with %d comments' % (doc['Id'],doc['CommentCount']))
    commentsLabel = []
    # for c in doc['Comments']:
    #     print('Predicting Comment: %s' % c)
    #     #TODO: Predictions
    #     predictLabel = 0
    #     commentsLabel.append(predictLabel)

    # TODO: prdict a list of labels, assign to commentsLabel
    # commentsLabel = 
    doc['FirstIterCommentsLabel'] = commentsLabel
    collection.save(doc)

from MongodbClient import MyMongoClient
import sys

collectionName = sys.argv[1] if len(sys.argv) > 1 else "PostFirstIter"

client = MyMongoClient()
collection = client.get_collection(collectionName)

it = collection.aggregate([{'$match':{'CommentsLabel':{'$exists':False},'FirstIterCommentsLabel':{'$exists':False},'Score':{'$lte':-5},'CommentCount':{'$gt':0}}},{'$sample':{'size':1000}}])

for doc in it:
    print('Predicting PostId %d with %d comments' % (doc['Id'],doc['CommentCount']))
    commentsLabel = []
    for c in doc['Comments']:
        print('Predicting Comment: %s' % c)
        #TODO: Predictions
        predictLabel = 0
        commentsLabel.append(predictLabel)
    doc['FirstIterCommentsLabel'] = commentsLabel
    collection.save(doc)

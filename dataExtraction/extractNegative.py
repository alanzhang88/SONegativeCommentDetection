from MongodbClient import MyMongoClient
import pandas as pd
import numpy as np

client = MyMongoClient()
collection = client.get_collection('PostFirstIter')

it = collection.find({'CommentsLabel':{'$exists':True}},{'CommentsLabel':1,'Comments':1})
arr = None
for doc in it:
    for i in range(len(doc['CommentsLabel'])):
        if doc['CommentsLabel'][i] == 0:
            if arr is None:
                arr = np.array([[doc['Comments'][i],doc['CommentsLabel'][i]]])
            else:
                arr = np.append(arr,[[doc['Comments'][i],doc['CommentsLabel'][i]]],axis=0)

data = pd.DataFrame(arr)
data.to_csv(path_or_buf='./neg.csv',header=False,index=False)

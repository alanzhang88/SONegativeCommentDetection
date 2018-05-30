MONGODB_URI = ['mongodb://cs230:1234@ds014658.mlab.com:14658/cs230',
               'mongodb://cs230:1234@ds014648.mlab.com:14648/cs230db1',
               'mongodb://cs230:1234@ds014648.mlab.com:14648/cs230db2',
               'mongodb://cs230:1234@ds014648.mlab.com:14648/cs230db3']
DB_NAME = ['cs230','cs230db1','cs230db2','cs230db3']
collection_name = ['PostFirstIter','PostFirstIter','PostFirstIter_0','PostFirstIter']

from MongodbClient import MyMongoClient
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

client = MyMongoClient(MONGODB_URI=MONGODB_URI,DB_NAME=DB_NAME)

y_true = []
y_pred = {'lstm':[],'cnn':[],'fasttext':[],'mixed':[]}


for i in range(len(collection_name)):
    collection = client.get_collection(collection_name[i])
    it = collection.find({'CommentsLabel':{'$exists':True},'FirstIterCommentsLabel':{'$exists':True}},{'CommentsLabel':1,'FirstIterCommentsLabel':1})
    for doc in it:
        y_true = y_true + doc['CommentsLabel']
        y_pred['lstm'] = y_pred['lstm'] + doc['FirstIterCommentsLabel']['LSTM']
        y_pred['cnn'] = y_pred['cnn'] + doc['FirstIterCommentsLabel']['CNN']
        y_pred['fasttext'] = y_pred['fasttext'] + doc['FirstIterCommentsLabel']['FastText']
        y_pred['mixed'] = y_pred['mixed'] + doc['FirstIterCommentsLabel']['All']
    client.switch_db()


lstmAcc = accuracy_score(y_true,y_pred['lstm'])
lstmPre = precision_score(y_true,y_pred['lstm'])
lstmRec = recall_score(y_true,y_pred['lstm'])
lstmF1 = f1_score(y_true,y_pred['lstm'])
lstmCon = confusion_matrix(y_true,y_pred['lstm'])
print('LSTM Accuracy: %f, Precision: %f, Recall: %f, F1: %f' %(lstmAcc,lstmPre,lstmRec,lstmF1))
print('LSTM Confusion Matrix')
print(lstmCon)

cnnAcc = accuracy_score(y_true,y_pred['cnn'])
cnnPre = precision_score(y_true,y_pred['cnn'])
cnnRec = recall_score(y_true,y_pred['cnn'])
cnnF1 = f1_score(y_true,y_pred['cnn'])
cnnCon = confusion_matrix(y_true,y_pred['cnn'])
print('CNN Accuracy: %f, Precision: %f, Recall: %f, F1: %f' %(cnnAcc,cnnPre,cnnRec,cnnF1))
print('CNN Confusion Matrix')
print(cnnCon)

fasttextAcc = accuracy_score(y_true,y_pred['fasttext'])
fasttextPre = precision_score(y_true,y_pred['fasttext'])
fasttextRec = recall_score(y_true,y_pred['fasttext'])
fasttextF1 = f1_score(y_true,y_pred['fasttext'])
fasttextCon = confusion_matrix(y_true,y_pred['fasttext'])
print('FastText Accuracy: %f, Precision: %f, Recall: %f, F1: %f' %(fasttextAcc,fasttextPre,fasttextRec,fasttextF1))
print('FastText Confusion Matrix')
print(fasttextCon)

mixedAcc = accuracy_score(y_true,y_pred['mixed'])
mixedPre = precision_score(y_true,y_pred['mixed'])
mixedRec = recall_score(y_true,y_pred['mixed'])
mixedF1 = f1_score(y_true,y_pred['mixed'])
mixedCon = confusion_matrix(y_true,y_pred['mixed'])
print('Mixed Accuracy: %f, Precision: %f, Recall: %f, F1: %f' %(mixedAcc,mixedPre,mixedRec,mixedF1))
print('Mixed Confusion Matrix')
print(mixedCon)

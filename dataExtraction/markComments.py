from MongodbClient import MyMongoClient
import sys
import pymongo
import argparse

# if len(sys.argv) <= 2:
#     print('Please input collection name and startId')
#     exit(1)

# collectionName = sys.argv[1]
# startId = int(sys.argv[2])

parser = argparse.ArgumentParser()
parser.add_argument('-c','--collection',dest='collectionName',help='Collection Name',type=str,required=True)
parser.add_argument('-i','--start',dest='startId',type=int,help='start id to label',required=True)
parser.add_argument('-s','--score',dest='score',type=int,help='upper bound score of post',required=True)
parser.add_argument('--sort',dest='sort',action='store_true',help='whether sort the output in Id Ascending order',default=False)
parser.add_argument('--sortscore',dest='sortscore',action='store_true',help='whether sort the output in Score Ascending order',default=False)
parser.add_argument('--zerocomment',dest='comment',action='store_true',help='whether include posts with zero comment',default=False)
args = parser.parse_args()

client = MyMongoClient()
collection = client.get_collection(args.collectionName)
if args.comment:
    commentCount = -1
else:
    commentCount = 0

collection.create_index('Id')

labledNum = collection.find({'BodyLabel':{'$exists':True}}).count()
print('Already labled %d documents' % labledNum)
if args.sort:
    it = collection.find({'$and':[{'Id':{'$gte':args.startId}},{'Score':{'$lte':args.score}},{'CommentCount':{'$gt':commentCount}},{'BodyLabel':{'$exists':False}}]}).sort('Id',pymongo.ASCENDING).limit(1000)
else:
    # it = collection.find({'$and':[{'Id':{'$gte':args.startId}},{'Score':{'$lte':args.score}},{'CommentCount':{'$gt':commentCount}},{'BodyLabel':{'$exists':False}}]}).limit(1000)
    it = collection.aggregate([{'$match':{'$and':[{'Id':{'$gte':args.startId}},{'Score':{'$lte':args.score}},{'CommentCount':{'$gt':commentCount}},{'BodyLabel':{'$exists':False}}]}},{'$sample':{'size':1000}}])
if args.sortscore:
    it = collection.find({'$and':[{'Id':{'$gte':args.startId}},{'Score':{'$lte':args.score}},{'CommentCount':{'$gt':commentCount}},{'BodyLabel':{'$exists':False}}]}).sort('Score',pymongo.ASCENDING).limit(1000)
else:
    it = collection.aggregate([{'$match':{'$and':[{'Id':{'$gte':args.startId}},{'Score':{'$lte':args.score}},{'CommentCount':{'$gt':commentCount}},{'BodyLabel':{'$exists':False}}]}},{'$sample':{'size':1000}}])
for doc in it:
    print('Currently View Comments from PostId %d with score %d and commentCount %d' % (doc['Id'],doc['Score'],doc['CommentCount']))
    print('Post Body: ')
    print(doc['Body'])
    label = None
    while label != '0' and label != '1':
        label = input('Input body label(0 for negative, 1 for otherwise)')
    bodyLabel = int(label)
    print('\n')
    doc['BodyLabel'] = bodyLabel
    if 'Comments' not in doc.keys():
        collection.save(doc)
        print('No Comments to label')
        continue
    count = 1
    commentLabels = []
    for c in doc['Comments']:
        print('Comment %d:' % count)
        print(c)
        label = None
        while label != '0' and label != '1':
            label = input('Input comment label(0 for negative, 1 for otherwise)')
        commentLabels.append(int(label))
        count += 1
        print('\n')

    doc['CommentsLabel'] = commentLabels
    collection.save(doc)

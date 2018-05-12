from MongodbClient import MyMongoClient
import sys
import pymongo

if len(sys.argv) <= 2:
    print('Please input collection name and startId')
    exit(1)

collectionName = sys.argv[1]
startId = int(sys.argv[2])
client = MyMongoClient()
collection = client.get_collection(collectionName)



collection.create_index('Id')
for doc in collection.find({'Id':{'$gte':startId}}).sort('Id',pymongo.ASCENDING):
    print('Currently View Comments from PostId %d' % doc['Id'])
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

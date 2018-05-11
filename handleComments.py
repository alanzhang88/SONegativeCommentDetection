from xml.etree.ElementTree import XMLPullParser
from MongodbClient import MyMongoClient
import sys

collectionName = sys.argv[1] if len(sys.argv) > 1 else 'PostFirstIter'
dbIdx = int(sys.argv[2]) if len(sys.argv) > 2 else 0
startId = int(sys.argv[3]) if len(sys.argv) > 3 else None
endId = int(sys.argv[4]) if len(sys.argv) > 4 else None

client = MyMongoClient()
if not client.set_db(dbIdx):
    exit(0)
collection = client.get_collection(collectionName)
# CommentFilePath = './Data/Comments.xml'
CommentFilePath = '/Volumes/Untitled/230Data/Comments.xml'
# grab a list of valid PostId from DB and store in set
res = []
if startId is None or endId is None:
    res = list(collection.find({'$and':[{'CommentCount':{'$ne':0}},{'Comments':{'$exists':False}}]},{'Id':1,'CommentCount':1}))
else:
    res = list(collection.find({'$and':[{'CommentCount':{'$ne':0}},{'Comments':{'$exists':False}},{'Id':{'$gte':startId}},{'Id':{'$lt':endId}}]},{'Id':1,'CommentCount':1}))
print('Found %d entries from DB' % len(res))
entriesNum = len(res)
postIdset = set()
commentCount = dict()
for d in res:
        postIdset.add(d['Id'])
        commentCount[d['Id']] = d['CommentCount']
comments = {}
parser = XMLPullParser(events=['end'])
with open(file=CommentFilePath) as f:
    counter = 0 #Things to fix, something wrong with the parser, we need to put line contraints on it
    for line in f:
        # if counter <= 1:
        #     parser.feed(line)
        counter += 1
        if counter % 1000000 == 0:
            print('At line %d' % counter)
            parser.feed('</comments>')
            parser.close()
            parser = XMLPullParser(events=['end'])
            parser.feed('<comments>')
        # if counter <= 56000000:
        #     continue
        # if counter > 56200000:
        #     break
        if entriesNum == 0:
            break
        parser.feed(line)
        for event,elem in parser.read_events():
            if(elem.tag == 'row'):
                postId = int(elem.get('PostId'))
                if postId in postIdset:
                    if postId in comments.keys():
                        comments[postId].append(elem.get('Text'))
                    else:
                        comments[postId] = [elem.get('Text')]
                    commentCount[postId] -= 1
                    if commentCount[postId] == 0:
                        print('PostId %s comments are all found' % postId)
                        collection.find_one_and_update({"Id": postId},{'$set':{'Comments':comments[postId]}})
                        del comments[postId]
                        del commentCount[postId]
                        entriesNum -= 1
                        continue
                    # if len(comments[postId]) >= 3:
                    #     print('PostId %s comments are partially found, %d comments left to found' % (postId,commentCount[postId]))
                    #     collection.find_one_and_update({"Id": postId},{'$push':{'Comments':{'$each': comments[postId]}}})
                    #     comments[postId] = []
    f.close()

# for id in comments.keys():
    # collection.find_one_and_update({"Id": id},{'$push':{'Comments':{'$each': comments[id]}}})
    # collection.find_one_and_update({"Id": postId},{'$set':{'Comments':comments[id]}})

# if collectionName == 'PostsWithNoAnswer':
#     dset = postIdset - set(comments.keys())
#     for id in dset:
#         collection.find_one_and_delete({'Id':id})

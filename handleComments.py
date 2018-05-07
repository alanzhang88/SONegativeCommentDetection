from xml.etree.ElementTree import XMLPullParser
from MongodbClient import MyMongoClient
import sys

collectionName = sys.argv[1] if len(sys.argv) > 1 else 'PostsWithNoAnswer'
dbIdx = int(sys.argv[2]) if len(sys.argv) > 2 else 0

client = MyMongoClient()
if not client.set_db(dbIdx):
    exit(0)
collection = client.get_collection(collectionName)
CommentFilePath = './Data/Comments.xml'

# grab a list of valid PostId from DB and store in set
res = list(collection.find({},{'Id':1,'CommentCount':1}))
postIdset = set()
commentCount = dict()
for d in res:
    cCount = int(d['CommentCount'])
    if cCount > 0:
        postIdset.add(d['Id'])
        commentCount[d['Id']] = cCount
comments = {}
parser = XMLPullParser(events=['end'])
with open(file=CommentFilePath) as f:
    for line in f:
        if len(commentCount.keys()) == 0:
            break
        parser.feed(line)
        for event,elem in parser.read_events():
            if(elem.tag == 'row'):
                postId = elem.get('PostId')
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
    f.close()

for id in comments.keys():
    # collection.find_one_and_update({"Id": id},{'$push':{'Comments':{'$each': comments[id]}}})
    collection.find_one_and_update({"Id": postId},{'$set':{'Comments':comments[id]}})

# if collectionName == 'PostsWithNoAnswer':
#     dset = postIdset - set(comments.keys())
#     for id in dset:
#         collection.find_one_and_delete({'Id':id})

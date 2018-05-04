from xml.etree.ElementTree import XMLPullParser
from MongodbClient import get_collection

collection = get_collection('PostsWithNoAnswer')
CommentFilePath = './sample/Comments.xml'

# grab a list of valid PostId from DB and store in set
postIdset = set([d['Id'] for d in list(collection.find({},{'Id':1}))])
comments = {}
parser = XMLPullParser(events=['end'])
with open(file=CommentFilePath) as f:
    for line in f:
        parser.feed(line)
        for event,elem in parser.read_events():
            if(elem.tag == 'row'):
                postId = elem.get('PostId')
                if postId in postIdset:
                    if postId in comments.keys():
                        comments[postId].append(elem.get('Text'))
                    else:
                        comments[postId] = [elem.get('Text')]
    f.close()

# collection.find_one_and_update({"Id": postId},{'$set':{'Comments':comments}})
for id in comments.keys():
    collection.find_one_and_update({"Id": id},{'$push':{'Comments':{'$each': comments[id]}}})

dset = postIdset - set(comments.keys())
for id in dset:
    collection.find_one_and_delete({'Id':id})

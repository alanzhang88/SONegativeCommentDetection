from xml.etree.ElementTree import XMLPullParser
from MongodbClient import MyMongoClient
import sys

if len(sys.argv) < 3:
    print('Not enough args')
    exit(0)

collectionName = sys.argv[1]
fieldsToFill = sys.argv[2:]

client = MyMongoClient()
collection = client.get_collection(collectionName)

PostsFilePath = './Data/Posts.xml'

postIdset = set([d['Id'] for d in list(collection.find({},{'Id':1}))])
print('Try to update %d data entries' % len(postIdset))

parser = XMLPullParser(events=['end'])
with open(file=PostsFilePath) as f:
    counter = 0
    for line in f:
        if len(postIdset) == 0:
            break
        counter += 1
        if counter % 1000000 == 0:
            parser.feed('</posts>')
            parser.close()
            parser = XMLPullParser(events=['end'])
            parser.feed('<posts>')
        parser.feed(line)
        for event,elem in parser.read_events():
            if(elem.tag == 'row'):
                Id = int(elem.get('Id'))
                if Id in postIdset:
                    print('Updating postId %s' % Id)
                    data_to_fill = {field:elem.get(field) for field in fieldsToFill}
                    collection.find_one_and_update({'Id':Id},{'$set':data_to_fill})
                    postIdset.remove(Id)
    f.close()

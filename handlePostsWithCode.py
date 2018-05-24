#Find posts whose body contains code segment

from xml.etree.ElementTree import XMLPullParser
from MongodbClient import MyMongoClient
import sys

client = MyMongoClient()
collection = client.get_collection('PostsWithCodeBlock')
list_of_str_keys = ['Id','Body']
PostsFilePath = './Data/Posts.xml'
startId = int(sys.argv[1]) if len(sys.argv) > 1 else 0
dbThreshold = int(sys.argv[2]) if len(sys.argv) > 2 else None
nextSwitchId = startId + dbThreshold if dbThreshold is not None else None

parser = XMLPullParser(events=['end'])
with open(file=PostsFilePath) as f:
    Id = 0
    counter = 0
    for line in f:
        counter += 1
        if counter % 1000000 == 0:
            parser.feed('</posts>')
            parser.close()
            parser = XMLPullParser(events=['end'])
            parser.feed('<posts>')
        parser.feed(line)
        for event,elem in parser.read_events():
            if (elem.tag == "row"):
                Id = int(elem.get('Id'))
                if Id < startId:
                    continue
                postBody = str(elem.get('Body'))
                print (postBody)
    f.close()



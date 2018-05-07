from xml.etree.ElementTree import XMLPullParser
from MongodbClient import MyMongoClient
import sys

client = MyMongoClient()
collection = client.get_collection('PostsWithNoAnswer')
list_of_str_keys = ['Id','Body']

PostsFilePath = './Data/Posts.xml'
startId = int(sys.argv[1]) if len(sys.argv) > 1 else 0
dbThreshold = int(sys.argv[2]) if len(sys.argv) > 2 else None
nextSwitchId = startId + dbThreshold if dbThreshold is not None else None

parser = XMLPullParser(events=['end'])
with open(file=PostsFilePath) as f:
    Id = 0
    for line in f:
        parser.feed(line)
        for event,elem in parser.read_events():
            if(elem.tag == 'row'):
                Id = int(elem.get('Id'))
                if Id < startId:
                    continue
                postTypeId = int(elem.get('PostTypeId'))
                if postTypeId == 1:
                    answerCount = int(elem.get('AnswerCount'))
                    commentCount = int(elem.get('CommentCount'))
                    if(answerCount == 0 and commentCount > 0):
                        data_to_save = {key: elem.get(key) for key in list_of_str_keys}
                        data_to_save['CommentCount'] = int(data_to_save['CommentCount'])
                        data_to_save['Score'] = int(data_to_save['Score'])
                        data_to_save['ViewCount'] = int(elem.get('ViewCount'))
                        collection.insert_one(data_to_save)
        if nextSwitchId is not None and Id >= nextSwitchId:
            nextSwitchId += dbThreshold
            ret = client.switch_db()
            if ret:
                collection = client.get_collection('PostsWithNoAnswer')
            else:
                break
    f.close()

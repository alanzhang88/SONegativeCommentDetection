from xml.etree.ElementTree import XMLPullParser
from MongodbClient import get_collection

collection = get_collection('PostsWithNoAnswer')
list_of_keys = ['Id','Score','ViewCount']

PostsFilePath = './DataSample/Posts.xml'

parser = XMLPullParser(events=['end'])
with open(file=PostsFilePath) as f:
    for line in f:
        parser.feed(line)
        for event,elem in parser.read_events():
            if(elem.tag == 'row'):
                postTypeId = int(elem.get('PostTypeId'))
                if postTypeId == 1:
                    answerCount = int(elem.get('AnswerCount'))
                    if(answerCount == 0):
                        data_to_save = {key: elem.get(key) for key in list_of_keys}
                        collection.insert_one(data_to_save)
    f.close()

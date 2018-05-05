from xml.etree.ElementTree import XMLPullParser
from MongodbClient import get_collection

collection = get_collection('PostsWithLowScore')
list_of_keys = ['Id','Score','ViewCount', 'Body']

PostsFilePath = './DataSample/Posts.xml'

scoreThreshold = -2
parser = XMLPullParser(events=['end'])
with open(file=PostsFilePath) as f:
    for line in f:
        parser.feed(line)
        for event,elem in parser.read_events():
            if(elem.tag == 'row'):
                score = int(elem.get('Score'))
                if score <= scoreThreshold:
                    postTypeId = int(elem.get('PostTypeId'))
                    data_to_save = {key: elem.get(key) for key in list_of_keys}
                    if postTypeId == 2:
                        data_to_save['ParentId'] = elem.get('ParentId')
                    collection.insert_one(data_to_save)
    f.close()

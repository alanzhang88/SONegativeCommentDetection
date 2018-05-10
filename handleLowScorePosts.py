from xml.etree.ElementTree import XMLPullParser
from MongodbClient import MyMongoClient
import sys

# python3 handleLowScorePosts.py startId dbThreshold

client = MyMongoClient()
collection = client.get_collection('PostFirstIter')
# list_of_keys = ['Id','Score','ViewCount', 'Body','CommentCount']
list_of_str_keys = ['Id','Body']

PostsFilePath = './Data/Posts.xml'
startId = int(sys.argv[1]) if len(sys.argv) > 1 else 0
dbThreshold = int(sys.argv[2]) if len(sys.argv) > 2 else None
nextSwitchId = startId + dbThreshold if dbThreshold is not None else None
startLine = int(sys.argv[3]) if len(sys.argv) > 3 else None

scoreThreshold = -1
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
        if startLine is None:
            parser.feed(line)
        else:
            if counter <= 2 or counter >= startLine:
                parser.feed(line)
            else:
                continue
        for event,elem in parser.read_events():
            if(elem.tag == 'row'):
                Id = int(elem.get('Id'))
                if Id < startId:
                    continue
                score = int(elem.get('Score'))
                if score <= scoreThreshold:
                    postTypeId = int(elem.get('PostTypeId'))
                    data_to_save = {key: elem.get(key) for key in list_of_str_keys}
                    data_to_save['Id'] = Id
                    data_to_save['OwnerId'] = int(elem.get('OwnerUserId'))
                    data_to_save['CommentCount'] = int(elem.get('CommentCount'))
                    data_to_save['Score'] = int(elem.get('Score'))
                    if elem.get('ViewCount') is not None:
                        data_to_save['ViewCount'] = int(elem.get('ViewCount'))
                    if postTypeId == 2:
                        data_to_save['ParentId'] = elem.get('ParentId')
                    print('Inserting postid %d' % Id)
                    collection.insert_one(data_to_save)
        if nextSwitchId is not None and Id >= nextSwitchId:
            nextSwitchId += dbThreshold
            ret = client.switch_db()
            if ret:
                collection = client.get_collection('PostFirstIter')
            else:
                break

    f.close()

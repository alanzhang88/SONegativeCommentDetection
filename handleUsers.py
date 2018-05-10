from xml.etree.ElementTree import XMLPullParser
from MongodbClient import MyMongoClient
import sys

client = MyMongoClient()
collection = client.get_collection('UsersLowRep')
UsersFilePath = './Data/Users.xml'
startId = int(sys.argv[1]) if len(sys.argv) > 1 else 0
dbThreshold = int(sys.argv[2]) if len(sys.argv) > 2 else None
nextSwitchId = startId + dbThreshold if dbThreshold is not None else None
reputationThreshold = 100
viewThreshold = 100

parser = XMLPullParser(events=['end'])
with open(file=UsersFilePath) as f:
    Id = 0
    counter = 0
    rep = 0
    for line in f:
        parser.feed(line)
        for event,elem in parser.read_events():
            if elem.tag == 'row':
                Id = int(elem.get('Id'))
                if Id < startId:
                    continue
                # rep += int(elem.get('Reputation'))
                # counter += 1
                reputation = int(elem.get('Reputation'))
                if elem.get('Views') is not None:
                    viewCount = int(elem.get('Views'))
                if elem.get('UpVotes') is not None:
                    upCount = int(elem.get('UpVotes'))
                if elem.get('DownVotes') is not None:
                    downCount = int(elem.get('DownVotes'))
                if reputation <= reputationThreshold and viewCount <= viewThreshold:
                    data_to_save = {}
                    data_to_save['Id'] = elem.get('Id')
                    data_to_save['AccountId'] = elem.get('AccountId')
                    data_to_save['Reputation'] = reputation
                    data_to_save['Views'] = viewCount
                    data_to_save['UpVotes'] = upCount
                    data_to_save['DownVotes'] = downCount
                    print(data_to_save)
                    collection.insert_one(data_to_save)
                    # counter +=1
                    # print(counter)
            # print(rep/counter)
        if nextSwitchId is not None and Id >= nextSwitchId:
            nextSwitchId += dbThreshold
            ret = client.switch_db()
            if ret:
                collection = client.get_collection('UsersLowRep')
            else:
                break
    f.close()

from xml.etree.ElementTree import XMLPullParser
from MongodbClient import MyMongoClient
import sys

client = MyMongoClient()
collection = client.get_collection('PostsWithOffensiveVote')

VotesFilePath = './Data/Votes.xml'

parser = XMLPullParser(events=['end'])
with open(file=VotesFilePath) as f:
    counter = 0
    for line in f:
        counter += 1
        if counter % 1000000 == 0:
            parser.feed('</votes>')
            parser.close()
            parser = XMLPullParser(events=['end'])
            parser.feed('<votes>')

        parser.feed(line)
        for event,elem in parser.read_events():
            if(elem.tag == 'row'):
                voteType = int(elem.get('VoteTypeId'))
                if voteType == 4:
                    Id = int(elem.get('PostId'))
                    print('Inserting postid %d' % Id)
                    collection.insert_one({'Id':Id,'VoteTypeId': voteType})
    f.close()

from pymongo import MongoClient

MONGODB_URI = 'mongodb://alanzhang88:1234@ds014658.mlab.com:14658/cs230'
DB_NAME = "cs230"

client = MongoClient(MONGODB_URI)
db = client[DB_NAME]

def get_collection(collection_name):
    return db[collection_name]

from pymongo import MongoClient

MONGODB_URI = ['mongodb://alanzhang88:1234@ds014658.mlab.com:14658/cs230']
DB_NAME = ['cs230']
# MONGODB_URI = ['mongodb://alanzhang88:1234@ds014648.mlab.com:14648/cs230db1']
# DB_NAME = ['cs230db1']
# MONGODB_URI = ['mongodb://alanzhang88:1234@ds014658.mlab.com:14658/cs230','mongodb://alanzhang88:1234@ds014648.mlab.com:14648/cs230db1']
# DB_NAME = ['cs230','cs230db1']
# index = 0
#
# client = MongoClient(MONGODB_URI[index])
# db = client[DB_NAME[index]]
#
# def get_collection(collection_name):
#     return db[collection_name]
#
# def switch_db():
#     index += 1
#     if index >= len(MONGODB_URI):
#         return False
#     else:
#         client = MongoClient(MONGODB_URI[index])
#         db = client[DB_NAME[index]]
#         return True
#
# def set_db(i):
#     if i < len(DB_NAME):
#         client = MongoClient(MONGODB_URI[i])
#         db = client[DB_NAME[i]]
#         return True
#     else:
#         return False


class MyMongoClient:
    def __init__(self,MONGODB_URI=MONGODB_URI,DB_NAME=DB_NAME):
        self.MONGODB_URI=MONGODB_URI
        self.DB_NAME = DB_NAME
        self.index = 0
        self.client = MongoClient(self.MONGODB_URI[self.index])
        self.db = self.client[self.DB_NAME[self.index]]

    def get_collection(self,collection_name):
        return self.db[collection_name]

    def switch_db(self):
        self.index += 1
        if self.index >= len(self.MONGODB_URI):
            print('No More DB to switch')
            return False
        else:
            self.client.close()
            self.client = MongoClient(self.MONGODB_URI[self.index])
            self.db = self.client[self.DB_NAME[self.index]]
            return True

    def set_db(self,i):
        if i < len(self.MONGODB_URI):
            self.client.close()
            self.client = MongoClient(self.MONGODB_URI[i])
            self.db = self.client[self.DB_NAME[i]]
            return True
        else:
            return False

from unicodedata import name
import pymongo

def connect_db():
    client = pymongo.MongoClient('mongodb+srv://admin:admin@chatbot.2qttl.mongodb.net/admin?authSource=admin&replicaSet=atlas-116lsw-shard-0&readPreference=primary&appname=MongoDB%20Compass&ssl=true')
    db = client['chatbot']
    return db

def get_collection(input):
    db= connect_db()
    collection = db[input]
    return collection

if __name__ == '__main__':
    connect_db()
    get_collection()

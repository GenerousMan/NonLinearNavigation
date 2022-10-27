import pymongo

client = pymongo.MongoClient("mongodb://localhost:27017/")

def init_(taskId):
    db = client["aliwood"]
    col = db["preprocess-progress"]
    doc = col.find_one({'taskId':taskId})
    if doc is None:
        new_progress = {'taskId':taskId, 'trans': 0, 'yolo':0, 'alphapose':0, 'features': 0, 'match': 0, 'render': 0}
        col.insert_one(new_progress)
    else:
        # taskId已经存在
        query = {'taskId':taskId}
        new_progress = { "$set": {'taskId':taskId, 'trans': 0, 'yolo':0, 'alphapose':0, 'features': 0, 'match': 0, 'render': 0}}
        col.update_one(query, new_progress)

def set_(taskId, item, now, lent):
    db = client["aliwood"]
    col = db["preprocess-progress"]
    query = { 'taskId': taskId }
    percent = int(now / lent * 100)
    new_value = { "$set": { item: percent }}
    col.update_one(query, new_value)
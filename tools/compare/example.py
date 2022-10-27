import pymongo

client = pymongo.MongoClient("mongodb://localhost:27017/")

def init_(example):
    db = client["aliwood"]
    col = db["example"]
    doc = col.find_one({'name': example['name']})
    if doc is None:
        col.insert_one(example)
    else:
        # taskId已经存在
        query = {'name': example['name']}
        new_example = { "$set": example }
        col.update_one(query, new_example)
    print(example)

def save_results(taskId, results):
    db = client["aliwood"]
    col = db["results"]
    doc = col.find_one({'taskId': taskId})
    if doc is None:
        col.insert_one({'taskId': taskId, 'results':results})
    else:
        # taskId已经存在
        query = { 'taskId': taskId }
        new_results = { "$set": {'taskId': taskId, 'results':results} }
        col.update_one(query, new_results)
import os
import pickle
import pymongo

client = pymongo.MongoClient("mongodb://localhost:27017/")

def save(taskId, shot_lines):
    match_result = []
    for shot_line in shot_lines:
        match_result.append(shot_line[0].get_match_data())
    # with pymongo.MongoClient("mongodb://localhost:27017/") as client:
    db = client["aliwood"]
    col = db["match-result"]
    doc = col.find_one({'taskId':taskId})
    if doc is None:
        data = {'taskId':taskId, 'match_result':match_result}
        col.insert_one(data)
    else:
        # taskId已经存在
        data = {'taskId':taskId, 'match_result':match_result}
        query = {'taskId':taskId}
        new_data = { "$set": data}
        col.update_one(query, new_data)
    # 顺便再保存一份数据，以备后续使用
    save_folder = '../match-results/'
    save_path = os.path.join(save_folder, '{}.data'.format(taskId))
    with open(save_path, 'wb') as f:
        pickle.dump(shot_lines, f)

def load(taskId):
    save_folder = '../match-results/'
    save_path = os.path.join(save_folder, '{}.data'.format(taskId))
    with open(save_path, 'rb') as f:
        shot_lines = pickle.load(f)
    return shot_lines
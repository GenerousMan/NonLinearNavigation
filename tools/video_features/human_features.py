def calc_view_of_videos(videos):
    from model.wood_view.view_clf import load_model
    clf = load_model()
    datas = []
    for video in videos:
        for frame in video.frames:
            data = [video.width, video.height] + frame.comb_data
            datas.append(data)
    results = clf.predict(datas)
    count = 0
    for video in videos:
        for frame in video.frames:
            frame.view = results[count]
            count += 1

def calc_direction_of_videos(videos):
    from model.wood_direction.direction_clf import load_model
    clf = load_model()
    datas = []
    for video in videos:
        for frame in video.frames:
            data = [video.width, video.height] + frame.comb_data
            datas.append(data)
    results = clf.predict(datas)
    count = 0
    for video in videos:
        for frame in video.frames:
            frame.direction = results[count]
            count += 1

def calc_pose_of_videos(videos, data_length):
    from model.wood_pose.pose_rf import load_model
    rfc = load_model()
    datas = []
    for video in videos:
        if len(video.frames) >= data_length:
            # units = []
            for i in range(len(video.frames)):
                first = i
                last = i + data_length
                if (last - 1) >= len(video.frames):
                    break
                data = []
                for j in range(first, last):
                    frame = video.frames[j]
                    data += ([video.width, video.height] + frame.comb_data)
                datas.append(data)
    results = rfc.predict(datas)
    count = 0
    for video in videos:
        if len(video.frames) >= data_length:
            for i in range(len(video.frames)):
                frame = video.frames[i]
                last = i + data_length
                if (last - 1) >= len(video.frames):
                    frame.pose = results[count-1]
                else:
                    frame.pose = results[count]
                    count += 1
            i = 0
            while i < len(video.frames):
                frame = video.frames[i]
                if frame.pose == 2 or frame.pose == 3:
                    next_i = i
                    for j in range(1, 6):
                        if i + j >= len(video.frames):
                            next_i = i + j
                            break
                        elif video.frames[i + j].pose == 0 or video.frames[j].pose == 4:
                            video.frames[i + j].pose = frame.pose
                            next_i = i + j + 1
                        else:
                            next_i = i + j
                            break
                    i = next_i
                else:
                    i += 1
        else:
            for frame in video.frames:
                frame.pose = 4

def calc_motion_of_videos(videos, data_length):
    from model.wood_motion.motion_rf import load_model
    rfc = load_model()
    datas = []
    for video in videos:
        if len(video.frames) >= data_length:
            # units = []
            for i in range(len(video.frames)):
                first = i
                last = i + data_length
                if (last - 1) >= len(video.frames):
                    break
                data = []
                for j in range(first, last):
                    frame = video.frames[j]
                    data += ([video.width, video.height] + frame.comb_data)
                datas.append(data)
    results = rfc.predict(datas)
    count = 0
    for video in videos:
        if len(video.frames) >= data_length:
            for i in range(len(video.frames)):
                frame = video.frames[i]
                last = i + data_length
                if (last - 1) >= len(video.frames):
                    frame.motion = results[count-1]
                else:
                    frame.motion = results[count]
                    count += 1
        else:
            for frame in video.frames:
                frame.motion = 3

feature_dict = {
    'view': 0,
    'direction': 1,
    'pose': 2,
    'motion': 3,
    'none': 4
}

def calc_order_of_tshot(tshot):
    from model.wood_order.order_rf import load_model
    rfr = load_model()
    view = tshot.view_mark
    direction = tshot.direction_mark
    pose = tshot.pose_mark
    motion = tshot.motion_mark
    data = [view, direction, pose, motion]
    view_weight = rfr.predict([data + [feature_dict['view']]])[0]
    direction_weight = rfr.predict([data + [feature_dict['direction']]])[0]
    pose_weight = rfr.predict([data + [feature_dict['pose']]])[0]
    motion_weight = rfr.predict([data + [feature_dict['motion']]])[0]
    weight_sum = view_weight + direction_weight + pose_weight + motion_weight
    tshot.view_weight = view_weight / weight_sum
    tshot.direction_weight = direction_weight / weight_sum
    tshot.pose_weight = pose_weight / weight_sum
    tshot.motion_weight = motion_weight / weight_sum
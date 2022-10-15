from classes.video import Video

def calc_features(videos, taskId=None):
    from tools.video_features.human_features import calc_view_of_videos
    calc_view_of_videos(videos)
    from tools.video_features.human_features import calc_direction_of_videos
    calc_direction_of_videos(videos)
    from tools.video_features.human_features import calc_pose_of_videos
    calc_pose_of_videos(videos, 6)
    from tools.video_features.human_features import calc_motion_of_videos
    calc_motion_of_videos(videos, 4)

def combine_roi_pose(videos):
    from tools.video_features.alphapose import get_pose_roi_data
    for video in videos:
        for frame in video.frames:
            pose = frame.alphapose
            roi = frame.roi
            frame.comb_data = get_pose_roi_data(pose, roi)
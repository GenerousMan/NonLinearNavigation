import os
import cv2
import json
import uuid
import pickle
import random
import skvideo.io
import numpy as np
import multiprocessing
import moviepy.editor as mpe
from classes.editor_shot import EditorShot

def get_random_id(length):
    alphabet = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    return ''.join(random.sample(alphabet, length))

class EditorProject(object):

    def __init__(self, taskId, projectId, template, shot_line, shot_groups):
        self.taskId = taskId
        self.projectId = projectId
        self.template = template
        self.editor_shot_line = []
        self.candidate_shots = []
        self.history = []
        for vshot, sgs in zip(shot_line.vshots, shot_groups):
            default_es = EditorShot(vshot)
            self.editor_shot_line.append(default_es)
            candidate_shots = [default_es]
            temp_shots = self.get_candidate_shots(sgs)
            for gvs in temp_shots[:10]:
                if gvs == vshot:
                    continue
                candidate_shots.append(EditorShot(gvs))
            self.candidate_shots.append(candidate_shots)

    def get_candidate_shots(self, sgs):
        temp_shots = []
        for vshot in sgs.vshots:
            for ts in temp_shots:
                if vshot.cross_with(ts):
                    break
            else:
                temp_shots.append(vshot)
        return temp_shots

    def generate_temp_files(self):
        # 生成预览图和视频素材
        # pool = multiprocessing.Pool(processes = 4)
        # for cad_shots in self.candidate_shots:
        #     for cad_shot in cad_shots:
        #         cad_shot.generate_temp_files()
        #         pool.apply_async(cad_shot.generate_temp_files)
        # pool.close()
        # pool.join()
        video_dict = {}
        # 初始化视频资源链
        for eshots in self.candidate_shots:
            for eshot in eshots:
                eshot.set_video_writer()
                path = eshot.video.path
                if path not in video_dict:
                    video_dict[path] = []
                video_dict[path].append(eshot)

        # 依次处理视频链
        for path in video_dict.keys():
            # 对视频链排序
            video_dict[path] = sorted(video_dict[path])
            videogen = skvideo.io.vreader(path)
            for fn, frame in enumerate(videogen):
                if len(video_dict[path]) == 0:
                    break
                # 查看最前面的eshot.start
                first = video_dict[path][0]
                if fn < first.start:
                    continue
                # 收集已经结束写帧的eshot
                temp = []
                for eshot in video_dict[path]:
                    if fn == eshot.mid:
                        eshot.save_thumbnail(frame)
                    if eshot.start <= fn < eshot.end:
                        eshot.writer.writeFrame(frame)
                    if fn == eshot.end - 1:
                        eshot.release_video_writer()
                        temp.append(eshot)
                # 移除已经结束写帧的eshot
                for eshot in temp:
                    video_dict[path].remove(eshot)
            # 进行最后阶段的清空列表
            for eshot in video_dict[path]:
                eshot.release_video_writer()

    def get_project_info(self):
        shot_line = [
            {'id':es.id, 'thumbnail':es.thumbnail, 'video':es.video_url} for es in self.editor_shot_line
        ]
        candidates = [
            [ {'id':cad_shot.id, 'thumbnail':cad_shot.thumbnail, 'video':cad_shot.video_url} for cad_shot in cad_shots] for cad_shots in self.candidate_shots
        ]
        return {
            'taskId':self.taskId,
            'projectId':self.projectId,
            'shotLine':shot_line,
            'candidates':candidates
        }

    def update_project_info(self, projectInfo):
        shotLine = projectInfo["shotLine"]
        for index, shot in enumerate(shotLine):
            for cad_shot in self.candidate_shots[index]:
                if str(cad_shot.id) == shot["id"]:
                    self.editor_shot_line[index] = cad_shot
                    break

    def get_project_preview(self):
        video_folder = '../api-center/public/tempoFiles/videos/'
        video_name = str(uuid.uuid4()) + '.mp4'
        video_path = os.path.join(video_folder, video_name)
        temp_video_name = str(uuid.uuid4()) + '.mp4'
        temp_video_path = os.path.join(video_folder, temp_video_name)
        writer = self.editor_shot_line[0].get_video_writer(temp_video_path)
        for index, es in enumerate(self.editor_shot_line):
            read_video_name = es.video_name
            read_video_path = os.path.join(video_folder, read_video_name)
            videogen = skvideo.io.vreader(read_video_path)
            for frame in videogen:
                frame = self.add_index_mark(index + 1, frame)
                writer.writeFrame(frame)
        writer.close()
        music_path = '../templates/{}/music.mp3'.format(self.template)
        my_clip = mpe.VideoFileClip(temp_video_path)
        audio_background = mpe.AudioFileClip(music_path)
        final_clip = my_clip.set_audio(audio_background)
        final_clip.write_videofile(video_path)
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        video = '/tempoFiles/videos/' + video_name
        return video

    def add_index_mark(self, index, frame):
        height, width, channel = frame.shape
        point = (width//2, height//2)
        point_size = int(0.05 * min(height, width))
        point_color = (64, 158, 255)
        thickness = -1
        frame = cv2.circle(frame, point, point_size, point_color, thickness)
        font = cv2.FONT_HERSHEY_SIMPLEX
        point = (width//2 - 3, height//2)
        frame = cv2.putText(frame, str(index), point, font, 3, (255, 255, 255), 6)
        return frame

    def get_project_result(self):
        video_folder = '../api-center/public/tempoFiles/videos/'
        resultId = get_random_id(8)
        video_name = resultId + '.mp4'
        video_path = os.path.join(video_folder, video_name)
        temp_video_name = str(uuid.uuid4()) + '.mp4'
        temp_video_path = os.path.join(video_folder, temp_video_name)
        writer = self.editor_shot_line[0].get_video_writer(temp_video_path)
        for index, es in enumerate(self.editor_shot_line):
            read_video_name = es.video_name
            read_video_path = os.path.join(video_folder, read_video_name)
            videogen = skvideo.io.vreader(read_video_path)
            for frame in videogen:
                writer.writeFrame(frame)
        writer.close()
        music_path = '../templates/{}/music.mp3'.format(self.template)
        my_clip = mpe.VideoFileClip(temp_video_path)
        audio_background = mpe.AudioFileClip(music_path)
        final_clip = my_clip.set_audio(audio_background)
        final_clip.write_videofile(video_path)
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        return resultId

    @staticmethod
    def save_project(editor_project, taskId, projectId):
        save_folder = 'temp/project/'
        save_name = '{}-{}.data'.format(taskId, projectId)
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        with open(os.path.join(save_folder, save_name), "wb") as f:
            pickle.dump(editor_project, f)

    @staticmethod
    def load_project(taskId, projectId):
        save_folder = 'temp/project/'
        save_name = '{}-{}.data'.format(taskId, projectId)
        with open(os.path.join(save_folder, save_name), "rb") as f:
            editor_project = pickle.load(f)
            return editor_project
        return None

import sys
import time
from tqdm import tqdm

from libs.AlphaPose.dataloader import ImageLoader, VideoLoader, CustomLoader, DetectionLoader, DetectionProcessor, DataWriter, Mscoco
from libs.AlphaPose.SPPE.src.main_fast_inference import *
from libs.AlphaPose.fn import getTime


def get_pose(humans):
    if len(humans) > 1:
        human_scores = []
        for human in humans:
            max_x, max_y = 0, 0
            min_x, min_y = 99999999, 99999999
            for i in range(17):
                x, y = human['keypoints'][i]
                max_x = x if x > max_x else max_x
                max_y = y if y > max_y else max_y
                min_x = x if x < min_x else min_x
                min_y = y if y < min_y else min_y
            temp_area = (max_x - min_x) * (max_y - min_y)
            score = human['proposal_score'][0].item()
            human_scores.append(temp_area * score)
        result = humans[human_scores.index(max(human_scores))]
    elif len(humans) == 1:
        result = humans[0]
    else:
        keypoints = torch.Tensor([[0, 0] for i in range(17)])
        kp_score = torch.Tensor([[0] for i in range(17)])
        proposal_score = torch.Tensor([0])
        result = {
            'keypoints': keypoints,
            'kp_score': kp_score,
            'proposal_score': proposal_score
        }
    return [result]

def calc_pose_of_videos_v2(videos, taskId=None):
    torch.cuda.empty_cache()
    start = time.time()
    total_frame_count = sum([video.frame_count for video in videos])

    # Load Videos
    print('Loading Videos..')
    data_loader = CustomLoader(videos, batchSize=1).start()

    # Load detection loader
    print('Loading YOLO model..')
    sys.stdout.flush()
    det_loader = DetectionLoader(data_loader, batchSize=1).start()
    det_processor = DetectionProcessor(det_loader).start()

    # Load pose model
    pose_dataset = Mscoco()
    with torch.no_grad():
        pose_model = InferenNet_fast(4 * 1 + 1, pose_dataset)
        pose_model.cuda()
        pose_model.eval()

    runtime_profile = {
        'dt': [],
        'pt': [],
        'pn': []
    }

    # Init data writer
    writer = DataWriter(False).start()

    data_len = data_loader.length()
    im_names_desc = tqdm(range(data_len))
    batchSize = 10

    for i in im_names_desc:
        start_time = getTime()
        with torch.no_grad():
            (inps, orig_img, im_name, boxes, scores, pt1, pt2) = det_processor.read()
            if boxes is None or boxes.nelement() == 0:
                writer.save(None, None, None, None, None, orig_img, im_name.split('/')[-1])
                continue

            ckpt_time, det_time = getTime(start_time)
            runtime_profile['dt'].append(det_time)
            # Pose Estimation
            
            datalen = inps.size(0)
            leftover = 0
            if (datalen) % batchSize:
                leftover = 1
            num_batches = datalen // batchSize + leftover
            hm = []
            for j in range(num_batches):
                inps_j = inps[j*batchSize:min((j + 1) * batchSize, datalen)].cuda()
                hm_j = pose_model(inps_j)
                hm.append(hm_j)
            hm = torch.cat(hm)
            ckpt_time, pose_time = getTime(ckpt_time)
            runtime_profile['pt'].append(pose_time)
            hm = hm.cpu()
            writer.save(boxes, scores, hm, pt1, pt2, orig_img, im_name.split('/')[-1])

            ckpt_time, post_time = getTime(ckpt_time)
            runtime_profile['pn'].append(post_time)

        im_names_desc.set_description(
            'det time: {dt:.3f} | pose time: {pt:.2f} | post processing: {pn:.4f}'.format(
                dt=np.mean(runtime_profile['dt']), pt=np.mean(runtime_profile['pt']), pn=np.mean(runtime_profile['pn']))
            )

    print('===========================> Finish Model Running.')
    while(writer.running()):
        pass
    writer.stop()
    results = writer.results()

    result_index = 0
    for video in videos:
        video.has_pose = True
        for frame in video.frames:
            frame.alphapose = get_pose(results[result_index]['result'])[0]
            result_index += 1

    end = time.time()
    cost = end - start
    print('Pose time cost:', cost)
    print('Frame per second:', total_frame_count / cost)

def get_pose_roi_data(pose, roi):
    data = []
    key_points = pose['keypoints']
    pro_points = pose['kp_score']
    for j in range(len(key_points)):
        # 保存和yolo框左上角相关的各个动作点坐标
        data.append(key_points[j][0].item() - roi[1])
        data.append(key_points[j][1].item() - roi[0])
        # 保存各个关节点的置信度
        data.append(pro_points[j][0].item())
    for j in range(4):
        # 保存yolo框
        data.append(roi[j])
    return data
# deepsortで得られた結果をkapaoと一緒にみてみる

import sys 
from pathlib import Path

FILE = Path().resolve()
sys.path.append(str(FILE))
import os

from pdb import set_trace as pst
import argparse
from pytube import YouTube
import os.path as osp
from utils.torch_utils import select_device, time_sync
from utils.general import check_img_size
from utils.datasets import LoadImages
from models.experimental import attempt_load
import torch
import pickle
import cv2
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

# from tqdm.notebook import tqdm
import imageio
from val import run_nms, post_process_batch


GRAY = (200, 200, 200)
CROWD_THRES = 200  # max bbox size for crowd classification
CROWD_ALPHA = 0.5
CROWD_KP_SIZE = 2
CROWD_KP_THICK = 2
CROWD_SEG_THICK = 2

BLUE = (245, 140, 66)
ORANGE = (66, 140, 245)
RED = (255, 0, 0)
PLAYER_ALPHA_BOX = 0.85
PLAYER_ALPHA_POSE = 0.3
PLAYER_KP_SIZE = 4
PLAYER_KP_THICK = 4
PLAYER_SEG_THICK = 4
FPS_TEXT_SIZE = 3

COLOR = (255, 0, 255)  # purple
ALPHA = 0.5
SEG_THICK = 3
FPS_TEXT_SIZE = 2


def get_args():
    """引数パラメータを獲得する"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/coco-kp.yaml")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--weights", default="kapao_s_coco.pt")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or cpu")
    parser.add_argument("--half", action="store_true")
    parser.add_argument(
        "--conf-thres", type=float, default=0.5, help="confidence threshold"
    )
    parser.add_argument(
        "--iou-thres", type=float, default=0.45, help="NMS IoU threshold"
    )
    parser.add_argument(
        "--no-kp-dets", action="store_true", help="do not use keypoint objects"
    )
    parser.add_argument("--conf-thres-kp", type=float, default=0.5)
    parser.add_argument("--conf-thres-kp-person", type=float, default=0.2)
    parser.add_argument("--iou-thres-kp", type=float, default=0.45)
    parser.add_argument("--overwrite-tol", type=int, default=50)
    parser.add_argument("--scales", type=float, nargs="+", default=[1])
    parser.add_argument("--flips", type=int, nargs="+", default=[-1])
    parser.add_argument(
        "--display", action="store_true", help="display inference results"
    )
    parser.add_argument("--fps", action="store_true", help="display fps")
    parser.add_argument("--gif", action="store_true", help="create fig")
    parser.add_argument("--start", type=int, default=20, help="start time (s)")
    parser.add_argument("--end", type=int, default=80, help="end time (s)")
    parser.add_argument("--sorted_mot", type=str)

    parser.add_argument("--video_name", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--poses", type=str)

    args = parser.parse_args()
    return args


def convert_tldr2ltwh(bboxes, scores) -> list:
    """convert topleft downright to left top width hight
    Args:
        bboxes (list): shape = [frames,nums,[x1 y1 x2 y2]]

    Returns:
        list: [<frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>]
    """
    ret = []
    for i, (bbox, score) in enumerate(zip(bboxes, scores), start=1):
        for (x1, y1, x2, y2), (conf) in zip(bbox, score):
            bb_left = x1
            bb_top = y1
            bb_width = abs(x2 - x1)
            bb_height = abs(y1 - y2)

            _box = [i, -1, bb_left, bb_top, bb_width, bb_height, conf, -1, -1, -1]

            ret.append(_box)
    return ret


def convert_ltwh2tldr(bboxes) -> list:
    """convert left top width height to topleft downright
    Args:
        bboxes (list):
            list: [<frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>]

    Returns:
        list: [<frame>, <id>, <x1>, <y1>, <x2>, <y2>, <conf>, <x>, <y>, <z>]
    """
    ret = []
    for row in bboxes:
        x2 = row[2] + row[4]
        y2 = row[3] + row[5]
        _row = [row[0], row[1], row[2], row[3], x2, y2, row[6], row[7], row[8], row[9]]
        assert len(row) == len(_row)
        del row
        ret.append(_row)
    return ret


def convert_poses_MOTformat(poses) -> list:
    """convert format from kapao to MOT-like
    Args:
        poses: [frame,person,poses]

    Returns:
        list: [<frame>, <id>, <poses>(np.ndarray)]
    """
    ret = []

    for i, frame_block in enumerate(poses, start=1):
        # 一フレームごと
        for j, _pose in enumerate(frame_block):
            ret.append([i, j, _pose])
    return ret


def get_masked_data(data: list, mask_id: int = -1):
    """get data whose mask id is the same."""
    ret = []
    for d in data:
        if d[0] == mask_id:
            ret.append(d)
        if d[0] > mask_id:
            break
    return ret


def get_max_frame(data: list) -> int:
    """get max frame number
    Args:
        data (list): MOT-like format

    Returns:
        int
    """
    return data[-1][0]

def get_sorted_bbox()->pd.DataFrame:
    """get sorted_mot(from deepsort)"""
    return pd.read_csv(args.sorted_mot,header=None)

def load_poses()->list:
    """load poses list created from kapao"""
    with open(args.poses,'rb') as f:
        ret = pickle.load(f)
    return ret

def load_ltwh_bboxes():
    pass






if __name__ == "__main__":
    args = get_args()
    with open(args.data) as f:
        data = yaml.safe_load(f)  # load data dict

    # add inference settings to data dict
    data["imgsz"] = args.imgsz
    data["conf_thres"] = args.conf_thres
    data["iou_thres"] = args.iou_thres
    data["use_kp_dets"] = not args.no_kp_dets
    data["conf_thres_kp"] = args.conf_thres_kp
    data["iou_thres_kp"] = args.iou_thres_kp
    data["conf_thres_kp_person"] = args.conf_thres_kp_person
    data["overwrite_tol"] = args.overwrite_tol
    data["scales"] = args.scales
    data["flips"] = [None if f == -1 else f for f in args.flips]
    VIDEO_NAME = args.video_name
    assert osp.isfile(VIDEO_NAME)

    device = select_device(args.device, batch_size=1)
    print("Using device: {}".format(device))

    # deepsort 
    sorted_mot = get_sorted_bbox()

    model = attempt_load(args.weights, map_location=device)  # load FP32 model
    half = args.half & (device.type != 'cpu')
    if half:  # half precision only supported on CUDA
        model.half()
    stride = int(model.stride.max())  # model stride

    imgsz = check_img_size(args.imgsz, s=stride)  # check image size
    # dataset = LoadImages('./{}'.format(VIDEO_NAME), img_size=imgsz, stride=stride, auto=True)
    dataset = LoadImages('{}'.format(VIDEO_NAME), img_size=imgsz, stride=stride, auto=True)

    cap = dataset.cap
    cap.set(cv2.CAP_PROP_POS_MSEC, args.start * 1000)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(fps * (args.end - args.start))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    gif_frames = []
    video_name = 'pingpong_inference_deepsort_{}_{}'.format(osp.splitext(args.weights)[0],VIDEO_NAME.split('/')[-1].split(".")[0])

    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    if not args.display:
        writer = cv2.VideoWriter(args.output_dir+video_name + '.mp4',cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        if not args.fps:  # tqdm might slows down inference
            progress_dataset = (dataset)

    ###########
    # 描画するところ #
    ###########

    tldr_sort_bboxes:list = convert_ltwh2tldr(sorted_mot.to_numpy())
    mot_poses = load_poses()
    # ltwh_bboxes = load_ltwh_bboxes()
    # tldr_bboxes = convert_ltwh2tldr(ltwh_bboxes)

    dataset = tqdm(dataset,total=n)
    t0 = time_sync()

    for frame_id , (path,img,im0,_) in enumerate(dataset,start=1):
        img = torch.from_numpy(img).to(device)
        img = img.half()
        img /= 255.0
        if len(img.shape) == 3:
            img = img[None]
        
        im0_copy = im0.copy()
        
        # 対象のフレームとして切り出した
        mask_poses = get_masked_data(mot_poses,frame_id)
        mask_bboxes = get_masked_data(tldr_bboxes,frame_id)
        mask_sort_bboxes = get_masked_data(tldr_sort_bboxes, frame_id)
        
        # kapaoの方
        for _poses, _bboxes in zip(mask_poses,mask_bboxes):
            x1,y1,x2,y2 = _bboxes[2:6]
            cv2.rectangle(im0_copy,(int(x1),int(y1)),(int(x2),int(y2)),GRAY, thickness=2)
        im0 = cv2.addWeighted(im0, CROWD_ALPHA, im0_copy, 1 - CROWD_ALPHA, gamma=0)

        
        im0_copy = im0.copy()
        for _sort_bbox in mask_sort_bboxes:
            x1,y1,x2,y2 = _sort_bbox[2:6]
            cv2.rectangle(im0_copy,(int(x1),int(y1)),(int(x2),int(y2)),RED, thickness=2)
        im0 = cv2.addWeighted(im0, CROWD_ALPHA, im0_copy, 1 - CROWD_ALPHA, gamma=0)
#     im0 = cv2.addWeighted(im0, PLAYER_ALPHA_BOX, im0_copy, 1 - PLAYER_ALPHA_BOX, gamma=0)
            
        
        
        if frame_id == 1:
            t = time_sync() - t0
        else:
            t = time_sync() - t1
        if args.gif:
            gif_frames.append(cv2.resize(im0, dsize=None, fx=0.25, fy=0.25)[:, :, [2, 1, 0]])
        elif not args.display:
            writer.write(im0)
        else:
            cv2.imshow('', cv2.resize(im0, dsize=None, fx=0.5, fy=0.5))
            cv2.waitKey(1)

        t1 = time_sync()
        if frame_id == n :
            break


        
    cv2.destroyAllWindows()
    cap.release()
    if not args.display:
        writer.release()
    if args.gif:
        print('Saving GIF...')
        with imageio.get_writer(video_name + '.gif', mode="I", fps=fps) as writer:
            for idx, frame in tqdm(enumerate(gif_frames)):
                writer.append_data(frame)


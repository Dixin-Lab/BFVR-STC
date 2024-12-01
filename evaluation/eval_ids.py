from model import Backbone
import torch
from PIL import Image
from mtcnn import MTCNN
import cv2
import numpy as np
import torch
from tqdm import tqdm
import os
import os.path as osp 
import time 
from torchvision.transforms import Compose, ToTensor, Normalize
import tqdm 

device = 'cuda'
mtcnn = MTCNN()
model = Backbone(num_layers=50, drop_ratio=0.6, mode='ir_se')
model.load_state_dict(torch.load('weights/model_ir_se50.pth'))
model.eval()
model.to(device)


def read_frames(video_path): 
    cap = cv2.VideoCapture(video_path)
    # Extract frames from the video 
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized_frame = cv2.resize(frame,(512,512))  # Resize the frame to match the expected input size
        frames.append(resized_frame)
    return np.array(frames)


def preprocess(i, device):
    face = mtcnn.align(i)
    transfroms = Compose(
        [ToTensor(), Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    return transfroms(face).to(device).unsqueeze(0)


def f(i1, i2): 
    i1 = Image.fromarray(i1)
    i2 = Image.fromarray(i2)
    i1 = preprocess(i1, device)
    i2 = preprocess(i2, device)

    emb1 = model(i1)[0]
    emb2 = model(i2)[0]
    sim_12 = emb1.dot(emb2).item()
    return sim_12


if __name__ == '__main__': 
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_dir", type=str)
    parser.add_argument("--gt_dir", type=str)
    parser.add_argument("--method", type=str, default='')
    args = parser.parse_args()
    s_time = time.time()
    METHOD = args.method
    gen_dir = args.gen_dir
    gt_dir = args.gt_dir

    ids = []

    videos = os.listdir(gt_dir)
    for video in tqdm(videos): 
        gen_path = osp.join(gen_dir, video) 
        gt_path = osp.join(gt_dir, video) 
        gen_list = read_frames(gen_path) 
        gt_list = read_frames(gt_path) 
        ids_ = []
        for index in range(24): 
            ids_ = f(gen_list[index], gt_list[index])
        print(f'{video}:\t\t{np.mean(ids_)}')
        ids.append(np.mean(ids_))

    print(f'ids: {np.mean(ids)}') 
    print(f'cost {time.time()-s_time}s.')
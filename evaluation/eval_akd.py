import cv2
import mediapipe as mp
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import argparse 
import os 
import os.path as osp

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
    return frames


def dis(ld1, ld2): 
    return np.sqrt(
        (ld1.x-ld2.x)**2+(ld1.y-ld2.y)**2
    )


def cal_akd(list1, list2): 
    n = len(list1) 
    ret = []
    for idx in range(n): 
        img1 = list1[idx]
        img2 = list2[idx]
        ld1 = face_mesh.process(img1).multi_face_landmarks[0].landmark
        ld2 = face_mesh.process(img2).multi_face_landmarks[0].landmark
        nn = len(ld1)
        ret_ = []
        for idx2 in range(nn):
            ret_.append(dis(ld1[idx2], ld2[idx2]))
        ret.append(np.mean(ret_))
    return np.mean(ret)

# key points detection
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True,  
                                  max_num_faces=3,         
                                  refine_landmarks=True,  
                                  min_detection_confidence=0.5, 
                                  min_tracking_confidence=0.5) 

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_dir", type=str)
    parser.add_argument("--gt_dir", type=str)
    args = parser.parse_args()

    gen_dir = args.gen_dir
    gt_dir = args.gt_dir 

    videos = os.listdir(gt_dir)
    results = []
    for video in videos: 
        gen_path = osp.join(gen_dir, video) 
        gt_path = osp.join(gt_dir, video) 
        gen_list = read_frames(gen_path) 
        gt_list = read_frames(gt_path) 
        result = cal_akd(gen_list, gt_list) 
        results.append(result)

    print(f'avg akd: {np.mean(results)}')




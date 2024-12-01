# https://github.com/JunyaoHu/common_metrics_on_video_quality
import sys
sys.path.append("./common_metrics_on_video_quality-main/")
import torch
import numpy as np
from tqdm import tqdm
import os
import os.path as osp 
import cv2
from fvd.styleganv.fvd import get_fvd_feats, frechet_distance, load_i3d_pretrained
import time 
import argparse

device = "cuda"
i3d = load_i3d_pretrained(device=device)

SIZE = 512
NF = 24

def read_in_video_tensor(video_path): 
    # .mp4 -> [t, h, w, c]
    img_gt_list = []
    cnt = 0
    cap = cv2.VideoCapture(video_path) 
    while(cap.isOpened()): 
        ret_flag, image = cap.read()
        if ret_flag == True: 
            image = cv2.resize(image, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
            img_gt_list.append(image)
            cnt = cnt + 1 
            if cnt == NF: 
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else: 
            break
    cap.release()
    return img_gt_list


def trans(x):
    # if greyscale images add channel
    if x.shape[-3] == 1:
        x = x.repeat(1, 1, 3, 1, 1)
    # permute BTCHW -> BCTHW
    x = x.permute(0, 2, 1, 3, 4) 
    return x


def trans2(imgs): 
    # x: [t,h,w,c] -> [1,t,c,h,w] 
    imgs = np.array(imgs)
    imgs = torch.from_numpy(imgs)
    imgs = imgs / 255.0
    imgs = torch.permute(imgs, (0,3,1,2))
    imgs = imgs.unsqueeze(0)
    return imgs


def calculate_fvd2(videos1, videos2):
    video_len = min(videos1.shape[1], videos2.shape[1])
    videos1 = videos1[:,0:video_len,:,:,:]
    videos2 = videos2[:,0:video_len,:,:,:]
    assert videos1.shape == videos2.shape

    # support grayscale input, if grayscale -> channel*3
    # BTCHW -> BCTHW
    # videos -> [batch_size, channel, timestamps, h, w]

    videos1 = trans(videos1)
    videos2 = trans(videos2)

    # for calculate FVD, each clip_timestamp must >= 10
    vlen = videos1.shape[-3]+1
    
    # get a video clip
    # videos_clip [batch_size, channel, timestamps[:clip], h, w]
    videos_clip1 = videos1[:, :, : vlen]
    videos_clip2 = videos2[:, :, : vlen]

    # get FVD features
    feats1 = get_fvd_feats(videos_clip1, i3d=i3d, device=device)
    feats2 = get_fvd_feats(videos_clip2, i3d=i3d, device=device)
    
    # calculate FVD when timestamps[:clip]
    fvd_results = frechet_distance(feats1, feats2)

    return fvd_results


def calculate_fvd(video1, video2): 
    """
    Calculate FVD,

    Args: 
        video1: [t, h, w, c] 
        video2: [t, h, w, c] 
    Return:
        fvd value
    """
    video1 = trans2(video1) 
    video2 = trans2(video2)
    fvd_res = calculate_fvd2(video1, video2)
    return fvd_res


if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_dir", type=str)
    parser.add_argument("--gt_dir", type=str)
    parser.add_argument("--method", type=str, default='')
    args = parser.parse_args()
    s_time = time.time()
    METHOD = args.method
    gen_dir = args.gen_dir
    gt_dir = args.gt_dir

    fvd = []

    videos = os.listdir(gt_dir)
    for video in videos: 
        img_gen_list = read_in_video_tensor(osp.join(gen_dir, video))
        img_gt_list = read_in_video_tensor(osp.join(gt_dir, video))
        fvd_ = calculate_fvd(img_gen_list, img_gt_list)
        fvd.append(fvd_)

    print(f'fvd: {np.mean(fvd)}') 
    print(f'cost {time.time()-s_time}s.')


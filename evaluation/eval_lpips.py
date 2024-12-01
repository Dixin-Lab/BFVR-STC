# https://github.com/JunyaoHu/common_metrics_on_video_quality
import cv2
import numpy as np
import lpips
import torchvision.transforms as transforms
import torch
import time 
import os 
import os.path as osp 


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


def calculate_lpips(frame_set1, frame_set2, func): 
    """
    Calculate LPIPS metrics

    Args: 
        frame_set1: [t,h,w,c], c(rgb)
        frame_set2: [t,h,w,c], c(rgb)
    """

    def trans(x):
        x = 2 * x - 1
        return x.astype(np.float32)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    n_f = len(frame_set1) 
    set1 = [transform(trans(np.array(frame_set1[index])/255.)) for index in range(n_f)]
    set2 = [transform(trans(np.array(frame_set2[index])/255.)) for index in range(n_f)]
    set1 = torch.stack(set1).cuda()
    set2 = torch.stack(set2).cuda()

    return np.mean(
        func(set1, set2).detach().cpu().tolist()
    )
    result_list = [] 
    for index in range(n_f): 
        img0 = frame_set1[index]
        img1 = frame_set2[index] 
        img0 = transform(img0).unsqueeze(0)
        img1 = transform(img1).unsqueeze(0)
        lpips_ = func(img0, img1) 
        result_list.append(lpips_.item()) 
        del lpips_ 
        torch.cuda.empty_cache()
    
    return np.mean(result_list) 


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

    ret = []
    func = lpips.LPIPS(net="vgg", spatial=False).cuda().eval() 

    videos = os.listdir(gen_dir)#[:2] 
    for video in videos: 
        img_gen_list = read_in_video_tensor(osp.join(gen_dir, video))
        img_gt_list = read_in_video_tensor(osp.join(gt_dir, video))
        lpips_ = calculate_lpips(img_gen_list, img_gt_list, func)
        ret.append(lpips_)

    print(f'lpips: {np.mean(ret)}') 
    print(f'cost {time.time()-s_time}s.')

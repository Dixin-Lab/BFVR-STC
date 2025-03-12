import os, sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
import argparse
import glob
import numpy as np
import os
import cv2
import torch
from torchvision.transforms.functional import normalize
from basicsr.utils import imwrite, img2tensor, tensor2img, img2tensor_np
from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.utils.download_util import load_file_from_url
import ffmpeg
import random
from tqdm import tqdm


def read_in_video_tensor(video_path):
    # .mp4 -> [t, h, w, c] -> [1, c, t, h, w]
    img_gt_list = []
    img_in_list = []
    cnt = 0
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret_flag, image = cap.read()
        if ret_flag == True:
            image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LINEAR)
            img_gt_list.append(image)
            image = img2tensor_np(image, bgr2rgb=True, float32=True)
            img_in_list.append(image)
            cnt = cnt + 1
            if cnt == 24:
                break
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break
    cap.release()
    img_gt_np = np.array(img_gt_list)
    img_in_np = np.array(img_in_list)

    img_in_np = img_in_np.astype(np.float32) / 255.0
    img_in_np = img_in_np.transpose(3, 0, 1, 2)
    img_in = torch.from_numpy(img_in_np).float()

    return img_gt_list, img_in.unsqueeze(0)


def transform_tensor_to_imgs(x):
    # [1, c, t, h, w] -> [t, h, w, c]
    x = x.squeeze(0).permute(1, 2, 3, 0).float().detach().cpu()
    x = (x - x.min()) / (x.max() - x.min())
    x = list(x.numpy())
    frame_list = []
    for idx in range(len(x)):
        img = x[idx]
        assert len(img.shape) == 3
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = (img * 255.0).round()
        img = img.astype(np.uint8)
        frame_list.append(img)
    return frame_list


def write_np_to_video(frame_list, save_path):
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter(save_path, fourcc, 24.0, (512, 512))
    for idx in range(len(frame_list)):
        out.write(frame_list[idx])
    out.release()
    print(f"Video is saved in {save_path}.")


def f(test_path, device, model, save_root):
    video_name = os.path.basename(test_path)
    video_name = video_name.split(".")[0]
    print(video_name)

    # read in input_video, transform it to tensor format.
    img_gt_list, img_in = read_in_video_tensor(test_path)
    img_in = img_in.to(device)

    with torch.no_grad():
        output = model(img_in)[0]
        # transform tensor to np array,
        # [1, c, t, h, w] -> [t, h, w, c]
        img_gen_list = transform_tensor_to_imgs(output)
    del output
    torch.cuda.empty_cache()

    write_np_to_video(img_gen_list, os.path.join(save_root, f"{video_name}.avi"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", type=str, default="./test_input/3.mp4")
    parser.add_argument("-o", "--output_base", type=str, default="./test_output")
    args = parser.parse_args()

    if args.output_base.endswith("/"):  # solve when path ends with /
        args.output_base = args.output_base[:-1]
    dir_name = os.path.abspath(args.output_base)
    os.makedirs(dir_name, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_path = args.input_path

    ckpt_path = load_file_from_url(
        url="https://github.com/Dixin-Lab/BFVR-STC/releases/download/init/ckpt-deflickersd.pth",
        model_dir="weights/STC",
        progress=True,
        file_name=None,
    )

    model = ARCH_REGISTRY.get("CodeFormer3D")(n_head=4, n_layers=6).to(device)
    checkpoint = torch.load(ckpt_path)["params_ema"]

    model.load_state_dict(checkpoint)
    model.eval()

    f(test_path, device, model, args.output_base)

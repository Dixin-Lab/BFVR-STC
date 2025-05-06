from tqdm import tqdm
import cv2
import os
from paddleocr import PaddleOCR
from PIL import Image
import numpy as np
import random

input_dir = "processed_videos_2"
save_dir = "processed_videos_3"
files = os.listdir(input_dir)


ocr = PaddleOCR(
    use_angle_cls=True, lang="ch"
)  # need to run only once to download and load model into memory


def is_has_text(x):
    result = ocr.ocr(x, cls=True)
    result = result[0]
    try:
        scores = [line[1][1] for line in result]
        for item in scores:
            if item > 0.95:
                return True
    except:
        return False
    return False


for file in tqdm(files):
    vpath = os.path.join(input_dir, file)
    spath = os.path.join(save_dir, file)
    if os.path.exists(spath):
        continue

    video = cv2.VideoCapture(vpath)
    flg = False
    while True:
        ret, frame = video.read()

        if ret:
            frame = cv2.resize(frame, (512, 512), interpolation=cv2.INTER_LINEAR)
            flg = is_has_text(frame)
            break
        else:
            break

    video.release()
    cv2.destroyAllWindows()
    print(f"{vpath} has text: {flg}")
    if not flg:
        cmd = f"cp {vpath} {spath}"
        os.system(cmd)

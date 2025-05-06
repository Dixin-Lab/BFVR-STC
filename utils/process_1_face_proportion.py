import cv2
import os
import os.path as osp
from tqdm import tqdm

input_dir = "raw"
save_dir = "processed_videos_1"

files = os.listdir(input_dir)
# load face detection model
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

for file in tqdm(files):
    video_path = osp.join(input_dir, file)

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # initialize the boundary value
    min_x, min_y, max_x, max_y = width, height, 0, 0

    # find the max face boundary
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        for (x, y, w, h) in faces:
            expanded_w = int(w * 1.3)
            expanded_h = int(h * 1.3)
            expanded_x = x - (expanded_w - w) // 2
            expanded_y = y - (expanded_h - h) // 2

            min_x = min(min_x, expanded_x)
            min_y = min(min_y, expanded_y)
            max_x = max(max_x, expanded_x + expanded_w)
            max_y = max(max_y, expanded_y + expanded_h)

    cap.release()
    cap = cv2.VideoCapture(video_path)

    crop_width = max_x - min_x
    crop_height = max_y - min_y

    min_x = max(min_x, 0)
    min_y = max(min_y, 0)
    crop_width = min(crop_width, width - min_x)
    crop_height = min(crop_height, height - min_y)

    output_path = osp.join(save_dir, file)
    fourcc = cv2.VideoWriter_fourcc(*"mjpg")
    out = cv2.VideoWriter(output_path, fourcc, fps, (512, 512))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        cropped_frame = frame[min_y : min_y + crop_height, min_x : min_x + crop_width]
        cropped_frame = cv2.resize(cropped_frame, (512, 512))

        out.write(cropped_frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

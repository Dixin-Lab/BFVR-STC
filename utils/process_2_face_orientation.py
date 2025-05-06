import cv2
import os
import dlib
from imutils import face_utils
from basicsr.utils.download_util import load_file_from_url
from tqdm import tqdm

input_dir = "processed_videos_1"
save_dir = "processed_videos_2"
files = os.listdir(input_dir)

ckpt_path = load_file_from_url(
    url="https://github.com/Dixin-Lab/BFVR-STC/releases/download/init/shape_predictor_68_face_landmarks.dat",
    model_dir="weights",
    progress=True,
    file_name=None,
)

detector = dlib.get_frontal_face_detector()
# load face landmark model
predictor = dlib.shape_predictor(ckpt_path)


def is_side_face(gray):
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)

        left_eye_x = landmarks[36, 0]  # Outer corner of the left eye
        right_eye_x = landmarks[45, 0]  # Outer corner of the right eye
        nose_x = landmarks[30, 0]  # Tip of the nose
        lc = abs(nose_x - left_eye_x)
        rc = abs(nose_x - right_eye_x)
        ratio = lc / rc
        if ratio < 0.4 or ratio > 2.5:
            return True
        if nose_x < left_eye_x or nose_x > right_eye_x:
            return True

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
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flg = is_side_face(gray)
            if flg:
                break

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

    video.release()
    cv2.destroyAllWindows()
    print(f"{vpath} profile: {flg}")
    if not flg:
        cmd = f"cp {vpath} {spath}"
        os.system(cmd)

import cv2
import math
import random
import numpy as np
import os.path as osp
from scipy.io import loadmat
from PIL import Image
import torch
import torch.utils.data as data
from torchvision.transforms.functional import (
    adjust_brightness,
    adjust_contrast,
    adjust_hue,
    adjust_saturation,
    normalize,
)
from basicsr.data import gaussian_kernels as gaussian_kernels
from basicsr.data.transforms import augment
from basicsr.data.data_util import paths_from_folder, brush_stroke_mask, random_ff_mask
from basicsr.utils import (
    FileClient,
    get_root_logger,
    imfrombytes,
    img2tensor,
    img2tensor_np,
)
from basicsr.utils.registry import DATASET_REGISTRY

from basicsr.utils.video_util import VideoReader, VideoWriter
import time
import random
import ffmpeg
import io
import av
import math
from PIL import Image
from io import BytesIO


@DATASET_REGISTRY.register()
class bfvr_Dataset(data.Dataset):
    def __init__(self, opt):
        super(bfvr_Dataset, self).__init__()
        logger = get_root_logger()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt["io_backend"]

        self.gt_folder = opt["dataroot_gt"]
        self.gt_size = opt.get("gt_size", 512)
        self.in_size = opt.get("in_size", 512)
        assert self.gt_size >= self.in_size, "Wrong setting."

        self.mean = opt.get("mean", [0.5, 0.5, 0.5])
        self.std = opt.get("std", [0.5, 0.5, 0.5])

        self.component_path = opt.get("component_path", None)
        self.latent_gt_path = opt.get("latent_gt_path", None)

        if self.component_path is not None:
            self.crop_components = True
            self.components_dict = torch.load(self.component_path)
            self.eye_enlarge_ratio = opt.get("eye_enlarge_ratio", 1.4)
            self.nose_enlarge_ratio = opt.get("nose_enlarge_ratio", 1.1)
            self.mouth_enlarge_ratio = opt.get("mouth_enlarge_ratio", 1.3)
        else:
            self.crop_components = False

        if self.latent_gt_path is not None:
            self.load_latent_gt = True
            self.latent_gt_dict = torch.load(self.latent_gt_path)
        else:
            self.load_latent_gt = False

        if self.io_backend_opt["type"] == "lmdb":
            self.io_backend_opt["db_paths"] = self.gt_folder
            if not self.gt_folder.endswith(".lmdb"):
                raise ValueError(
                    "'dataroot_gt' should end with '.lmdb', "
                    f"but received {self.gt_folder}"
                )
            with open(osp.join(self.gt_folder, "meta_info.txt")) as fin:
                self.paths = [line.split(".")[0] for line in fin]
        else:
            if isinstance(self.gt_folder, list):
                self.paths = []
                for path_ in self.gt_folder:
                    self.paths += paths_from_folder(path_)
            else:
                self.paths = paths_from_folder(self.gt_folder)

        # inpainting mask
        self.gen_inpaint_mask = opt.get("gen_inpaint_mask", False)
        if self.gen_inpaint_mask:
            logger.info(f"generate mask ...")
            # self.mask_max_angle = opt.get('mask_max_angle', 10)
            # self.mask_max_len = opt.get('mask_max_len', 150)
            # self.mask_max_width = opt.get('mask_max_width', 50)
            # self.mask_draw_times = opt.get('mask_draw_times', 10)
            # # print
            # logger.info(f'mask_max_angle: {self.mask_max_angle}')
            # logger.info(f'mask_max_len: {self.mask_max_len}')
            # logger.info(f'mask_max_width: {self.mask_max_width}')
            # logger.info(f'mask_draw_times: {self.mask_draw_times}')

        # perform corrupt
        self.use_corrupt = opt.get("use_corrupt", True)
        self.use_motion_kernel = False
        # self.use_motion_kernel = opt.get('use_motion_kernel', True)

        if self.use_corrupt and not self.gen_inpaint_mask:
            # degradation configurations
            self.blur_kernel_size = opt["blur_kernel_size"]
            self.blur_sigma = opt["blur_sigma"]
            self.kernel_list = opt["kernel_list"]
            self.kernel_prob = opt["kernel_prob"]
            self.downsample_range = opt["downsample_range"]
            self.noise_range = opt["noise_range"]
            self.jpeg_range = opt["jpeg_range"]
            # print
            logger.info(
                f'Blur: blur_kernel_size {self.blur_kernel_size}, sigma: [{", ".join(map(str, self.blur_sigma))}]'
            )
            logger.info(
                f'Downsample: downsample_range [{", ".join(map(str, self.downsample_range))}]'
            )
            logger.info(f'Noise: [{", ".join(map(str, self.noise_range))}]')
            logger.info(f'JPEG compression: [{", ".join(map(str, self.jpeg_range))}]')

        # color jitter
        self.color_jitter_prob = opt.get("color_jitter_prob", None)
        self.color_jitter_pt_prob = opt.get("color_jitter_pt_prob", None)
        self.color_jitter_shift = opt.get("color_jitter_shift", 20)
        if self.color_jitter_prob is not None:
            logger.info(
                f"Use random color jitter. Prob: {self.color_jitter_prob}, shift: {self.color_jitter_shift}"
            )

        # to gray
        self.gray_prob = opt.get("gray_prob", 0.0)
        if self.gray_prob is not None:
            logger.info(f"Use random gray. Prob: {self.gray_prob}")
        self.color_jitter_shift /= 255.0

    @staticmethod
    def color_jitter(img, shift, np_state=None):
        """jitter color: randomly jitter the RGB values, in numpy formats"""
        if np_state != None:
            np.random.set_state(np_state)
        jitter_val = np.random.uniform(-shift, shift, 3).astype(np.float32)
        img = img + jitter_val
        np.clip(img, 0, 1, out=img)
        return img

    @staticmethod
    def color_jitter_pt(img, brightness, contrast, saturation, hue):
        """jitter color: randomly jitter the brightness, contrast, saturation, and hue, in torch Tensor formats"""
        fn_idx = torch.randperm(4)
        for fn_id in fn_idx:
            if fn_id == 0 and brightness is not None:
                brightness_factor = (
                    torch.tensor(1.0).uniform_(brightness[0], brightness[1]).item()
                )
                img = adjust_brightness(img, brightness_factor)

            if fn_id == 1 and contrast is not None:
                contrast_factor = (
                    torch.tensor(1.0).uniform_(contrast[0], contrast[1]).item()
                )
                img = adjust_contrast(img, contrast_factor)

            if fn_id == 2 and saturation is not None:
                saturation_factor = (
                    torch.tensor(1.0).uniform_(saturation[0], saturation[1]).item()
                )
                img = adjust_saturation(img, saturation_factor)

            if fn_id == 3 and hue is not None:
                hue_factor = torch.tensor(1.0).uniform_(hue[0], hue[1]).item()
                img = adjust_hue(img, hue_factor)
        return img

    def get_component_locations(self, name, status):
        components_bbox = self.components_dict[name]
        if status[0]:  # hflip
            # exchange right and left eye
            tmp = components_bbox["left_eye"]
            components_bbox["left_eye"] = components_bbox["right_eye"]
            components_bbox["right_eye"] = tmp
            # modify the width coordinate
            components_bbox["left_eye"][0] = (
                self.gt_size - components_bbox["left_eye"][0]
            )
            components_bbox["right_eye"][0] = (
                self.gt_size - components_bbox["right_eye"][0]
            )
            components_bbox["nose"][0] = self.gt_size - components_bbox["nose"][0]
            components_bbox["mouth"][0] = self.gt_size - components_bbox["mouth"][0]

        locations_gt = {}
        locations_in = {}
        for part in ["left_eye", "right_eye", "nose", "mouth"]:
            mean = components_bbox[part][0:2]
            half_len = components_bbox[part][2]
            if "eye" in part:
                half_len *= self.eye_enlarge_ratio
            elif part == "nose":
                half_len *= self.nose_enlarge_ratio
            elif part == "mouth":
                half_len *= self.mouth_enlarge_ratio
            loc = np.hstack((mean - half_len + 1, mean + half_len))
            loc = torch.from_numpy(loc).float()
            locations_gt[part] = loc
            loc_in = loc / (self.gt_size // self.in_size)
            locations_in[part] = loc_in
        return locations_gt, locations_in

    def degrade_img(self, opt, img, degrade_en, np_state=None):
        mean = opt.get("mean", [0.5, 0.5, 0.5])
        std = opt.get("std", [0.5, 0.5, 0.5])
        color_jitter_shift = opt.get("color_jitter_shift", 20)
        color_jitter_shift /= 255.0

        if np_state != None:
            np.random.set_state(np_state)

        if self.use_corrupt:
            # gaussian blur
            if degrade_en["gaussian_en"]:
                kernel = gaussian_kernels.random_mixed_kernels(
                    opt["kernel_list"],
                    opt["kernel_prob"],
                    opt["blur_kernel_size"],
                    opt["blur_sigma"],
                    opt["blur_sigma"],
                    [-math.pi, math.pi],
                    noise_range=None,
                    np_state=np_state,
                )
                img = cv2.filter2D(img, -1, kernel)

            # downsample
            if degrade_en["downsample_en"]:
                if np_state != None:
                    np.random.set_state(np_state)
                scale = np.random.uniform(
                    opt["downsample_range"][0], opt["downsample_range"][1]
                )
                img = cv2.resize(
                    img,
                    (int(opt["gt_size"] // scale), int(opt["gt_size"] // scale)),
                    interpolation=cv2.INTER_LINEAR,
                )

            # noise
            if degrade_en["noise_en"]:
                if np_state != None:
                    np.random.set_state(np_state)
                noise_sigma = np.random.uniform(
                    opt["noise_range"][0] / 255.0, opt["noise_range"][1] / 255.0
                )
                if np_state != None:
                    np.random.set_state(np_state)
                noise = np.float32(np.random.randn(*(img.shape))) * noise_sigma
                img = img + noise
                np.clip(img, 0, 1, out=img)

            # # color jitter
            # if degrade_en["color_jitter_en"]:
            #     img = color_jitter(img, color_jitter_shift, np_state)

            # # random to grey
            # if degrade_en["grey_en"]:
            #     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            #     img = np.tile(img[:, :, None], [1, 1, 3])

        # resize to in_size
        img = cv2.resize(
            img, (opt["in_size"], opt["in_size"]), interpolation=cv2.INTER_LINEAR
        )
        np.clip(img, 0, 1, out=img)
        return img

    def gen_video_flickers(self, opt, x):
        frame_list = []
        img_gt_np = list(x)
        for frame in img_gt_np:
            flick_en = np.random.uniform(0, 1) < opt["flick_probe"]
            if flick_en:
                brightness_factor = np.random.uniform(
                    opt["flick_range"][0], opt["flick_range"][1]
                )
                frame = cv2.convertScaleAbs(frame, alpha=brightness_factor, beta=0)
            frame_list.append(frame)

        frame_np = np.array(frame_list)
        return frame_np

    def do_ffmpeg_degrade(self, opt, x):
        crf_range = opt["crf_range"]
        crf = np.random.randint(crf_range[0], crf_range[1])
        ret_list = list(x)
        buf = io.BytesIO()
        with av.open(buf, "w", "mp4") as container:
            codec = "mpeg4"
            stream = container.add_stream(codec, rate=1)
            stream.height = opt["in_size"]
            stream.width = opt["in_size"]
            stream.pix_fmt = "yuv420p"
            stream.options = {"crf": str(crf)}

            for img_lq in ret_list:
                img_lq = np.clip(img_lq * 255, 0, 255).astype(np.uint8)
                img_lq = cv2.cvtColor(img_lq, cv2.COLOR_RGB2BGR)
                frame = av.VideoFrame.from_ndarray(img_lq, format="rgb24")
                frame.pict_type = "NONE"
                for packet in stream.encode(frame):
                    container.mux(packet)

            # Flush stream
            for packet in stream.encode():
                container.mux(packet)

        ret_list = []
        with av.open(buf, "r", "mp4") as container:
            if container.streams.video:
                for frame in container.decode(**{"video": 0}):
                    ret_list.append(frame.to_rgb().to_ndarray())

        ret_list = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in ret_list]
        ret_np = np.array(ret_list).astype(np.float32)
        ret_np /= 255.0
        return ret_np

    def load_sd_flicker_videos(self, opt, name, n_frame, start_idx):
        lq_base = "/data/vfhq-lq/flickeringsd"
        lq_path = os.path.join(lq_base, name + ".mp4")
        img_gt_list = []
        cnt = 0
        cap = cv2.VideoCapture(lq_path)
        while cap.isOpened():
            ret_flag, image = cap.read()
            if ret_flag == True:
                image = cv2.resize(
                    image,
                    (int(opt["gt_size"]), int(opt["gt_size"])),
                    interpolation=cv2.INTER_LINEAR,
                )
                image = img2tensor_np(image, bgr2rgb=True, float32=True)
                img_gt_list.append(image)
                cnt = cnt + 1
                if cnt == 240:
                    break
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                break
        cap.release()

        n_all = len(img_gt_list)
        img_gt_np = np.array(
            img_gt_list[start_idx : start_idx + n_frame]
        )  # [T, C, H, W], T = 240
        img_gt_np = img_gt_np.astype(np.float32) / 255.0
        return img_gt_np

    def __getitem__(self, index):
        # load gt image
        gt_path = self.paths[index]
        name = osp.basename(gt_path)[:-4]

        n_frame = 24
        img_gt_list = []
        cnt = 0
        cap = cv2.VideoCapture(gt_path)
        while cap.isOpened():
            ret_flag, image = cap.read()
            if ret_flag == True:
                image = cv2.resize(
                    image,
                    (int(self.opt["gt_size"]), int(self.opt["gt_size"])),
                    interpolation=cv2.INTER_LINEAR,
                )
                image = img2tensor_np(image, bgr2rgb=True, float32=True)
                img_gt_list.append(image)
                cnt = cnt + 1
                if cnt == 240:
                    break
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                break
        cap.release()

        n_all = len(img_gt_list)
        start_idx = random.randint(0, n_all - n_frame)

        img_gt_np = np.array(img_gt_list[start_idx : start_idx + n_frame])
        img_gt_np = img_gt_np.astype(np.float32) / 255.0

        degrade_en = {}
        degrade_en["gaussian_en"] = np.random.uniform() < self.opt.get(
            "gaussian_en", 0.1
        )
        degrade_en["grey_en"] = np.random.uniform() < self.opt.get("grey_en", 0.1)
        degrade_en["color_jitter_en"] = np.random.uniform() < self.opt.get(
            "color_jitter_en", 0.1
        )
        degrade_en["noise_en"] = np.random.uniform() < self.opt.get("noise_en", 0.1)
        degrade_en["ffmpeg_en"] = np.random.uniform() < self.opt.get("ffmpeg_en", 0.1)
        degrade_en["downsample_en"] = np.random.uniform() < self.opt.get(
            "downsample_en", 0.1
        )
        degrade_en["inpaint_en"] = np.random.uniform() < self.opt.get("inpaint_en", 0.1)
        degrade_en["flick_en"] = np.random.uniform() < self.opt.get("flick_en", 0.1)
        degrade_en["flick_sd_en"] = np.random.uniform() < self.opt.get(
            "flick_sd_en", 0.1
        )

        np_state = np.random.get_state()

        # generate in image
        img_in_list = []
        for fidx in range(n_frame):
            lq_img = self.degrade_img(self.opt, img_gt_np[fidx], degrade_en, np_state)
            img_in_list.append(lq_img)
        img_in_np = np.array(img_in_list)

        if degrade_en["flick_en"]:
            img_in_np = img_in_np * 255.0
            img_in_np = self.gen_video_flickers(self.opt, img_in_np)
            img_in_np = img_in_np / 255.0

        if degrade_en["ffmpeg_en"]:
            img_in_np = self.do_ffmpeg_degrade(self.opt, img_in_np)

        if degrade_en["flick_sd_en"]:
            img_in_np = self.load_sd_flicker_videos(self.opt, name, n_frame, start_idx)

        img_in_np = img_in_np.transpose(3, 0, 1, 2)
        img_gt_np = img_gt_np.transpose(3, 0, 1, 2)

        img_in = torch.from_numpy(img_in_np).float()
        img_gt = torch.from_numpy(img_gt_np).float()

        return_dict = {"in": img_in, "gt": img_gt, "gt_path": gt_path}
        return return_dict

    def __len__(self):
        return len(self.paths)

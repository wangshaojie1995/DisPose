import os
import argparse
import logging
import math
from omegaconf import OmegaConf
from datetime import datetime
import time
from pathlib import Path
import PIL.Image
import numpy as np
import torch.jit
from torchvision.datasets.folder import pil_loader
from torchvision.transforms.functional import pil_to_tensor, resize, center_crop
from torchvision.transforms.functional import to_pil_image
from torchvision import transforms
import torch.nn.functional as F
from torchvision.transforms import PILToTensor
import torchvision

import decord
from einops import rearrange, repeat
from mimicmotion.utils.dift_utils import SDFeaturizer
from mimicmotion.utils.utils import points_to_flows, bivariate_Gaussian, sample_inputs_flow, get_cmp_flow, pose2track
from  mimicmotion.utils.visualizer import Visualizer, vis_flow_to_video
import cv2



from mimicmotion.utils.geglu_patch import patch_geglu_inplace
patch_geglu_inplace()

from constants import ASPECT_RATIO
from mimicmotion.utils.loader import create_ctrl_pipeline
from mimicmotion.utils.utils import save_to_mp4
from mimicmotion.dwpose.preprocess import get_video_pose, get_image_pose
from mimicmotion.modules.cmp_model import CMP


import pdb
logging.basicConfig(level=logging.INFO, format="%(asctime)s: [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preprocess(video_path, image_path, dift_model_path, resolution=576, sample_stride=2):
    """preprocess ref image pose and video pose

    Args:
        video_path (str): input video pose path
        image_path (str): reference image path
        resolution (int, optional):  Defaults to 576.
        sample_stride (int, optional): Defaults to 2.
    """
    image_pixels = pil_loader(image_path)
    image_pixels = pil_to_tensor(image_pixels) # (c, h, w)
    h, w = image_pixels.shape[-2:]
    ############################ compute target h/w according to original aspect ratio ###############################
    if h>w:
        w_target, h_target = resolution, int(resolution / ASPECT_RATIO // 64) * 64
    elif h==w:
        w_target, h_target = resolution, resolution
    else:
        w_target, h_target = int(resolution / ASPECT_RATIO // 64) * 64, resolution
    h_w_ratio = float(h) / float(w)
    if h_w_ratio < h_target / w_target:
        h_resize, w_resize = h_target, math.ceil(h_target / h_w_ratio)
    else:
        h_resize, w_resize = math.ceil(w_target * h_w_ratio), w_target
    image_pixels = resize(image_pixels, [h_resize, w_resize], antialias=None)
    image_pixels = center_crop(image_pixels, [h_target, w_target])
    # h_target, w_target = image_pixels.shape[-2:]
    image_pixels = image_pixels.permute((1, 2, 0)).numpy()
    ##################################### get video flow #################################################
    transform = transforms.Compose(
        [
        
        transforms.Resize((h_target, w_target), antialias=None), 
        transforms.CenterCrop((h_target, w_target)), 
        transforms.ToTensor()
        ]
    )
    
    ref_img = transform(PIL.Image.fromarray(image_pixels))

    ##################################### get image&video pose value #################################################
    image_pose, ref_point = get_image_pose(image_pixels)
    ref_point_body, ref_point_head = ref_point["bodies"], ref_point["faces"]
    video_pose, body_point, face_point = get_video_pose(video_path, image_pixels, sample_stride=sample_stride)
    body_point_list = [ref_point_body] + body_point
    face_point_list = [ref_point_head] + face_point

    pose_pixels = np.concatenate([np.expand_dims(image_pose, 0), video_pose])
    image_pixels = np.transpose(np.expand_dims(image_pixels, 0), (0, 3, 1, 2))
    
    dift_model = SDFeaturizer(sd_id = dift_model_path, weight_dtype=torch.float16)
    category="human"
    prompt = f'photo of a {category}'
    dift_ref_img = (image_pixels / 255.0 - 0.5) *2
    dift_ref_img = torch.from_numpy(dift_ref_img).to(device, torch.float16)
    dift_feats = dift_model.forward(dift_ref_img, prompt=prompt, t=[261,0], up_ft_index=[1,2], ensemble_size=8)


    model_length = len(body_point_list)
    traj_flow = points_to_flows(body_point_list, model_length, h_target, w_target)
    blur_kernel = bivariate_Gaussian(kernel_size=199, sig_x=20, sig_y=20, theta=0, grid=None, isotropic=True)

    for i in range(0, model_length-1):
        traj_flow[i] = cv2.filter2D(traj_flow[i], -1, blur_kernel)

    traj_flow = rearrange(traj_flow, "f h w c -> f c h w") 
    traj_flow = torch.from_numpy(traj_flow)
    traj_flow = traj_flow.unsqueeze(0)

    cmp = CMP(
        './mimicmotion/modules/cmp/experiments/semiauto_annot/resnet50_vip+mpii_liteflow/config.yaml',
        42000
    ).to(device)
    cmp.requires_grad_(False)

    pc, ph, pw = ref_img.shape
    poses, poses_subset = pose2track(body_point_list, ph, pw)
    poses = torch.from_numpy(poses).permute(1,0,2)
    poses_subset = torch.from_numpy(poses_subset).permute(1,0,2)

    # pdb.set_trace()
    val_controlnet_image, val_sparse_optical_flow, \
    val_mask, val_first_frame_384, \
        val_sparse_optical_flow_384, val_mask_384 = sample_inputs_flow(ref_img.unsqueeze(0).float(), poses.unsqueeze(0), poses_subset.unsqueeze(0))

    fb, fl, fc, fh, fw = val_sparse_optical_flow.shape

    val_controlnet_flow = get_cmp_flow(
        cmp, 
        val_first_frame_384.unsqueeze(0).repeat(1, fl, 1, 1, 1).to(device), 
        val_sparse_optical_flow_384.to(device), 
        val_mask_384.to(device)
    )

    if fh != 384 or fw != 384:
        scales = [fh / 384, fw / 384]
        val_controlnet_flow = F.interpolate(val_controlnet_flow.flatten(0, 1), (fh, fw), mode='nearest').reshape(fb, fl, 2, fh, fw)
        val_controlnet_flow[:, :, 0] *= scales[1]
        val_controlnet_flow[:, :, 1] *= scales[0]
    
    vis_flow = val_controlnet_flow[0]

    return torch.from_numpy(pose_pixels.copy()) / 127.5 - 1, torch.from_numpy(image_pixels) / 127.5 - 1, val_controlnet_flow, val_controlnet_image, body_point_list, dift_feats, traj_flow


def run_pipeline(pipeline, image_pixels, pose_pixels,
                controlnet_flow, controlnet_image, point_list, dift_feats, traj_flow,
                device, task_config):
    image_pixels = [to_pil_image(img.to(torch.uint8)) for img in (image_pixels + 1.0) * 127.5]
    generator = torch.Generator(device=device)
    generator.manual_seed(task_config.seed)
    with torch.autocast("cuda"):
        frames = pipeline(
            image_pixels, image_pose=pose_pixels, num_frames=pose_pixels.size(0),
            tile_size=task_config.num_frames, tile_overlap=task_config.frames_overlap,
            height=pose_pixels.shape[-2], width=pose_pixels.shape[-1], fps=7,
            controlnet_flow=controlnet_flow, controlnet_image=controlnet_image, point_list=point_list, dift_feats=dift_feats, traj_flow=traj_flow,
            noise_aug_strength=task_config.noise_aug_strength, num_inference_steps=task_config.num_inference_steps,
            generator=generator, min_guidance_scale=task_config.guidance_scale, 
            max_guidance_scale=task_config.guidance_scale, decode_chunk_size=task_config.decode_chunk_size, output_type="pt", device=device
        ).frames.cpu()
    video_frames = (frames * 255.0).to(torch.uint8)

    for vid_idx in range(video_frames.shape[0]):
        # deprecated first frame because of ref image
        _video_frames = video_frames[vid_idx, 1:]

    return _video_frames


@torch.no_grad()
def main(args):
    if not args.no_use_float16 :
        torch.set_default_dtype(torch.float16)

    infer_config = OmegaConf.load(args.inference_config)
    pipeline = create_ctrl_pipeline(infer_config, device)

    for task in infer_config.test_case:
        ############################################## Pre-process data ##############################################
        pose_pixels, image_pixels, controlnet_flow, controlnet_image, point_list, dift_feats, traj_flow = preprocess(
            task.ref_video_path, task.ref_image_path, infer_config.dift_model_path, 
            resolution=task.resolution, sample_stride=task.sample_stride
        )
        ########################################### Run MimicMotion pipeline ###########################################
        _video_frames = run_pipeline(
            pipeline, 
            image_pixels, pose_pixels, controlnet_flow, controlnet_image, point_list, dift_feats, traj_flow,
            device, task
        )
        ################################### save results to output folder. ###########################################
        save_to_mp4(
            _video_frames, 
            f"{args.output_dir}/{datetime.now().strftime('%Y%m%d')}_{args.name}/{datetime.now().strftime('%H%M%S')}_{os.path.basename(task.ref_image_path).split('.')[0]}_to_{os.path.basename(task.ref_video_path).split('.')[0]}" \
            f"_CFG{task.guidance_scale}_{task.num_frames}_{task.fps}.mp4",
            fps=task.fps,
        )

def set_logger(log_file=None, log_level=logging.INFO):
    log_handler = logging.FileHandler(log_file, "w")
    log_handler.setFormatter(
        logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s]: %(message)s")
    )
    log_handler.setLevel(log_level)
    logger.addHandler(log_handler)


if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_file", type=str, default=None)
    parser.add_argument("--inference_config", type=str, default="configs/test.yaml") #ToDo
    parser.add_argument("--output_dir", type=str, default="outputs/", help="path to output")
    parser.add_argument("--name", type=str, default="")
    parser.add_argument("--no_use_float16",
                        action="store_true",
                        help="Whether use float16 to speed up inference",
    )
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
    logger.info(f"--- Finished ---")


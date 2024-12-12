import json
import random
from typing import List

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from decord import VideoReader
from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPImageProcessor
# from ..utils.common import get_moved_area_mask, calculate_motion_score
import os
import pdb
import copy
import cv2
import pickle
# from src.dataset.data_utils import process_bbox, crop_bbox, mask_to_bbox, mask_to_bkgd
# from tools.point2flow import interpolate_trajectory, divide_points_afterinterpolate, get_sparseflow_and_mask_forward

def crop_bbox(img, bbox, do_resize=False, size=512):
    
    if isinstance(img, (Path, str)):
        img = Image.open(img)
    cropped_img = img.crop(bbox)
    if do_resize:
        cropped_W, cropped_H = cropped_img.size
        ratio = size / max(cropped_W, cropped_H)
        new_W = cropped_W * ratio
        new_H = cropped_H * ratio
        cropped_img = cropped_img.resize((new_W, new_H))
    
    return cropped_img

class HumanDanceVideoDataset(Dataset):
    def __init__(
        self,
        sample_rate,
        n_sample_frames,
        width,
        height,
        img_scale=(1.0, 1.0),
        img_ratio=(1.0, 1.0),
        drop_ratio=0.1,
        data_meta_paths=["./data/fashion_meta.json"],
        motion_mask_resize=False,
        move_th=20,
        # 
        bbox_crop=False,
        bbox_resize_ratio=(0.8, 1.2),
        aug_type: str = "Resize",  # "Resize" or "Padding"
        first_frames=True,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_sample_frames = n_sample_frames
        self.width = width
        self.height = height
        self.img_size = (width, height)
        self.img_scale = img_scale
        self.img_ratio = img_ratio
        self.motion_mask_resize = motion_mask_resize
        self.move_th = move_th

        self.bbox_crop = bbox_crop
        self.bbox_resize_ratio = bbox_resize_ratio
        self.aug_type = aug_type
        self.first_frames = first_frames
        if self.first_frames:
            self.n_sample_frames = n_sample_frames -1
        vid_meta = []
        for data_meta_path in data_meta_paths:
            vid_meta.extend(json.load(open(data_meta_path, "r")))
        self.vid_meta = vid_meta

        self.clip_image_processor = CLIPImageProcessor()
        self.pixel_transform, self.cond_transform = self.setup_transform()
        # self.pixel_transform = transforms.Compose(
        #     [
        #         transforms.RandomResizedCrop(
        #             (height, width),
        #             scale=self.img_scale,
        #             ratio=self.img_ratio,
        #             interpolation=transforms.InterpolationMode.BILINEAR,
        #         ),
        #         transforms.ToTensor(),
        #         transforms.Normalize([0.5], [0.5]),
        #     ]
        # )

        # self.cond_transform = transforms.Compose(
        #     [
        #         transforms.RandomResizedCrop(
        #             (height, width),
        #             scale=self.img_scale,
        #             ratio=self.img_ratio,
        #             interpolation=transforms.InterpolationMode.BILINEAR,
        #         ),
        #         transforms.ToTensor(),
        #     ]
        # )

        self.drop_ratio = drop_ratio

    def augmentation(self, images, transform, state=None):
        if state is not None:
            torch.set_rng_state(state)
        if isinstance(images, List):
            transformed_images = [transform(img) for img in images]
            ret_tensor = torch.stack(transformed_images, dim=0)  # (f, c, h, w)
        else:
            ret_tensor = transform(images)  # (c, h, w)
        return ret_tensor

    def resize_long_edge(self, img):
        img_W, img_H = img.size
        long_edge = max(img_W, img_H)
        scale = self.image_size / long_edge
        new_W, new_H = int(img_W * scale), int(img_H * scale)
        
        img = F.resize(img, (new_H, new_W))
        return img

    def padding_short_edge(self, img):
        img_W, img_H = img.size
        width, height = self.image_size, self.image_size
        padding_left = (width - img_W) // 2
        padding_right = width - img_W - padding_left
        padding_top = (height - img_H) // 2
        padding_bottom = height - img_H - padding_top
        
        img = F.pad(img, (padding_left, padding_top, padding_right, padding_bottom), 0, "constant")
        return img

    def setup_transform(self):
        if True:
            if self.aug_type == "Resize":
                pixel_transform = transforms.Compose([
                    transforms.Resize((self.img_size[1], self.img_size[0])),
                    transforms.ToTensor(),
                    # transforms.Normalize([0.5], [0.5]),
                ])
                guid_transform = transforms.Compose([
                    transforms.Resize((self.img_size[1], self.img_size[0])),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ])
                
            elif self.aug_type == "Padding":
                pixel_transform = transforms.Compose([
                    transforms.Lambda(self.resize_long_edge),
                    transforms.Lambda(self.padding_short_edge),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ])
                guid_transform = transforms.Compose([
                    transforms.Lambda(self.resize_long_edge),
                    transforms.Lambda(self.padding_short_edge),
                    transforms.ToTensor(),
                ])
            else:
                raise NotImplementedError("Do not support this augmentation")
        
        else:
            pixel_transform = transforms.Compose([
                transforms.RandomResizedCrop(size=(self.img_size[1], self.img_size[0]), scale=(0.9, 1.0), ratio=(1.0, 1.0)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])
            guid_transform = transforms.Compose([
                transforms.RandomResizedCrop(size=(self.img_size[1], self.img_size[0]), scale=(0.9, 1.0), ratio=(1.0, 1.0)),
                transforms.ToTensor(),
            ])
        
        return pixel_transform, guid_transform   

    def __getitem__(self, index):
        video_meta = self.vid_meta[index]
        video_path = video_meta["video_path"]
        kps_path = video_meta["dwpose_path"]
        # mask_path = video_meta["mask_path"]

        try:
            video_reader = VideoReader(video_path)
            kps_reader = VideoReader(kps_path)
            # mask_reader = VideoReader(mask_path)
            # array_path = kps_path.replace("video_dwpose_no_face","video_dwpose_array")+".pkl"
            array_path = kps_path +".pkl"
            with open(array_path, 'rb') as f:
                dwpose_array = pickle.load(f) 
            H_tmp = dwpose_array["H"]
            W_tmp = dwpose_array["W"]
            poses = dwpose_array["poses"]
            if H_tmp <= W_tmp:
                # pdb.set_trace()
                print(f"H: {H_tmp} <= W: {W_tmp}")
                return self.__getitem__(np.random.randint(0, len(self.vid_meta)))
        except Exception as e:
            print("Error:", e)
            try:
                return self.__getitem__(np.random.randint(0, len(self.vid_meta)))
            except Exception as e:
                print("Error:", e)
                return self.__getitem__(0)
        # assert len(video_reader) == len(
            # kps_reader
        # ), f"{len(video_reader) = } != {len(kps_reader) = } in {video_path}"
        if len(video_reader) != len(kps_reader):
            print(f"{len(video_reader) = } != {len(kps_reader) = } in {video_path}")
            # self.error_video.append(video_path)
            # with open('error_video.json', 'a+') as f:
                # f.write(json.dumps(video_path) + '\n')
                # json.dump(video_path, f)
            return self.__getitem__(np.random.randint(0, len(self.vid_meta)))

        video_length = len(video_reader)

        # sample_rate = random.choice(self.sample_rate)
        sample_rate = self.sample_rate

        clip_length = min(
            video_length, (self.n_sample_frames - 1) * sample_rate + 1
        )
        start_idx = random.randint(0, video_length - clip_length)
        batch_index = np.linspace(
            start_idx, start_idx + clip_length - 1, self.n_sample_frames, dtype=int
        ).tolist()
        # batch_index = [start_idx] + batch_index

        ref_img_idx = random.randint(0, video_length - 1)
        if self.first_frames:
            batch_index = [ref_img_idx] + batch_index

        ref_img = Image.fromarray(video_reader[ref_img_idx].asnumpy())
        ref_pose = Image.fromarray(kps_reader[ref_img_idx].asnumpy())
        ref_point = poses[ref_img_idx]['bodies']
        # ref_mask = mask_reader[ref_img_idx].asnumpy()

        # tmp_mask = copy.deepcopy(ref_mask)
        # # ref_mask.save("vis_img/ref_mask.png")
        # tmp_mask = tmp_mask.astype(np.float32) / 255.0
        # tmp_mask[tmp_mask < 0.5] = 0
        # tmp_mask[tmp_mask >= 0.5] = 1

        # masked_ref_img = ref_img * (tmp_mask < 0.5)
        # masked_ref_img_pil = Image.fromarray((masked_ref_img).astype(np.uint8))
        # # masked_ref_img_pil.save("vis_img/masked_ref_img.png")
        # ref_mask = cv2.cvtColor(ref_mask, cv2.COLOR_BGR2GRAY)
        # ref_mask_pil = Image.fromarray((ref_mask).astype(np.uint8))

        # read frames and kps
        vid_pil_image_list = []
        pose_pil_image_list = []
        mask_pil_image_list = []
        tgt_points = []
        ############################## for point2flow #############################
        # input_drag_384_inmask_list = []
        # input_mask_384_inmask_list = []
        # input_drag_384_outmask_list = []
        # input_mask_384_outmask_list = []
        # motion_brush_mask_384_list=[]
        ############################## for point2flow #############################

        ref_W, ref_H = ref_img.size
        state = torch.get_rng_state()
        # bbox crop
        if self.bbox_crop:

            bbox_path = mask_path.replace("video_mask","video_mask_bbox")+".json"
            with open(bbox_path) as bbox_fp:
                human_bboxes = json.load(bbox_fp)
            x, y, width, height = human_bboxes['x'], human_bboxes['y'], human_bboxes['width'], human_bboxes['height']
            bbox = (x, y, x + width, y + height)

            resize_scale = random.uniform(*self.bbox_resize_ratio)
            bbox = process_bbox(bbox, ref_H, ref_W, resize_scale)
            ref_img = crop_bbox(ref_img, bbox)
            ref_pose = crop_bbox(ref_pose, bbox)
        # pose_pil_image_list.append(ref_pose)
        for index in batch_index:

            img = video_reader[index]
            vid_pil_image = Image.fromarray(img.asnumpy())
            vid_pil_image_list.append(crop_bbox(vid_pil_image, bbox) if self.bbox_crop else vid_pil_image)

            img = kps_reader[index]
            pose_pil_image = Image.fromarray(img.asnumpy())
            # pose_pil_image.save("vis_img/tgt_pose.png")
            pose_pil_image_list.append(crop_bbox(pose_pil_image, bbox) if self.bbox_crop else pose_pil_image)

            # img = mask_reader[index].asnumpy()
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # mask_pil_image = Image.fromarray(img)
            # mask_pil_image_list.append(crop_bbox(mask_pil_image, bbox) if self.bbox_crop else mask_pil_image)

            tgt_points.append(poses[index]['bodies'])
            ############################## for point2flow #############################
            # ref_pose = 
            # p1 = np.expand_dims(poses[ref_img_idx]['bodies']['candidate'], axis=1) * np.array([ref_W,ref_H])
            # p2 = np.expand_dims(poses[index]['bodies']['candidate'], axis=1) * np.array([ref_W,ref_H])

            # tracking_points = np.concatenate([p1, p2], axis=1).astype(np.int64)
            # s1 = poses[ref_img_idx]['bodies']['subset']
            # s2 = poses[index]['bodies']['subset']
            # tracking_subset = np.concatenate([s1, s2], axis=0).T.astype(np.int64)
            
            # # transparent_background = Image.fromarray((ref_img).astype(np.uint8)).convert('RGBA')
            # transparent_background = ref_img.convert('RGBA')
            # crop_w, crop_h = transparent_background.size
            # transparent_layer = np.zeros((crop_h, crop_w, 4))
            # # ref_img_pil.save("vis_img/resize_crop_ref_img_pil.png")

            # for idx, (track, subset) in enumerate(zip(tracking_points, tracking_subset)):
            #     if -1 in subset:
            #         continue
            #     if len(track) > 1:
            #         for i in range(len(track)-1):
            #             start_point = track[i]
            #             end_point = track[i+1]
            #             if start_point[1] > ref_H or end_point[1] > ref_H or start_point[0] >ref_W or end_point[0] > ref_W:
            #                 # print(track)
            #                 tracking_subset[idx][0] = -1
            #                 continue
                        # # crop resize
                        # tracking_points[idx][i][0] = start_point[0] - bbox[0]
                        # tracking_points[idx][i][1] = start_point[1] - bbox[1]
                        # tracking_points[idx][i+1][0] = end_point[0] - bbox[0]
                        # tracking_points[idx][i+1][1] = end_point[1] - bbox[1]

            #             start_point = tracking_points[idx][i]
            #             end_point = tracking_points[idx][i+1]

            #             vx = end_point[0] - start_point[0]
            #             vy = end_point[1] - start_point[1]
            #             arrow_length = np.sqrt(vx**2 + vy**2)
            #             cv2.circle(transparent_layer, tuple(track[0]), 3, (255, 0, 0, 255), -1)
            #             cv2.circle(transparent_layer, tuple(track[1]), 3, (255, 0, 0, 255), -1)
            #             if i == len(track)-2:
            #                 cv2.arrowedLine(transparent_layer, tuple(start_point), tuple(end_point), (255, 0, 0, 255), 2, tipLength=(8 / arrow_length))
            #             else:
            #                 cv2.line(transparent_layer, tuple(start_point), tuple(end_point), (255, 0, 0, 255), 2,)
            #     else:
            #         cv2.circle(transparent_layer, tuple(track[0]), 3, (255, 0, 0, 255), -1)
            # transparent_layer = Image.fromarray(transparent_layer.astype(np.uint8))
            # trajectory_map = Image.alpha_composite(transparent_background, transparent_layer)
            # trajectory_map.save("vis_img/test_traj_map.png")
            # transparent_background.save("vis_img/test_tarj_bg.png")
            # transparent_layer.save("vis_img/test_tarj_layer.png")

            # track_mask = (tracking_subset.min(1)>-1)
            # tracking_points = tracking_points[track_mask]

            # w, h = self.img_size
            # # motion_brush_mask = np.zeros((h, w))
            # # motion_brush_mask = tmp_mask
            # motion_brush_mask = self.augmentation(mask_pil_image, self.cond_transform, state)
            # motion_brush_mask = (motion_brush_mask.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            # # motion_brush_mask = (motion_brush_mask * 255).astype(np.uint8)

            # # Image.fromarray(motion_brush_mask, mode="L").save("vis_img/motion_brush_mask.png")
            # original_width, original_height = crop_w, crop_h
            # model_length = 5
            # input_all_points = tracking_points

            # resized_all_points = [tuple([tuple([int(e1[0]*w/original_width), int(e1[1]*h/original_height)]) for e1 in e]) for e in input_all_points]
            # resized_all_points_384 = [tuple([tuple([int(e1[0]*384/original_width), int(e1[1]*384/original_height)]) for e1 in e]) for e in input_all_points]

            # new_resized_all_points = []
            # new_resized_all_points_384 = []
            # for tnum in range(len(resized_all_points)):
            #     new_resized_all_points.append(interpolate_trajectory(input_all_points[tnum], model_length))
            #     new_resized_all_points_384.append(interpolate_trajectory(resized_all_points_384[tnum], model_length))

            # resized_all_points = np.array(new_resized_all_points)
            # resized_all_points_384 = np.array(new_resized_all_points_384)

            # motion_brush_mask_384 = cv2.resize(motion_brush_mask, (384, 384), cv2.INTER_NEAREST)

            # try:
            #     resized_all_points_384_inmask, resized_all_points_384_outmask = \
            #         divide_points_afterinterpolate(resized_all_points_384, motion_brush_mask_384)
            # except Exception as e:
            #     print("Error:", e)
            #     return self.__getitem__(np.random.randint(0, len(self.vid_meta)))

            # in_mask_flag = False
            # out_mask_flag = False
            
            # if resized_all_points_384_inmask.shape[0] != 0:
            #     in_mask_flag = True
            #     input_drag_384_inmask, input_mask_384_inmask = \
            #         get_sparseflow_and_mask_forward(
            #             resized_all_points_384_inmask, 
            #             model_length - 1, 384, 384
            #         )
            # else:
            #     input_drag_384_inmask, input_mask_384_inmask = \
            #         np.zeros((model_length - 1, 384, 384, 2)), \
            #             np.zeros((model_length - 1, 384, 384))

            # if resized_all_points_384_outmask.shape[0] != 0:
            #     out_mask_flag = True
            #     input_drag_384_outmask, input_mask_384_outmask = \
            #         get_sparseflow_and_mask_forward(
            #             resized_all_points_384_outmask, 
            #             model_length - 1, 384, 384
            #         )
            # else:
            #     input_drag_384_outmask, input_mask_384_outmask = \
            #         np.zeros((model_length - 1, 384, 384, 2)), \
            #             np.zeros((model_length - 1, 384, 384))
            # input_drag_384_inmask_list.append(torch.from_numpy(input_drag_384_inmask)),
            # input_mask_384_inmask_list.append(torch.from_numpy(input_mask_384_inmask)),
            # input_drag_384_outmask_list.append(torch.from_numpy(input_drag_384_outmask)),
            # input_mask_384_outmask_list.append(torch.from_numpy(input_mask_384_outmask)),
            # motion_brush_mask_384_list.append(torch.from_numpy(motion_brush_mask_384)),

        ############################## for point2flow #############################


        # motion_mask, mask_list = get_moved_area_mask(np.stack(pose_pil_image_list), move_th=self.move_th)
        # motion_mask, mask_list = get_moved_area_mask(np.stack([ref_img] + vid_pil_image_list), move_th=self.move_th)
        # motion_mask_pil = Image.fromarray(motion_mask)
        # mask_list_pil = [Image.fromarray(m) for m in mask_list]
        assert vid_pil_image_list[0].size == pose_pil_image_list[0].size == ref_img.size

        # transform
        pixel_values_vid = self.augmentation(
            vid_pil_image_list, self.pixel_transform, state
        )
        pixel_values_pose = self.augmentation(
            pose_pil_image_list, self.cond_transform, state
        )
        ref_pose = self.augmentation(ref_pose, self.cond_transform, state)
        # pixel_mask = self.augmentation(mask_pil_image_list, self.cond_transform, state)
        pixel_values_ref_img = self.augmentation(ref_img, self.pixel_transform, state)
        flow_tgt_imgs = self.augmentation(vid_pil_image_list, self.cond_transform, state)
        flow_ref_img = self.augmentation(ref_img, self.cond_transform, state)
        input_first_frame = self.augmentation(ref_img, self.cond_transform, state)
        clip_ref_img = self.clip_image_processor(
            images=ref_img, return_tensors="pt"
        ).pixel_values[0]
        # motion_mask = self.augmentation(motion_mask_pil, self.cond_transform, state)
        # temp_motion_mask = self.augmentation(mask_list_pil, self.cond_transform, state)
        # ref_mask = self.augmentation(ref_mask_pil, self.cond_transform, state)
        # masked_ref_img = self.augmentation(masked_ref_img_pil, self.pixel_transform, state)
        # mask = get_moved_area_mask(pixel_values_vid.permute([0,2,3,1]).numpy())

        sample = dict(
            video_dir=video_path,
            pixel_values_vid=pixel_values_vid,
            pixel_values_pose=pixel_values_pose,
            pixel_values_ref_img=pixel_values_ref_img,
            # pixel_mask=pixel_mask,
            clip_ref_img=clip_ref_img,
            pixel_values_flow=flow_tgt_imgs,
            flow_ref_img=flow_ref_img,
            ref_pose = ref_pose,
            ref_point = ref_point,
            tgt_points = tgt_points,
            # motion_mask=motion_mask,
            # temp_motion_mask=temp_motion_mask,
            # ref_mask = ref_mask,
            # masked_ref_img = masked_ref_img
            # input_drag_384_inmask = torch.stack(input_drag_384_inmask_list),
            # input_mask_384_inmask = torch.stack(input_mask_384_inmask_list),
            # input_drag_384_outmask = torch.stack(input_drag_384_outmask_list),
            # input_mask_384_outmask = torch.stack(input_mask_384_outmask_list),
            # input_first_frame=input_first_frame,
            # motion_brush_mask_384=torch.stack(motion_brush_mask_384_list),
        )

        return sample

    def __len__(self):
        return len(self.vid_meta)

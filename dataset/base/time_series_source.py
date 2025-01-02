# !/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project : Deep-Learning-Project-Template
@Product : PyCharm
@File : time_series_source.py.py
@Author : Xuepu Zeng (2307665474zxp@gmail.com)
@Date : 2025/1/2 12:41
'''
from typing import Dict, Tuple, List, Callable
from omegaconf import OmegaConf
import os
import abc
import random
import logging
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
from torch import Tensor

logger = logging.getLogger()

class time_series_source(abc.ABC):
    """
        The base class for all pixel sources of a scene.
    """
    # define a transformation matrix to convert the opencv camera coordinate system to the dataset camera coordinate system
    data_cfg: OmegaConf = None
    # the dataset name, choose from ["waymo", "kitti", "nuscenes", "pandaset", "argoverse"]
    dataset_name: str = None
    # the dict of camera data
    camera_data: Dict[int, CameraData] = {}
    # the normalized time of all images (normalized to [0, 1]), shape: (num_frames,)
    _normalized_time: Tensor = None
    # timestep indices of frames, shape: (num_frames,)
    _timesteps: Tensor = None
    # image error buffer, (num_images, )
    image_error_buffer: Tensor = None
    # whether the buffer is computed
    image_error_buffered: bool = False
    # the downscale factor of the error buffer
    buffer_downscale: float = 1.0

    # -------------- object annotations
    # (num_frame, num_instances, 4, 4)
    instances_pose: Tensor = None
    # (num_instances, 3)
    instances_size: Tensor = None
    # (num_instances, )
    instances_true_id: Tensor = None
    # (num_instances, )
    instances_model_types: Tensor = None
    # (num_frame, num_instances)
    per_frame_instance_mask: Tensor = None

    def __init__(
            self, dataset_name, pixel_data_config: OmegaConf, device: torch.device = torch.device("cpu")
    ) -> None:
        # hold the config of the pixel data
        self.dataset_name = dataset_name
        self.data_cfg = pixel_data_config
        self.device = device
        self._downscale_factor = 1 / pixel_data_config.downscale
        self._old_downscale_factor = []

    @abc.abstractmethod
    def load_cameras(self) -> None:
        """
        Load the camera intrinsics, extrinsics, timestamps, etc.
        Load the images, dynamic masks, sky masks, etc.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def load_objects(self) -> None:
        """
        Load the object annotations.
        """
        raise NotImplementedError

    def load_data(self) -> None:
        """
        A general function to load all data.
        """
        self.load_cameras()
        self.build_image_error_buffer()
        logger.info("[Pixel] All Pixel Data loaded.")

        if self.data_cfg.load_objects:
            self.load_objects()
            logger.info("[Pixel] All Object Annotations loaded.")

        # set initial downscale factor
        for cam_id in self.camera_list:
            self.camera_data[cam_id].set_downscale_factor(self._downscale_factor)

    def to(self, device: torch.device) -> "ScenePixelSource":
        """
        Move the dataset to the given device.
        Args:
            device: the device to move the dataset to.
        """
        self.device = device
        if self._timesteps is not None:
            self._timesteps = self._timesteps.to(device)
        if self._normalized_time is not None:
            self._normalized_time = self._normalized_time.to(device)
        if self.instances_pose is not None:
            self.instances_pose = self.instances_pose.to(device)
        if self.instances_size is not None:
            self.instances_size = self.instances_size.to(device)
        if self.per_frame_instance_mask is not None:
            self.per_frame_instance_mask = self.per_frame_instance_mask.to(device)
        if self.instances_model_types is not None:
            self.instances_model_types = self.instances_model_types.to(device)
        return self

    def get_aabb(self) -> Tensor:
        """
        Returns:
            aabb_min, aabb_max: the min and max of the axis-aligned bounding box of the scene
        Note:
            We compute the coarse aabb by using the front camera positions / trajectories. We then
            extend this aabb by 40 meters along horizontal directions and 20 meters up and 5 meters
            down along vertical directions.
        """
        front_camera_trajectory = self.front_camera_trajectory

        # compute the aabb
        aabb_min = front_camera_trajectory.min(dim=0)[0]
        aabb_max = front_camera_trajectory.max(dim=0)[0]

        # extend aabb by 40 meters along forward direction and 40 meters along the left/right direction
        # aabb direction: x, y, z: front, left, up
        aabb_max[0] += 40
        aabb_max[1] += 40
        # when the car is driving uphills
        aabb_max[2] = min(aabb_max[2] + 20, 20)

        # for waymo, there will be a lot of waste of space because we don't have images in the back,
        # it's more reasonable to extend the aabb only by a small amount, e.g., 5 meters
        # we use 40 meters here for a more general case
        aabb_min[0] -= 40
        aabb_min[1] -= 40
        # when a car is driving downhills
        aabb_min[2] = max(aabb_min[2] - 5, -5)
        aabb = torch.tensor([*aabb_min, *aabb_max])
        logger.info(f"[Pixel] Auto AABB from camera: {aabb}")
        return aabb

    @property
    def front_camera_trajectory(self) -> Tensor:
        """
        Returns:
            the front camera trajectory.
        """
        front_camera = self.camera_data[0]
        assert (
                front_camera.cam_to_worlds is not None
        ), "Camera poses not loaded, cannot compute front camera trajectory."
        return front_camera.cam_to_worlds[:, :3, 3]

    def parse_img_idx(self, img_idx: int) -> Tuple[int, int]:
        """
        Parse the image index to the camera index and frame index.
        Args:
            img_idx: the image index.
        Returns:
            cam_idx: the camera index.
            frame_idx: the frame index.
        """
        unique_cam_idx = img_idx % self.num_cams
        frame_idx = img_idx // self.num_cams
        return unique_cam_idx, frame_idx

    def get_image(self, img_idx: int) -> Dict[str, Tensor]:
        """
        Get the rays for rendering the given image index.
        Args:
            img_idx: the image index.
        Returns:
            a dict containing the rays for rendering the given image index.
        """
        unique_cam_idx, frame_idx = self.parse_img_idx(img_idx)
        for cam_id in self.camera_list:
            if unique_cam_idx == self.camera_data[cam_id].unique_cam_idx:
                return self.camera_data[cam_id].get_image(frame_idx)

    @property
    def camera_list(self) -> List[int]:
        """
        Returns:
            the list of camera indices
        """
        return self.data_cfg.cameras

    @property
    def num_cams(self) -> int:
        """
        Returns:
            the number of cameras in the dataset
        """
        return len(self.data_cfg.cameras)

    @property
    def num_frames(self) -> int:
        """
        Returns:
            the number of frames in the dataset
        """
        return len(self._timesteps)

    @property
    def num_timesteps(self) -> int:
        """
        Returns:
            the number of image timesteps in the dataset
        """
        return len(self._timesteps)

    @property
    def num_imgs(self) -> int:
        """
        Returns:
            the number of images in the dataset
        """
        return self.num_cams * self.num_frames

    @property
    def timesteps(self) -> Tensor:
        """
        Returns:
            the integer timestep indices of all images,
            shape: (num_imgs,)
        Note:
            the difference between timestamps and timesteps is that
            timestamps are the actual timestamps (minus 1e9) of images
            while timesteps are the integer timestep indices of images.
        """
        return self._timesteps

    @property
    def normalized_time(self) -> Tensor:
        """
        Returns:
            the normalized timestamps of all images
            (normalized to the range [0, 1]),
            shape: (num_imgs,)
        """
        return self._normalized_time

    def register_normalized_timestamps(self) -> None:
        # normalized timestamps are between 0 and 1
        normalized_time = (self._timesteps - self._timesteps.min()) / (
                self._timesteps.max() - self._timesteps.min()
        )

        self._normalized_time = normalized_time.to(self.device)
        self._unique_normalized_timestamps = self._normalized_time.unique()

    def find_closest_timestep(self, normed_timestamp: float) -> int:
        """
        Find the closest timestep to the given timestamp.
        Args:
            normed_timestamp: the normalized timestamp to find the closest timestep for.
        Returns:
            the closest timestep to the given timestamp.
        """
        return torch.argmin(
            torch.abs(self._normalized_time - normed_timestamp)
        )

    def propose_training_image(
            self,
            candidate_indices: Tensor = None,
    ) -> Dict[str, Tensor]:
        if random.random() < self.buffer_ratio and self.image_error_buffered:
            # sample according to the image error buffer
            image_mean_error = self.image_error_buffer[candidate_indices]
            start_enhance_weight = self.data_cfg.sampler.get('start_enhance_weight', 1)
            if start_enhance_weight > 1:
                # increase the error of the first 10% frames
                frame_num = int(self.num_imgs / self.num_cams)
                error_weight = torch.cat((
                    torch.linspace(start_enhance_weight, 1, int(frame_num * 0.1)),
                    torch.ones(frame_num - int(frame_num * 0.1))
                ))
                error_weight = error_weight[..., None].repeat(1, self.num_cams).reshape(-1)
                error_weight = error_weight[candidate_indices].to(self.device)

                image_mean_error = image_mean_error * error_weight
            idx = torch.multinomial(
                image_mean_error, 1, replacement=False
            ).item()
            img_idx = candidate_indices[idx]
        else:
            # random sample one from candidate_indices
            img_idx = random.choice(candidate_indices)

        return img_idx

    def build_image_error_buffer(self) -> None:
        """
        Build the image error buffer.
        """
        if self.buffer_ratio > 0:
            for cam_id in self.camera_list:
                self.camera_data[cam_id].build_image_error_buffer()
        else:
            logger.info("Not building image error buffer because buffer_ratio <= 0.")

    def update_image_error_maps(self, render_results: Dict[str, Tensor]) -> None:
        """
        Update the image error buffer with the given render results for each camera.
        """
        # (img_num, )
        image_error_buffer = torch.zeros(self.num_imgs, device=self.device)
        image_cam_id = torch.from_numpy(np.stack(render_results["cam_ids"], axis=0))
        for cam_id in self.camera_list:
            cam_name = self.camera_data[cam_id].cam_name
            gt_rgbs, pred_rgbs = [], []
            Dynamic_opacities = []
            for img_idx, img_cam in enumerate(render_results["cam_names"]):
                if img_cam == cam_name:
                    gt_rgbs.append(render_results["gt_rgbs"][img_idx])
                    pred_rgbs.append(render_results["rgbs"][img_idx])
                    if "Dynamic_opacities" in render_results:
                        Dynamic_opacities.append(render_results["Dynamic_opacities"][img_idx])

            camera_results = {
                "gt_rgbs": gt_rgbs,
                "rgbs": pred_rgbs,
            }
            if len(Dynamic_opacities) > 0:
                camera_results["Dynamic_opacities"] = Dynamic_opacities
            self.camera_data[cam_id].update_image_error_maps(
                camera_results
            )

            # update the image error buffer
            image_error_buffer[image_cam_id == cam_id] = \
                self.camera_data[cam_id].image_error_maps.mean(dim=(1, 2))

        self.image_error_buffer = image_error_buffer
        self.image_error_buffered = True
        logger.info("Successfully updated image error buffer")

    def get_image_error_video(self, layout: Callable) -> List[np.ndarray]:
        """
        Get the image error buffer video.
        Returns:
            frames: the pixel sample weights video.
        """
        per_cam_video = {}
        for cam_id in self.camera_list:
            per_cam_video[cam_id] = self.camera_data[cam_id].get_image_error_video()

        all_error_images = []
        all_cam_names = []
        for frame_id in range(self.num_frames):
            for cam_id in self.camera_list:
                all_error_images.append(per_cam_video[cam_id][frame_id])
                all_cam_names.append(self.camera_data[cam_id].cam_name)

        merged_list = []
        for i in range(len(all_error_images) // self.num_cams):
            frames = all_error_images[
                     i
                     * self.num_cams: (i + 1)
                                      * self.num_cams
                     ]
            frames = [
                np.stack([frame, frame, frame], axis=-1) for frame in frames
            ]
            cam_names = all_cam_names[
                        i
                        * self.num_cams: (i + 1)
                                         * self.num_cams
                        ]
            tiled_img = layout(frames, cam_names)
            merged_list.append(tiled_img)

        merged_video = np.stack(merged_list, axis=0)
        merged_video -= merged_video.min()
        merged_video /= merged_video.max()
        merged_video = np.clip(merged_video * 255, 0, 255).astype(np.uint8)
        return merged_video

    @property
    def downscale_factor(self) -> float:
        """
        Returns:
            downscale_factor: the downscale factor of the images
        """
        return self._downscale_factor

    def update_downscale_factor(self, downscale: float) -> None:
        """
        Args:
            downscale: the new downscale factor
        Updates the downscale factor
        """
        self._old_downscale_factor.append(self._downscale_factor)
        self._downscale_factor = downscale
        for cam_id in self.camera_list:
            self.camera_data[cam_id].set_downscale_factor(self._downscale_factor)

    def reset_downscale_factor(self) -> None:
        """
        Resets the downscale factor to the original value
        """
        assert len(self._old_downscale_factor) > 0, "No downscale factor to reset to"
        self._downscale_factor = self._old_downscale_factor.pop()
        for cam_id in self.camera_list:
            self.camera_data[cam_id].set_downscale_factor(self._downscale_factor)

    @property
    def buffer_ratio(self) -> float:
        """
        Returns:
            buffer_ratio: the ratio of the rays sampled from the image error buffer
        """
        return self.data_cfg.sampler.buffer_ratio

    @property
    def buffer_downscale(self) -> float:
        """
        Returns:
            buffer_downscale: the downscale factor of the image error buffer
        """
        return self.data_cfg.sampler.buffer_downscale

    def prepare_novel_view_render_data(self, dataset_type: str, traj: torch.Tensor) -> list:
        """
        Prepare all necessary elements for novel view rendering.

        Args:
            dataset_type (str): Type of dataset
            traj (torch.Tensor): Novel view trajectory, shape (N, 4, 4)

        Returns:
            list: List of dicts, each containing elements required for rendering a single frame:
                - cam_infos: Camera information (extrinsics, intrinsics, image dimensions)
                - image_infos: Image-related information (indices, normalized time, viewdirs, etc.)
        """
        if dataset_type == "argoverse":
            cam_id = 1  # Use cam_id 1 for Argoverse dataset
        else:
            cam_id = 0  # Use cam_id 0 for other datasets_processor

        intrinsics = self.camera_data[cam_id].intrinsics[0]  # Assume intrinsics are constant across frames
        H, W = self.camera_data[cam_id].HEIGHT, self.camera_data[cam_id].WIDTH

        original_frame_count = self.num_frames
        scaled_indices = torch.linspace(0, original_frame_count - 1, len(traj))
        normed_time = torch.linspace(0, 1, len(traj))

        render_data = []
        for i in range(len(traj)):
            c2w = traj[i]

            # Generate ray origins and directions
            x, y = torch.meshgrid(torch.arange(W), torch.arange(H), indexing='xy')
            x, y = x.to(self.device), y.to(self.device)

            origins, viewdirs, direction_norm = get_rays(x.flatten(), y.flatten(), c2w, intrinsics)
            origins = origins.reshape(H, W, 3)
            viewdirs = viewdirs.reshape(H, W, 3)
            direction_norm = direction_norm.reshape(H, W, 1)

            cam_infos = {
                "camera_to_world": c2w,
                "intrinsics": intrinsics,
                "height": torch.tensor([H], dtype=torch.long, device=self.device),
                "width": torch.tensor([W], dtype=torch.long, device=self.device),
            }

            image_infos = {
                "origins": origins,
                "viewdirs": viewdirs,
                "direction_norm": direction_norm,
                "img_idx": torch.full((H, W), i, dtype=torch.long, device=self.device),
                "frame_idx": torch.full((H, W), scaled_indices[i].round().long(), device=self.device),
                "normed_time": torch.full((H, W), normed_time[i], dtype=torch.float32, device=self.device),
                "pixel_coords": torch.stack(
                    [y.float() / H, x.float() / W], dim=-1
                ),  # [H, W, 2]
            }

            render_data.append({
                "cam_infos": cam_infos,
                "image_infos": image_infos,
            })

        return render_data
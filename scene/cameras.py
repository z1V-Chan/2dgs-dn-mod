#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from typing import NamedTuple
import torch
from torch import nn
import numpy as np
from PIL import Image
from utils.general_utils import PILtoTorch
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
import os

class GroundTruth(NamedTuple):
    image: torch.Tensor
    alpha: torch.Tensor | None
    depth_cam: torch.Tensor | None
    depth_est: torch.Tensor | None
    depth_conf: torch.Tensor | None = None
    inpaint_mask: torch.Tensor | None = None
    inpaint_depth: torch.Tensor | None = None

class Camera(nn.Module):

    preload = False

    def __init__(
        self,
        colmap_id,
        R: np.ndarray,
        T: np.ndarray,
        FoVx,
        FoVy,
        resolution,
        image_path: str,
        depth_cam_path: str | None,
        depth_est_path: str | None,
        inpaint_mask_path: str | None,
        inpaint_depth_path: str | None,
        image_name,
        uid,
        trans=np.array([0.0, 0.0, 0.0]),
        scale=1.0,
        data_device="cuda",
    ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R # W2C.T
        self.T = T # W2C
        self.FoVx = FoVx
        self.FoVy = FoVy

        self.FoVx_old = FoVx
        self.FoVy_old = FoVy

        self.image_name = image_name

        self.resolution = resolution
        self.resolution_original = resolution
        # self.data_device = torch.device("cpu")

        self.__original_image = image_path
        # move to device at dataloader to reduce VRAM requirement
        self.__sensor_depth = depth_cam_path + ".png" if depth_cam_path is not None else None
        self.__pred_depth = depth_est_path + ".npz" if depth_est_path is not None else None
        self.__inpaint_mask = inpaint_mask_path + ".png" if inpaint_mask_path is not None else None
        # self.__inpaint_depth = inpaint_depth_path + ".png" if inpaint_depth_path is not None else None
        self.__inpaint_depth = inpaint_depth_path + ".pt" if inpaint_depth_path is not None else None
        
        # self.image_width = resolution[0]
        # self.image_height = resolution[1]

        self._gt = (
            load_image(
                resolution,
                image_path,
                self.__sensor_depth,
                self.__pred_depth,
                self.__inpaint_mask,
                self.__inpaint_depth,
            )
            if Camera.preload
            else None
        )

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = (
            torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        )
        self.projection_matrix = (
            getProjectionMatrix(
                znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy
            )
            .transpose(0, 1)
            .cuda()
        )
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    def gt(self, release=True):
        """
        Load ground truth data if not preloaded.
        If `release` is True, release the data after returning it.
        """
        if self._gt is None:
            original_image, gt_alpha_mask, sensor_depth, pred_depth = load_image(
                self.resolution_original,
                self.__original_image,
                self.__sensor_depth,
                self.__pred_depth,
            )
            gt = GroundTruth(
                original_image,
                gt_alpha_mask,
                sensor_depth,
                pred_depth,
            )
        else: 
            gt = self._gt

        if release:
            self._gt = None
        else:
            self._gt = gt

        return gt

class MiniCam:
    def __init__(
        self,
        width,
        height,
        fovy,
        fovx,
        znear,
        zfar,
        world_view_transform,
        full_proj_transform,
    ):
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]


def load_image(
    resolution,
    image_path: str,
    depth_cam_path: str | None = None,
    depth_est_path: str | None = None,
    inpaint_mask_path: str | None = None,
    inpaint_depth_path: str | None = None,
):
    """
    Load the image and depth maps from the FreeImage objects
    RAM usage by PIL would be free
    return Tensor are all in device cpu
    """
    image_pil = Image.open(image_path)
    depth_cam_pil = Image.open(depth_cam_path) if depth_cam_path is not None else None
    # depth_est_pil = Image.open(depth_est_path) if depth_est_path is not None else None
    depth_est_dict: dict[str, np.ndarray] = np.load(depth_est_path) if depth_est_path is not None else None
    depth_est_np = depth_est_dict["depth"].astype(np.float32) if depth_est_dict is not None else None
    depth_est_conf = depth_est_dict["conf"].astype(np.float32) if depth_est_dict is not None and "conf" in depth_est_dict else None

    # add for inpainting
    if inpaint_mask_path is not None:
        inpaint_mask_pil = Image.open(inpaint_mask_path) 
        # inpainted_depth_pil = Image.open(inpaint_depth_path).convert('L') 
        # inpainted_depth_name = inpaint_depth_path.split('/')[-1] 
        # inpainted_depth_dir = os.path.dirname(inpaint_depth_path)
        # depth_range_name = inpainted_depth_name.replace('.png', '.pt')
        
        # d_min, d_max = torch.load(os.path.join(inpainted_depth_dir, depth_range_name), weights_only=False)
        
        inpaint_mask_tensor = PILtoTorch(inpaint_mask_pil, resolution)
        if not os.path.exists(inpaint_depth_path):
            inpaint_depth_tensor = None
        else:
            inpaint_depth_tensor = torch.load(inpaint_depth_path, weights_only=False)
        
        # inpaint_depth_tensor = PILtoTorch(inpainted_depth_pil, resolution)
        # inpaint_depth_tensor = inpaint_depth_tensor * (d_max - d_min) + d_min
    else:
        inpaint_mask_tensor = None
        inpaint_depth_tensor = None  

    if len(image_pil.split()) > 3:
        # assert False, "Image has more than 3 channels, not supported"

        resized_image_rgb = torch.cat([PILtoTorch(im, resolution) for im in image_pil.split()[:3]], dim=0)
        loaded_mask = PILtoTorch(image_pil.split()[3], resolution)
        gt_image = resized_image_rgb
    else:
        resized_image_rgb = PILtoTorch(image_pil, resolution)
        loaded_mask = None
        gt_image = resized_image_rgb

    resized_depth_cam = PILtoTorch(depth_cam_pil, resolution, scale=1e3) if depth_cam_pil is not None else None

    # resized_depth_est = PILtoTorch(depth_est_pil, resolution, scale=1e3) if depth_est_pil is not None else None
    depth_est_tensor = torch.tensor(depth_est_np, dtype=torch.float32, device="cpu").unsqueeze(0) if depth_est_np is not None else None
    depth_est_conf_tensor = torch.tensor(depth_est_conf, dtype=torch.float32, device="cpu").unsqueeze(0) if depth_est_conf is not None else None
    # resized_depth_est = torch.nn.functional.interpolate(
    #     resized_depth_est.unsqueeze(0),
    #     size=(resolution[1], resolution[0]),
    #     mode="bicubic",
    #     align_corners=True,
    # ).squeeze(0) if depth_est_np is not None else None

    image_pil.close()
    if depth_cam_pil is not None:
        depth_cam_pil.close()
    # if depth_est_pil is not None:
    #     depth_est_pil.close()

    return GroundTruth(
        gt_image.clamp(0.0, 1.0),
        loaded_mask,
        resized_depth_cam,
        depth_est_tensor,
        depth_est_conf_tensor,
        inpaint_mask_tensor,
        inpaint_depth_tensor,
    )

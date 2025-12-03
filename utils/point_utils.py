import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os, cv2
import matplotlib.pyplot as plt
import math

from scene.cameras import Camera

def depths_to_points(view: Camera, depthmap: torch.Tensor):
    h, w = depthmap.shape[1:]

    c2w = (view.world_view_transform.T).inverse()
    W, H = view.resolution
    ndc2pix = torch.tensor([
        [W / 2, 0, 0, (W) / 2],
        [0, H / 2, 0, (H) / 2],
        [0, 0, 0, 1]]).float().cuda().T
    projection_matrix = c2w.T @ view.full_proj_transform
    intrins = (projection_matrix @ ndc2pix)[:3,:3].T

    if w != W or h != H:
        fx, fy, cx, cy = intrins[0,0], intrins[1,1], intrins[0,2], intrins[1,2]
        scale_x = w / W
        scale_y = h / H
        fx_new = fx * scale_x
        fy_new = fy * scale_y
        cx_new = cx * scale_x
        cy_new = cy * scale_y
        intrins = torch.tensor([
            [fx_new, 0, cx_new],
            [0, fy_new, cy_new],
            [0, 0, 1]
        ]).float().cuda()

    grid_x, grid_y = torch.meshgrid(
        torch.arange(w, device="cuda").float(),
        torch.arange(h, device="cuda").float(),
        indexing="xy",
    )
    points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3) # [h*w, 3]
    rays_d = points @ intrins.inverse().T @ c2w[:3,:3].T
    rays_o = c2w[:3,3]
    points = depthmap.reshape(-1, 1) * rays_d + rays_o
    return points

def depth_to_normal(view: Camera, depth: torch.Tensor):
    """
        view: view camera
        depth: depthmap [1, H, W]
    """
    points = depths_to_points(view, depth).reshape(*depth.shape[1:], 3)
    output = torch.zeros_like(points)
    dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
    dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
    normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    output[1:-1, 1:-1, :] = normal_map
    return output

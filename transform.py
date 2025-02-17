import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scene.colmap_loader import qvec2rotmat
from utils.graphics_utils import getProjectionMatrix
import math
import torch
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render
import torchvision
from argparse import ArgumentParser
from arguments import PipelineParams


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret

def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))

class VirtualCamera(nn.Module):
    def __init__(
        self,
        R=np.array(
            [
                [0, 0, -1],
                [0, 1, 0],
                [1, 0, 0],
            ]
        ),
        T=np.array(
            [2, 0, 0],
            dtype=np.float32,
        ),
        FoVy=math.radians(30),
        image_height=1024,
        image_width=1024,
    ):
        self.R: np.array = R
        self.T: np.array = T
        self.FoVy: float = FoVy
        self.image_height: int = image_height
        self.image_width: int = image_width
        self.update_attributes()

    def update_attributes(self):

        self.FoVx = compute_fovX(
            fovY=self.FoVy,
            height=self.image_height,
            width=self.image_width,
        )

        self.zfar = 100.0
        self.znear = 0.01

        self.world_view_transform = getViewMatrix(self.R, self.T)
        self.projection_matrix = getProjectionMatrix(
            znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy
        )
        self.full_proj_transform = self.projection_matrix @ self.world_view_transform

        self.world_view_transform = self.world_view_transform.transpose(0, 1).cuda()
        self.projection_matrix = self.projection_matrix.transpose(0, 1).cuda()
        self.full_proj_transform = self.full_proj_transform.transpose(0, 1).cuda()

        self.camera_center = torch.tensor(self.T, dtype=torch.float32).cuda()

        return

    def setLookAt(self, cam_pos, target_pos):
        cam_pos = np.array(cam_pos, dtype=np.float32)
        target_pos = np.array(target_pos, dtype=np.float32)
        y_approx = np.array([0, 1, 0], dtype=np.float32)
        self.R = LookAt(cam_pos, target_position=target_pos, y_approx=y_approx)
        self.T = cam_pos
        self.update_attributes()
        return


def compute_fovX(fovY, height, width):
    w_h_ratio = float(width) / float(height)

    return math.atan(math.tan(fovY * 0.5) * w_h_ratio) * 2


def getViewMatrix(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.T
    Rt[:3, 3] = -R.T @ t
    Rt[3, 3] = 1.0

    return torch.tensor(Rt, dtype=torch.float32)


def LookAt(camera_position, target_position, y_approx):

    look_dir = target_position - camera_position
    z_axis = look_dir
    z_axis /= np.linalg.norm(z_axis)  # 归一化

    x_axis = np.cross(y_approx, z_axis)
    x_axis /= np.linalg.norm(x_axis)  # 归一化

    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)  # 归一化

    R = np.zeros((3, 3))
    R[:, 0] = x_axis
    R[:, 1] = y_axis
    R[:, 2] = z_axis

    return R


@torch.no_grad()
def main():
    # Set up command line argument parser
    parser = ArgumentParser(description="gaussian splatting")
    pipe = PipelineParams(parser)
    parser.add_argument(
        "-p","--ply",
        type=str,
        required=True,
        help="Path to the input PLY file",
    )
    parser.add_argument(
        "-o","--output",
        type=str,
        required=True,
        help="Path to the output PLY file",
    )
    parser.add_argument("-r", "--rotation_qvec", type=str, required=True,
                        help="Rotation quaternion [w, x, y, z] to apply to the model")
    parser.add_argument("-t", "--translation", type=str, required=True,
                        help="Translation vector [x, y, z] to apply to the model")
    args = parser.parse_args()

    gaussians = GaussianModel(3)
    gaussians.load_ply(args.ply)

    # Compute the rotation matrix
    r_qvec = np.array([float(value) for value in args.rotation_qvec.split()], dtype=float)
    t_global = np.array([[float(value) for value in args.translation.split()]], dtype=float)
    R_global = qvec2rotmat(r_qvec)

    R = torch.tensor(R_global, dtype=torch.float32, device="cuda")
    t = torch.tensor(t_global, dtype=torch.float32, device="cuda")

    # Rotate the gaussians
    pcd_cor = gaussians._xyz
    gaussians._xyz = pcd_cor @ R.T + t
    _rotation = quaternion_to_matrix(gaussians._rotation)
    _rotation = torch.einsum("ij,njk->nik", R, _rotation)
    gaussians._rotation = matrix_to_quaternion(_rotation)

    cam = VirtualCamera()
    cam.setLookAt(cam_pos=[0, 0.8, -2], target_pos=[0, 0, 0])

    bg = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    rendering = render(cam, gaussians, pipe, bg)["render"]
    torchvision.utils.save_image(rendering, f"./img.png")

    gaussians.save_ply(args.output)


if __name__ == "__main__":
    main()

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

import os
import torch
from random import randint
from utils.loss_utils import isotropic_loss, l1_loss, ssim, l1_loss_mask
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import PILtoTorch, TorchToPIL, get_expon_lr_func, safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import depth_normalize_, psnr, render_net_image
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.point_utils import depth_to_normal
from torchvision.utils import save_image
import numpy as np
import lpips
import cv2
import math

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
    

def crop_using_bbox(image, bbox):
    xmin, ymin, xmax, ymax = bbox
    return image[:, ymin:ymax+1, xmin:xmax+1]

def mask_to_bbox(mask):
    # Find the rows and columns where the mask is non-zero
    rows = torch.any(mask, dim=1)
    cols = torch.any(mask, dim=0)
    ymin, ymax = torch.where(rows)[0][[0, -1]]
    xmin, xmax = torch.where(cols)[0][[0, -1]]
    return xmin, ymin, xmax, ymax

def divide_into_patches(image, K):
    B, C, H, W = image.shape
    patch_h, patch_w = H // K, W // K
    patches = torch.nn.functional.unfold(image, (patch_h, patch_w), stride=(patch_h, patch_w))
    patches = patches.view(B, C, patch_h, patch_w, -1)
    return patches.permute(0, 4, 1, 2, 3)

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal)) 

def compute_lpips_multi_mask_contours(image, gt_image, inpaint_mask, LPIPS, min_area=100, K=2):
    """
    对mask中每个独立轮廓区域计算LPIPS并累加。
    """
    if inpaint_mask.ndim == 3:
        inpaint_mask = inpaint_mask.squeeze(0)

    mask_np = inpaint_mask.detach().cpu().numpy().astype(np.uint8)
    
    # 找轮廓，cv2.RETR_EXTERNAL 表示只取最外层轮廓
    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    total_lpips = 0.0
    valid_regions = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 2000:
            continue  # 过滤噪声区域
        
        x, y, w, h = cv2.boundingRect(cnt)
        bbox = (x, y, x+w, y+h)
        if w < 32 or h < 32:
            continue  # 过滤过小区域

        cropped_image = crop_using_bbox(image, bbox)
        cropped_gt_image = crop_using_bbox(gt_image, bbox)

        rendering_patches = divide_into_patches(cropped_image[None, ...], K)
        gt_patches = divide_into_patches(cropped_gt_image[None, ...], K)

        lpips_loss = LPIPS(rendering_patches.squeeze()*2-1,
                           gt_patches.squeeze()*2-1).mean()

        total_lpips += lpips_loss
        valid_regions += 1

    if valid_regions == 0:
        return torch.tensor(0.0, device=image.device)
    return total_lpips / valid_regions

def save_ply(points, filename):
    """
    将 (N,6) numpy 数组保存为带颜色的 PLY 点云文件
    前三维为 xyz，后三维为 rgb（范围 [0,1]）
    """
    assert points.ndim == 2 and points.shape[1] == 6, "points 必须是 (N,6) 形状"

    # 拆分 xyz 和 rgb
    xyz = points[:, :3]
    rgb = points[:, 3:]

    # rgb 转成 [0,255] uint8
    rgb_uint8 = np.clip(rgb * 255, 0, 255).astype(np.uint8)

    # 拼成结构化数组，符合 PLY 格式
    vertex = np.empty(points.shape[0],
                      dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    vertex['x'] = xyz[:, 0]
    vertex['y'] = xyz[:, 1]
    vertex['z'] = xyz[:, 2]
    vertex['red'] = rgb_uint8[:, 0]
    vertex['green'] = rgb_uint8[:, 1]
    vertex['blue'] = rgb_uint8[:, 2]

    # 写入 PLY 文件
    with open(filename, 'wb') as f:
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {len(vertex)}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property uchar red\n"
            "property uchar green\n"
            "property uchar blue\n"
            "end_header\n"
        )
        f.write(bytearray(header, 'ascii'))
        vertex.tofile(f)

    print(f"✅ 已保存点云到: {filename}  (共 {len(points)} 个点)")
    
def save_ply_xyz(points, filename):
    """
    将 (N,3) numpy 数组保存为不带颜色的 PLY 点云文件
    points: 前三维为 xyz
    """
    assert points.ndim == 2 and points.shape[1] >= 3, "points 必须至少是 (N,3) 形状"

    # 取 xyz
    xyz = points[:, :3].astype(np.float32)

    # 结构化数组：只有 xyz
    vertex = np.empty(
        xyz.shape[0],
        dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
    )
    vertex['x'] = xyz[:, 0]
    vertex['y'] = xyz[:, 1]
    vertex['z'] = xyz[:, 2]

    # 写入 PLY 文件
    with open(filename, 'wb') as f:
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {len(vertex)}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "end_header\n"
        )
        f.write(bytearray(header, 'ascii'))
        vertex.tofile(f)

    print(f"✅ 已保存点云到: {filename}  (共 {len(vertex)} 个点)")

def get_init_points(cam, default_depth=False, custom_mask=None, gt_image=None):
    """
    从深度图与mask中生成初始点云 (N,6): xyz + rgb
    """
    # ---- 1. 提取输入 ----
    depth = default_depth[0].cpu().numpy()  # (H, W)
    mask2d = custom_mask[0].cpu().numpy().astype(bool)  # (H, W)
    rgb_image = gt_image.cpu().numpy().transpose(1, 2, 0)  # (H, W, 3)，范围(0,1)

    # ---- 2. 获取相机内参 ----
    R, T, FovY, FovX = cam.R, cam.T, cam.FoVy, cam.FoVx
    width, height = cam.resolution
    f_x = fov2focal(FovX, width)
    f_y = fov2focal(FovY, height)
    c_x = width / 2.0
    c_y = height / 2.0

    # 内参矩阵
    A = np.array([
        [f_x, 0, c_x],
        [0, f_y, c_y],
        [0,   0, 1.0]
    ])

    # ---- 3. 构造相机外参 (w2c) ----
    w2c_extrinsic = np.eye(4)
    w2c_extrinsic[:3, :3] = R.T
    w2c_extrinsic[:3, 3] = T

    # ---- 4. 取出mask区域像素 ----
    ys, xs = np.where(mask2d)
    zs = depth[ys, xs]

    # ---- 5. 像素坐标 -> 相机坐标 (逆内参投影) ----
    x_cam = (xs - c_x) * zs / f_x
    y_cam = (ys - c_y) * zs / f_y
    pts_cam = np.stack([x_cam, y_cam, zs], axis=-1)  # (N, 3)

    # ---- 6. 相机坐标 -> 世界坐标 ----
    # c2w = inverse(w2c)
    c2w = np.linalg.inv(w2c_extrinsic)
    pts_homo = np.concatenate([pts_cam, np.ones((pts_cam.shape[0], 1))], axis=1)
    pts_world = (c2w @ pts_homo.T).T[:, :3]  # (N, 3)

    # ---- 7. 提取对应RGB ----
    colors = rgb_image[ys, xs, :]  # (N, 3)

    # ---- 8. 拼接 (xyz + rgb) ----
    points_with_color = np.concatenate([pts_world, colors], axis=-1)  # (N, 6)

    return points_with_color

def project_ref_pcd(ref_imgs, viewpoints):
    pcds = []
    for i, view in enumerate(viewpoints):
        if view.image_name in ref_imgs:
            gt = view.gt(release=False)
            gt_image = gt.image.to(device="cuda", non_blocking=True)
            inpaint_mask = gt.inpaint_mask.to(device="cuda", non_blocking=True).bool()
            inpaint_depth = gt.inpaint_depth.to(device="cuda", non_blocking=True).unsqueeze(0)
            
            points_3d = get_init_points(view, default_depth=inpaint_depth, custom_mask=inpaint_mask, gt_image=gt_image)
            pcds.append(points_3d)
    return np.concatenate(pcds, axis=0)


def training(
    dataset: ModelParams,
    opt: OptimizationParams,
    pipe: PipelineParams,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    gs_path,
    remove_mask = None,
    ref_imgs = ["00064"]
):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, crop_gs_path=gs_path, load_iteration=30000)
    # gaussians.training_setup(opt)
    
    # load crop 3d mask and initialize new gaussian points
    if gs_path is None:
        gs_path = scene.gaussian_path
        
    if remove_mask is None:
        dir = os.path.dirname(gs_path)
        remove_mask = os.path.join(dir, "remove_3d_mask.npy")

    if os.path.exists(remove_mask):
        remove_mask_np = np.load(remove_mask)
    
    training_viewpoint = scene.getTrainCameras().copy()
    
    # inpaint_pcds = project_ref_pcd(ref_imgs, training_viewpoint)
    inpaint_pcds = None

    gaussians.inpaint_setup(opt, remove_mask_np, inpaint_pcds)
    
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_depth_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    debug_dir = "debug"   
    os.makedirs(debug_dir, exist_ok=True)
    LPIPS = lpips.LPIPS(net='vgg')
    for param in LPIPS.parameters():
        param.requires_grad = False
    LPIPS.cuda()
    # debug_new_xyz = []
    freeze_gs_num = remove_mask_np[remove_mask_np==False].shape[0]
    gaussians.stop_grad(opt, freeze_gs_num)
    for iteration in range(first_iter, opt.iterations + 1):        

        iter_start.record()

        # gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()
        
        # Pick a random Camera
        # if not viewpoint_stack:
        viewpoint_stack = scene.getTrainCameras().copy()

        bg = torch.rand((3), device="cuda") if opt.random_background else background
        # viewpoint_cam = viewpoint_stack.pop(0)
        while True:
  
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

            gt = viewpoint_cam.gt(release=False)
            if gt.inpaint_mask.max() > 0.0:
                break

            if len(viewpoint_stack) == 0:
                raise RuntimeError("❌ 所有视角的 inpaint_mask 都为空，无法找到有效样本！")
            

        gt_image = gt.image.to(device="cuda", non_blocking=True)
        inpaint_mask = gt.inpaint_mask.to(device="cuda", non_blocking=True).bool()
        # inpaint_depth = gt.inpaint_depth.to(device="cuda", non_blocking=True)

        drop_rate = opt.drop_rate * (iteration/10000) if iteration > opt.drop_from_iter else 0.0

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, drop_rate=drop_rate)
        viewspace_point_tensor, visibility_filter, radii = render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        image: torch.Tensor = render_pkg["render"] # [3, H, W]
        # rend_depth: torch.Tensor = render_pkg["render_depth"] # [1, H, W]
        rend_normal: torch.Tensor = render_pkg['render_normal'] # [3, H, W]
        surf_normal: torch.Tensor = render_pkg['surf_normal'] # [3, H, W]
        rend_dist: torch.Tensor = render_pkg["render_dist"] # [1, H, W]

        
        rend_normal = rend_normal * inpaint_mask.float()
        surf_normal = surf_normal * inpaint_mask.float()
        
        lambda_normal = opt.lambda_normal if iteration > 700 else 0.0
        lambda_dist = 100 if iteration > 300 else 0.0
        
        
        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        normal_loss = lambda_normal * (normal_error).mean()
        dist_loss = lambda_dist * (rend_dist).mean()
        ssim_loss = 1 - ssim(image, gt_image)
        # ssim_loss = 1 - ssim(image * inpaint_mask.float(), gt_image * inpaint_mask.float())
        # Ll1 = mask_l1_loss(image, gt_image, inpaint_mask)
        Ll1 = l1_loss(image, gt_image)
        # Ll1 = l1_loss(image, gt_image)
        # total_loss = Ll1  + normal_loss + dist_loss
        # total_loss = Ll1  + dist_loss
        total_loss = 0.8 * Ll1 + 0.2 * ssim_loss
        total_loss.backward()
        xyz = gaussians.get_xyz
        # print("frozen grad sum:", xyz.grad[:freeze_gs_num].abs().sum())
        # print("trainable grad sum:", xyz.grad[freeze_gs_num:].abs().sum())

        if iteration % 10 == 0:
            loss_dict = {
                "Loss": f"{Ll1:.{4}f}",
                "dist": f"{dist_loss:.{4}f}",
                "ssim": f"{ssim_loss:.{4}f}",
                
            }
            progress_bar.set_postfix(loss_dict)
            progress_bar.update(10)
        
        with torch.no_grad():
            if iteration < 15000:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if  iteration % 500 == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, freeze_num=freeze_gs_num)
                    # gaussians.reset_opacity()
                    
        gaussians.zero_frozen_grads(freeze_gs_num)
        gaussians.optimizer.step()
        gaussians.optimizer.zero_grad(set_to_none = True)
        if iteration in saving_iterations:
            print("\n[ITER {}] Saving Gaussians".format(iteration))         
            scene.save_inpaint(iteration)
    # merge_new_xyz = np.concatenate(debug_new_xyz, axis=0)   

    print("\n[ITER {}] Saving Inpaint Gaussians".format(opt.iterations + 1))         
    # scene.save_inpaint(opt.iterations + 1)           
                
     


def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

@torch.no_grad()
def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/reg_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0).to("cuda")
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        from utils.general_utils import colormap
                        depth = render_pkg["surf_depth"]
                        norm = depth.max()
                        depth = depth / norm
                        depth = colormap(depth.cpu().numpy()[0], cmap='turbo')
                        tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), depth[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)

                        try:
                            rend_alpha = render_pkg['rend_alpha']
                            rend_normal = render_pkg["rend_normal"] * 0.5 + 0.5
                            surf_normal = render_pkg["surf_normal"] * 0.5 + 0.5
                            tb_writer.add_images(config['name'] + "_view_{}/rend_normal".format(viewpoint.image_name), rend_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/surf_normal".format(viewpoint.image_name), surf_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/rend_alpha".format(viewpoint.image_name), rend_alpha[None], global_step=iteration)

                            rend_dist = render_pkg["rend_dist"]
                            rend_dist = colormap(rend_dist.cpu().numpy()[0])
                            tb_writer.add_images(config['name'] + "_view_{}/rend_dist".format(viewpoint.image_name), rend_dist[None], global_step=iteration)
                        except:
                            pass

                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6008)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[5000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--gs_path", type=str, default = None)
    parser.add_argument("--remove_mask", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    

    # Initialize system state (RNG)
    safe_state(args.quiet)

    print("Optimizing " + args.model_path)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, 
             args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, 
             gs_path=args.gs_path, remove_mask=args.remove_mask)

    # All done
    print("\nTraining complete.")

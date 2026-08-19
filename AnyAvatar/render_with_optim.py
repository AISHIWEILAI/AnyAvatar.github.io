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

import torch
from torch.utils.data import DataLoader
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
import concurrent.futures
import multiprocessing
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import numpy as np
import torchvision
from time import perf_counter
import json
import random
import subprocess
from lpipsPyTorch import lpips

from gaussian_renderer import render
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, FlameGaussianModel
from mesh_renderer import NVDiffRenderer
from utils.pose_utils import get_tensor_from_camera
from utils.loss_utils import l1_loss, ssim
from utils.image_utils import psnr


mesh_renderer = NVDiffRenderer()

def write_data(path2data):
    for path, data in path2data.items():
        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)

        if path.suffix in [".png", ".jpg"]:
            data = data.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
            Image.fromarray(data).save(path)
        elif path.suffix in [".obj"]:
            with open(path, "w") as f:
                f.write(data)
        elif path.suffix in [".txt"]:
            with open(path, "w") as f:
                f.write(data)
        elif path.suffix in [".npz"]:
            np.savez(path, **data)
        else:
            raise NotImplementedError(f"Unknown file type: {path.suffix}")

def render_set(dataset : ModelParams, name, iteration, views, gaussians, pipeline, background, render_mesh):
    if dataset.select_camera_id != -1:
        name = f"{name}_{dataset.select_camera_id}"
    iter_path = Path(dataset.model_path) / name / f"ours_{iteration}"
    render_path = iter_path / "renders"
    gts_path = iter_path / "gt"
    if render_mesh:
        render_mesh_path = iter_path / "renders_mesh"

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    views_loader = DataLoader(views, batch_size=None, shuffle=False, num_workers=8)
    max_threads = multiprocessing.cpu_count()
    print('Max threads: ', max_threads)
    worker_args = []
    psnr_list = []
    ssim_list = []
    lpips_list = []
    for idx, view in enumerate(tqdm(views_loader, desc="Rendering progress")):
        if gaussians.binding != None:
            gaussians.select_mesh_by_timestep(view.timestep)
        
        # Get optimized camera pose if available
        camera_pose = None
        if gaussians.P is not None:
            pose = gaussians.get_RT(camera=view)
            if pose is not None:
                camera_pose = pose
        
        rendering = render(view, gaussians, pipeline, background, camera_pose=camera_pose)["render"]
        gt = view.original_image[0:3, :, :]
        # PSNR/SSIM: input (1, 3, H, W)
        r = rendering.unsqueeze(0)
        g = gt.unsqueeze(0).to(r.device)
        psnr_list.append(psnr(r, g).mean().item())
        ssim_list.append(ssim(r, g).item())
        lpips_list.append(lpips(r, g, net_type='vgg').item())
        if render_mesh:
            out_dict = mesh_renderer.render_from_camera(gaussians.verts, gaussians.faces, view)
            rgba_mesh = out_dict['rgba'].squeeze(0).permute(2, 0, 1)  # (C, W, H)
            rgb_mesh = rgba_mesh[:3, :, :]
            alpha_mesh = rgba_mesh[3:, :, :]
            mesh_opacity = 0.5
            rendering_mesh = rgb_mesh * alpha_mesh * mesh_opacity  + gt.to(rgb_mesh) * (alpha_mesh * (1 - mesh_opacity) + (1 - alpha_mesh))

        path2data = {}
        path2data[Path(render_path) / f'{idx:05d}.png'] = rendering
        path2data[Path(gts_path) / f'{idx:05d}.png'] = gt
        if render_mesh:
            path2data[Path(render_mesh_path) / f'{idx:05d}.png'] = rendering_mesh
        worker_args.append([path2data])

        if len(worker_args) == max_threads or idx == len(views_loader)-1:
            with concurrent.futures.ThreadPoolExecutor(max_threads) as executor:
                futures = [executor.submit(write_data, *args) for args in worker_args]
                concurrent.futures.wait(futures)
            worker_args = []
    
    if psnr_list:
        mean_psnr = sum(psnr_list) / len(psnr_list)
        mean_ssim = sum(ssim_list) / len(ssim_list)
        mean_lpips = sum(lpips_list) / len(lpips_list)
        print(f"[{name}] PSNR: {mean_psnr:.4f}  SSIM: {mean_ssim:.4f}  LPIPS: {mean_lpips:.4f}  (n={len(psnr_list)})")
        results_file = iter_path / "render_results.txt"
        with open(results_file, "w") as f:
            f.write(f"PSNR: {mean_psnr:.4f}\n")
            f.write(f"SSIM: {mean_ssim:.4f}\n")
            f.write(f"LPIPS: {mean_lpips:.4f}\n")
            f.write(f"num_views: {len(psnr_list)}\n")
        print(f"Metrics saved to {results_file}")
    
    try:
        os.system(f"ffmpeg -y -framerate 25 -f image2 -pattern_type glob -i '{render_path}/*.png' -pix_fmt yuv420p {iter_path}/renders.mp4")
        os.system(f"ffmpeg -y -framerate 25 -f image2 -pattern_type glob -i '{gts_path}/*.png' -pix_fmt yuv420p {iter_path}/gt.mp4")
        # high quality: -q:v 0 -q:a 0, image pattern %05d.png
        subprocess.run([
            'ffmpeg', '-y', '-r', '25', '-f', 'image2', '-i', f'{render_path}/%05d.png',
            '-pix_fmt', 'yuv420p', '-q:v', '0', '-q:a', '0', f'{iter_path}/high_renders.mp4'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        subprocess.run([
            'ffmpeg', '-y', '-r', '25', '-f', 'image2', '-i', f'{gts_path}/%05d.png',
            '-pix_fmt', 'yuv420p', '-q:v', '0', '-q:a', '0', f'{iter_path}/high_gt.mp4'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if render_mesh:
            os.system(f"ffmpeg -y -framerate 25 -f image2 -pattern_type glob -i '{render_mesh_path}/*.png' -pix_fmt yuv420p {iter_path}/renders_mesh.mp4")
    except Exception as e:
        print(e)
        
def render_set_optimize(dataset, name, iteration, views, gaussians, pipeline, background, optim_test_pose_iter=500, test_fps=False, scene=None):
    if dataset.select_camera_id != -1:
        name = f"{name}_{dataset.select_camera_id}"
    iter_path = Path(dataset.model_path) / name / f"ours_{iteration}"
    render_path = iter_path / "renders"
    gts_path = iter_path / "gt"

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    gaussians._xyz.requires_grad_(False)
    gaussians._features_dc.requires_grad_(False)
    gaussians._features_rest.requires_grad_(False)
    gaussians._opacity.requires_grad_(False)
    gaussians._scaling.requires_grad_(False)
    gaussians._rotation.requires_grad_(False)

    views_by_camera_id = {}
    views_list = list(views) if hasattr(views, '__iter__') else [v for v in views]
    
    for view in views_list:
        spatial_id = view.camera_id if view.camera_id is not None else view.colmap_id
        if spatial_id is not None:
            if spatial_id not in views_by_camera_id:
                views_by_camera_id[spatial_id] = []
            views_by_camera_id[spatial_id].append(view)
    
    print(f"\nFound {len(views_by_camera_id)} unique camera IDs in test views:")
    for camera_id, view_list in views_by_camera_id.items():
        print(f"  camera_id={camera_id}: {len(view_list)} views")
    
    optimized_poses = {}
    render_idx = 0
    
    for camera_id, view_list in views_by_camera_id.items():
        print(f"\nOptimizing pose for camera_id={camera_id} with {len(view_list)} views...")
        
        if camera_id not in gaussians.cameraid_to_pose_idx:
            test_cameras = scene.getTestCameras()
            test_cameras_list = list(test_cameras) if hasattr(test_cameras, '__iter__') else [v for v in test_cameras]
            
            test_view = None
            for test_view_candidate in test_cameras_list:
                test_spatial_id = test_view_candidate.camera_id if test_view_candidate.camera_id is not None else test_view_candidate.colmap_id
                if test_spatial_id == camera_id:
                    test_view = test_view_candidate
                    break
            
            initial_pose = get_tensor_from_camera(test_view.world_view_transform.transpose(0, 1))
            new_pose = initial_pose.unsqueeze(0).to(gaussians.P.device)
            gaussians.P = torch.cat([gaussians.P, new_pose], dim=0)
            pose_idx = gaussians.P.shape[0] - 1
            gaussians.cameraid_to_pose_idx[camera_id] = pose_idx
            print(f"Added new pose for camera_id={camera_id} from test cameras, pose_idx={pose_idx}")
        else:
            pose_idx = gaussians.cameraid_to_pose_idx[camera_id]
        
        optimizable_pose = gaussians.P[pose_idx].clone().detach().requires_grad_(True)
        
        pose_optimizer = torch.optim.Adam([
            {"params": [optimizable_pose], "lr": 0.001}
        ],
        betas=(0.9, 0.999),
        weight_decay=1e-4
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(pose_optimizer, T_max=optim_test_pose_iter, eta_min=0.0001)
        
        with tqdm(total=optim_test_pose_iter, desc=f"Optimizing camera_id={camera_id}", leave=True) as progress_bar:
            best_pose = None
            best_loss = float('inf')

            for opt_iter in range(optim_test_pose_iter):
                view = random.choice(view_list)
                
                if gaussians.binding != None:
                    gaussians.select_mesh_by_timestep(view.timestep)
                camera_pose = optimizable_pose
                
                rendering = render(view, gaussians, pipeline, background, camera_pose=camera_pose)["render"]
                gt = view.original_image[0:3, :, :]
                if gt.device.type != 'cuda':
                    gt = gt.cuda()
                
                black_hole_threshold = 0.0
                mask = (rendering > black_hole_threshold).float()
                loss = (l1_loss(rendering, gt) * mask).mean()
                
                loss.backward()
                
                with torch.no_grad():
                    pose_optimizer.step()
                    pose_optimizer.zero_grad(set_to_none=True)

                    if loss.item() < best_loss:
                        best_loss = loss.item()
                        best_pose = optimizable_pose.clone().detach()

                    progress_bar.update(1)
                    progress_bar.set_postfix(loss=loss.item(), best_loss=best_loss)
                
                scheduler.step()

            if best_pose is not None:
                optimal_pose = best_pose
            else:
                optimal_pose = optimizable_pose.detach()
        
        with torch.no_grad():
            optimal_pose_detached = optimal_pose.detach().to(gaussians.P.device)
            gaussians.P.data[pose_idx] = optimal_pose_detached
        optimized_poses[pose_idx] = optimal_pose_detached.clone()
        print(f"Updated pose for camera_id={camera_id}, pose_idx={pose_idx}, best_loss={best_loss:.6f}")
        
    return optimized_poses

def load_checkpoint_and_optimize_nvs_poses(dataset, iteration, pipeline, optim_test_pose_iter=500):
    if dataset.bind_to_mesh:
        gaussians = FlameGaussianModel(dataset.sh_degree)
    else:
        gaussians = GaussianModel(dataset.sh_degree)
    
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    
    checkpoint_path = os.path.join(dataset.model_path, f"chkpnt{scene.loaded_iter}.pth")
    if not os.path.exists(checkpoint_path):
        print(f"Warning: Checkpoint not found at {checkpoint_path}, skipping pose optimization")
        return scene, gaussians
    
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint_data = torch.load(checkpoint_path, map_location='cuda')
    
    from arguments import OptimizationParams
    temp_parser = ArgumentParser()
    opt = OptimizationParams(temp_parser)
    opt.optim_pose = getattr(dataset, 'optim_pose', False)
    model_params = checkpoint_data['model']
    
    gaussians.restore(model_params, opt)
    
    if gaussians.P is None:
        print("Warning: gaussians.P is None after restore, initializing from cameras")
        all_cameras = list(scene.train_cameras[1.0])
        test_cameras_list = scene.test_cameras[1.0]
        all_cameras.extend(test_cameras_list)
        gaussians.init_RT_seq(all_cameras, trainable=False)
    
    print(f"Loaded gaussians.P with shape: {gaussians.P.shape}")
    if hasattr(gaussians, 'cameraid_to_pose_idx') and gaussians.cameraid_to_pose_idx:
        print(f"Camera ID to pose mapping: {dict(sorted(gaussians.cameraid_to_pose_idx.items()))}")
    
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    total_optimized = 0

    nvs_val = scene.getValCameras()
    if len(nvs_val) > 0:
        print(f"\nOptimizing poses for {len(nvs_val)} NVS val views")
        opt_val = render_set_optimize(
            dataset, "val", scene.loaded_iter, nvs_val, gaussians,
            pipeline, background, optim_test_pose_iter=optim_test_pose_iter, scene=scene
        )
        total_optimized += len(opt_val)

    nvs_test = scene.getTestCameras()
    if len(nvs_test) > 0:
        print(f"\nOptimizing poses for {len(nvs_test)} NVS test views (so test metrics use pose-optimized P)")
        opt_test = render_set_optimize(
            dataset, "test", scene.loaded_iter, nvs_test, gaussians,
            pipeline, background, optim_test_pose_iter=optim_test_pose_iter, scene=scene
        )
        total_optimized += len(opt_test)

    if total_optimized > 0:
        print(f"\nTotal optimized {total_optimized} pose(s). P updated in memory (not saved to checkpoint).")
    else:
        print("No NVS views in val/test, skipping pose optimization")

    return scene, gaussians

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_val : bool, skip_test : bool, render_mesh: bool, optimize_nvs_poses=False, optim_test_pose_iter=500):
    if optimize_nvs_poses:
        scene, gaussians = load_checkpoint_and_optimize_nvs_poses(dataset, iteration, pipeline, optim_test_pose_iter)
    else:
        if dataset.bind_to_mesh:
            gaussians = FlameGaussianModel(dataset.sh_degree)
        else:
            gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    with torch.no_grad():
        if dataset.target_path != "":
             name = os.path.basename(os.path.normpath(dataset.target_path))
             # when loading from a target path, test cameras are merged into the train cameras
             render_set(dataset, f'{name}', scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, render_mesh)
        else:
            if not skip_train:
                render_set(dataset, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, render_mesh)
            
            if not skip_val:
                render_set(dataset, "val", scene.loaded_iter, scene.getValCameras(), gaussians, pipeline, background, render_mesh)

            if not skip_test:
                render_set(dataset, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, render_mesh)



if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_val", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--render_mesh", action="store_true")
    parser.add_argument("--optim_nvs_pose", action="store_true", help="Optimize NVS (test view) poses before rendering")
    parser.add_argument("--optim_test_pose_iter", default=500, type=int, help="Number of iterations for test pose optimization")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(
        model.extract(args), args.iteration, pipeline.extract(args), 
        args.skip_train, args.skip_val, args.skip_test, args.render_mesh,
        optimize_nvs_poses=args.optim_nvs_pose,
        optim_test_pose_iter=args.optim_test_pose_iter
    )
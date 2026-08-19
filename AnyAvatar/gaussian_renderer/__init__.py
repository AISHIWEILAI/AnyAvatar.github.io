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
import math
from typing import Union
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene import GaussianModel, FlameGaussianModel
from utils.sh_utils import eval_sh
from utils.pose_utils import get_camera_from_tensor, quadmultiply


def quad2rotation(q):
    if not isinstance(q, torch.Tensor):
        q = torch.tensor(q, dtype=torch.float32)

    if q.dim() == 1:
        q = q.unsqueeze(0)
        is_single = True
    else:
        is_single = False

    norm = torch.sqrt(
        q[:, 0] * q[:, 0] + q[:, 1] * q[:, 1] + q[:, 2] * q[:, 2] + q[:, 3] * q[:, 3]
    )
    q = q / (norm[:, None] + 1e-8)

    w = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    batch_size = q.size(0)
    rot = torch.zeros((batch_size, 3, 3), device=q.device, dtype=q.dtype)
    rot[:, 0, 0] = 1 - 2 * (y * y + z * z)
    rot[:, 0, 1] = 2 * (x * y - w * z)
    rot[:, 0, 2] = 2 * (x * z + w * y)
    rot[:, 1, 0] = 2 * (x * y + w * z)
    rot[:, 1, 1] = 1 - 2 * (x * x + z * z)
    rot[:, 1, 2] = 2 * (y * z - w * x)
    rot[:, 2, 0] = 2 * (x * z - w * y)
    rot[:, 2, 1] = 2 * (y * z + w * x)
    rot[:, 2, 2] = 1 - 2 * (x * x + y * y)

    if is_single:
        rot = rot.squeeze(0)

    return rot


def getWorld2View2_tensor(R, T, translate=None, scale=1.0):
    translate = torch.zeros(3, device=R.device, dtype=R.dtype)
    scale = torch.tensor(scale, device=R.device, dtype=R.dtype)

    Rt = torch.zeros(4, 4, device=R.device, dtype=torch.float32)
    Rt[:3, :3] = R.transpose(0, 1)
    Rt[:3, 3] = T
    Rt[3, 3] = 1.0

    R_T = Rt[:3, :3].transpose(0, 1)
    t_vec = Rt[:3, 3]
    C2W = torch.zeros(4, 4, device=R.device, dtype=torch.float32)
    C2W[:3, :3] = R_T
    C2W[:3, 3] = -R_T @ t_vec
    C2W[3, 3] = 1.0

    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center

    R_T_final = C2W[:3, :3].transpose(0, 1)
    t_vec_final = C2W[:3, 3]
    Rt_final = torch.zeros(4, 4, device=R.device, dtype=torch.float32)
    Rt_final[:3, :3] = R_T_final
    Rt_final[:3, 3] = -R_T_final @ t_vec_final
    Rt_final[3, 3] = 1.0

    return Rt_final


def render(viewpoint_camera, pc : Union[GaussianModel, FlameGaussianModel], pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, camera_pose=None):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    screenspace_points = (
        torch.zeros_like(
            pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda"
        )
        + 0
    )
    try:
        screenspace_points.retain_grad()
    except:
        pass

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    if camera_pose is None:
        w2c = viewpoint_camera.world_view_transform.cuda()
        projmatrix = viewpoint_camera.full_proj_transform.cuda()
        campos = viewpoint_camera.camera_center.cuda()
    else:
        w2c = torch.eye(4).cuda()
        proj_matrix = viewpoint_camera.projection_matrix.cuda()
        projmatrix = (w2c.unsqueeze(0).bmm(proj_matrix.unsqueeze(0))).squeeze(0)
        campos = camera_pose

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=w2c,
        projmatrix=projmatrix,
        sh_degree=pc.active_sh_degree,
        campos=campos,
        prefiltered=False,
        debug=pipe.debug,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    if camera_pose is not None:
        if camera_pose.device.type != 'cuda':
            camera_pose = camera_pose.cuda()

        rel_w2c = get_camera_from_tensor(camera_pose)
        gaussians_xyz = pc.get_xyz.clone()
        gaussians_rot = pc.get_rotation.clone()

        xyz_ones = torch.ones(gaussians_xyz.shape[0], 1).cuda().float()
        xyz_homo = torch.cat((gaussians_xyz, xyz_ones), dim=1)
        gaussians_xyz_trans = (rel_w2c @ xyz_homo.T).T[:, :3]
        gaussians_rot_trans = quadmultiply(camera_pose[:4], gaussians_rot)
        means3D = gaussians_xyz_trans
        rotations_use = gaussians_rot_trans
    else:
        means3D = pc.get_xyz
        rotations_use = pc.get_rotation

    means2D = screenspace_points
    opacity = pc.get_opacity

    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = rotations_use

    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features().transpose(1, 2).view(
                -1, 3, (pc.max_sh_degree + 1) ** 2
            )
            dir_pp = pc.get_xyz - viewpoint_camera.camera_center.repeat(
                pc.get_features().shape[0], 1
            )
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            refined_R = quad2rotation(camera_pose[:4])
            refined_T = camera_pose[4:]
            world_view_transform_from_RT = getWorld2View2_tensor(refined_R, refined_T).transpose(0, 1)
            refined_camera_center = torch.linalg.inv(world_view_transform_from_RT)[3, :3]
            shs = pc.get_features(cam_pos=refined_camera_center)
    else:
        colors_precomp = override_color

    rendered_image, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )

    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
    }

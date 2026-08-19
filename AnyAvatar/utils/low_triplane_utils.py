
import torch
import itertools
import torch.nn as nn
import torch.nn.functional as F

relu = torch.nn.ReLU()

def grid_sample_wrapper(grid: torch.Tensor, coords: torch.Tensor, feature_dim, align_corners: bool = True) -> torch.Tensor:
    # grid: [1, C, H, W], coords: [B, N, 2]
    grid_dim = coords.shape[-1]
    if grid.dim() == grid_dim + 1:
        grid = grid.unsqueeze(0)
    if coords.dim() == 2:
        coords = coords.unsqueeze(0)
    coords = coords.to(grid.device)
    coords = coords.view([coords.shape[0]] + [1] * (grid_dim - 1) + list(coords.shape[1:]))
    interp_grid = grid[:, :feature_dim, :, :]
    B, feature_dim = interp_grid.shape[:2]
    n = coords.shape[-2]
    out = F.grid_sample(interp_grid, coords, align_corners=align_corners, mode='bilinear', padding_mode='border')
    out = out.view(B, feature_dim, n).transpose(-1, -2)
    return out.squeeze()

def init_lowrank_planes(grid_dim, in_dim, out_dim, resolution, a, b, rank):
    planes = list(itertools.combinations(range(in_dim), grid_dim))
    plane_coefs = nn.ParameterList()
    for i, plane in enumerate(planes):
        feat_dim = int(out_dim / 2) if i == 0 else int(out_dim / 4)
        U = nn.Parameter(torch.empty(1, feat_dim, rank, resolution[plane[0]])).cuda()
        V = nn.Parameter(torch.empty(1, feat_dim, rank, resolution[plane[1]])).cuda()
        nn.init.uniform_(U, a=a, b=b)
        nn.init.uniform_(V, a=a, b=b)
        plane_coefs.append((U, V))
    return plane_coefs

def reconstruct_plane(U, V):
    # U: [1, C, r, H], V: [1, C, r, W] -> [1, C, H, W]
    return torch.einsum('bcrh,bcrw->bchw', U, V)

def interpolate_ms_features(points, triplane, plane_dim, concat_f, num_levels):
    planes = list(itertools.combinations(range(points.shape[-1]), plane_dim))
    multi_scale_interp = [] if concat_f else 0.
    for scale, plane in enumerate(triplane[:num_levels] if num_levels is not None else triplane):
        interp_space = []
        for ci, coo_comb in enumerate(planes):
            U, V = plane[ci]
            feat_plane = reconstruct_plane(U, V)
            feature_dim = feat_plane.shape[1]
            interp_out_plane = grid_sample_wrapper(feat_plane, points[..., coo_comb], feature_dim).view(-1, feature_dim)
            interp_space.append(interp_out_plane)
        interp_space = torch.cat(interp_space, dim=-1)
        if concat_f:
            multi_scale_interp.append(interp_space)
        else:
            multi_scale_interp = multi_scale_interp + interp_space
    if concat_f:
        multi_scale_interp = torch.cat(multi_scale_interp, dim=-1)
    return multi_scale_interp

class MLP(nn.Module):
    def __init__(self, mlptype, dim_in, dim_out, dim_hidden, num_layers):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers
        self.mlptype = mlptype
        net = []
        for l in range(num_layers):
            net.append(nn.Linear(self.dim_in if l == 0 else self.dim_hidden, self.dim_out if l == num_layers - 1 else self.dim_hidden, bias=False))
        self.net = nn.ModuleList(net)
    def forward(self, x):
        for l in range(self.num_layers):
            x = self.net[l](x)
            if l != self.num_layers - 1:
                x = relu(x)
        return x

class LowRankTriPlaneNetwork(nn.Module):
    def __init__(self, spatial_bounds, grid_dim=2, in_dim=3, out_dim=64, resolution=[64, 64, 64], rank=8, a=-0.1, b=0.1):
        super().__init__()
        self.grid_dim = grid_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.base_resolution = resolution
        self.rank = rank
        self.multi_scale_res = [1]
        self.concat_feature = False
        self.spatial_bounds = spatial_bounds
        assert self.in_dim == len(self.base_resolution), "Resolution must have same number of elements as input-dimension"
        self.tri_plane = nn.ModuleList()
        for i in range(len(self.multi_scale_res)):
            res_scale = self.multi_scale_res[i]
            resolution = [res_scale * res for res in self.base_resolution]
            plane_coefs = init_lowrank_planes(self.grid_dim, self.in_dim, self.out_dim, resolution, a=a, b=b, rank=self.rank)
            if self.concat_feature:
                self.feature_dim += sum([U.shape[1] for U, V in plane_coefs])
            else:
                self.feature_dim = sum([U.shape[1] for U, V in plane_coefs])
            self.tri_plane.append(plane_coefs)
        self.hidden_dim_view = 128
        self.num_layers_view = 2
        self.color_out_dim = 3
        self.shs_net = MLP(mlptype='color', dim_in=self.feature_dim, dim_out=self.color_out_dim,
                           dim_hidden=self.hidden_dim_view, num_layers=self.num_layers_view)

    def forward(self, xyz):
        assert len(xyz.shape) == 2 and xyz.shape[-1] == self.in_dim, 'input points dim must be (num_points, 3)'
        xyz_norm = (xyz - self.spatial_bounds[0]) / (self.spatial_bounds[1] - self.spatial_bounds[0]) * 2 - 1

        canonical_f = interpolate_ms_features(xyz_norm, self.tri_plane, plane_dim=self.grid_dim, concat_f=self.concat_feature, num_levels=None)

        color = self.shs_net(canonical_f)
        return color.reshape((-1, 1, 3))



# Initialize low-rank factorized planes
# def init_low_rank_planes(grid_dim, in_dim, out_dim, resolution, a, b, rank):
#     planes = list(itertools.combinations(range(in_dim), grid_dim))  # Generate the 2D plane combinations
#     plane_coefs = nn.ParameterList()
#     for i, plane in enumerate(planes):
#         if i == 0:
#             # First plane (i=0, XY plane): [1, 32, 128, 128]
#             U = nn.Parameter(torch.empty([1, rank, resolution[plane[0]], resolution[plane[1]]]))  # U shape
#             V = nn.Parameter(torch.empty([1, rank, resolution[plane[0]], resolution[plane[1]]]))  # V shape
#         else:
#             # Other planes (XZ, YZ): [1, 16, 128, 128]
#             U = nn.Parameter(torch.empty([1, rank, resolution[plane[0]], resolution[plane[1]]]))  # U shape
#             V = nn.Parameter(torch.empty([1, rank, resolution[plane[0]], resolution[plane[1]]]))  # V shape
#         nn.init.uniform_(U, a=a, b=b)
#         nn.init.uniform_(V, a=a, b=b)
#         plane_coefs.append((U, V))
#     return plane_coefs

# Tri-plane network with low-rank factorization
# class LowRankTriPlaneNetwork(nn.Module):
#     def __init__(self, spatial_bounds, grid_dim=2, in_dim=3, out_dim=64, resolution=[64, 64, 64], rank=16, a=-0.1, b=0.1):
#         super().__init__()
#         self.grid_dim = grid_dim
#         self.in_dim = in_dim
#         self.out_dim = out_dim
#         self.base_resolution = resolution
#         self.rank = rank  # Low-rank factor
#         self.multi_scale_res = [1]
#         self.concat_feature = False
#         self.spatial_bounds = spatial_bounds

#         assert self.in_dim == len(self.base_resolution), "Resolution must have same number of elements as input-dimension"
#         self.tri_plane = nn.ModuleList()
#         for i in range(len(self.multi_scale_res)):
#             res_scale = self.multi_scale_res[i]
#             resolution = [res_scale * res for res in self.base_resolution]
#             plane_coefs = init_low_rank_planes(self.grid_dim, self.in_dim, self.out_dim, resolution, a=a, b=b, rank=self.rank)

#             if self.concat_feature:
#                 self.feature_dim += plane_coefs[-1].shape[1] 
#             else:
#                 self.feature_dim = 32 + 16 + 16
#             self.tri_plane.append(plane_coefs)

#         # self.pe, self.view_dim = get_embedder(4)
#         self.hidden_dim_view = 128
#         self.num_layers_view = 2
#         self.color_out_dim = 3
#         self.shs_net = MLP(mlptype='color', dim_in=self.feature_dim, dim_out=self.color_out_dim,
#                            dim_hidden=self.hidden_dim_view, num_layers=self.num_layers_view)

#     def forward(self, xyz_cano, dirs):  
#         assert len(xyz_cano.shape) == 2 and xyz_cano.shape[-1] == self.in_dim, 'input points dim must be (num_points, 3)'

#         xyz_norm = (xyz_cano - self.spatial_bounds[0]) / (self.spatial_bounds[1] - self.spatial_bounds[0]) * 2 - 1
#         canonical_f = interpolate_ms_features(xyz_norm, self.tri_plane, plane_dim=self.grid_dim, concat_f=self.concat_feature, num_levels=None)

#         input = canonical_f
#         color = self.shs_net(input)
#         return color.reshape((-1, 1, 3))

    # def get_tvloss(self):
    #     tv_loss = 0.0
    #     for triplane in self.tri_plane:
    #         for U, V in triplane:
    #             # Compute TV loss for U and V (encourages smoothness)
    #             tv_loss += torch.sum(torch.abs(U[:, :, :, :-1] - U[:, :, :, 1:])) + torch.sum(torch.abs(V[:, :, :-1, :] - V[:, :, 1:, :]))
    #     return tv_loss / (len(self.tri_plane) * len(triplane))


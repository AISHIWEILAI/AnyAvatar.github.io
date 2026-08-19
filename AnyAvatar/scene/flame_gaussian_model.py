# 
# Toyota Motor Europe NV/SA and its affiliated companies retain all intellectual 
# property and proprietary rights in and to this software and related documentation. 
# Any commercial use, reproduction, disclosure or distribution of this software and 
# related documentation without an express license agreement from Toyota Motor Europe NV/SA 
# is strictly prohibited.
#

from pathlib import Path
import numpy as np
import torch
import pickle
# from vht.model.flame import FlameHead
from flame_model.flame import FlameHead

from .gaussian_model import GaussianModel
from utils.graphics_utils import compute_face_orientation
# from pytorch3d.transforms import matrix_to_quaternion
from roma import rotmat_to_unitquat, quat_xyzw_to_wxyz
from utils.triplane_utils import TriPlaneNetwork
from utils.low_triplane_utils import LowRankTriPlaneNetwork


class Struct(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)


class FlameGaussianModel(GaussianModel):
    def __init__(self, sh_degree : int, disable_flame_static_offset=False, not_finetune_flame_params=False, n_shape=300, n_expr=100):
        super().__init__(sh_degree)

        self.disable_flame_static_offset = disable_flame_static_offset
        self.not_finetune_flame_params = not_finetune_flame_params
        self.n_shape = n_shape
        self.n_expr = n_expr

        self.flame_model = FlameHead(
            n_shape, 
            n_expr,
            add_teeth=True,
        ).cuda()
        self.flame_param = None
        self.flame_param_orig = None
        self.mesh_translation_offset = None

        # binding is initialized once the mesh topology is known
        if self.binding is None:
            self.binding = torch.arange(len(self.flame_model.faces)).cuda()
            self.binding_counter = torch.ones(len(self.flame_model.faces), dtype=torch.int32).cuda()

        with open('flame_model/assets/flame/FLAME_masks.pkl', "rb") as f:
            ss = pickle.load(f, encoding="latin1")
            self.flame_masks = Struct(**ss)

            self.vert_in_face = torch.from_numpy(self.flame_masks.face).cuda()
            self.tri_in_face = torch.where(torch.all(torch.isin(self.flame_model.faces, self.vert_in_face), dim = -1))[0]

    def load_meshes(self, train_meshes, test_meshes, tgt_train_meshes, tgt_test_meshes, train_cameras=None):
        if self.flame_param is None:
            meshes = {**train_meshes, **test_meshes}
            tgt_meshes = {**tgt_train_meshes, **tgt_test_meshes}
            pose_meshes = meshes if len(tgt_meshes) == 0 else tgt_meshes
            
            self.num_timesteps = max(pose_meshes) + 1  # required by viewers
            num_verts = self.flame_model.v_template.shape[0]

            if not self.disable_flame_static_offset:
                static_offset = torch.from_numpy(meshes[0]['static_offset'])
                if static_offset.shape[0] != num_verts:
                    static_offset = torch.nn.functional.pad(static_offset, (0, 0, 0, num_verts - meshes[0]['static_offset'].shape[1]))
            else:
                static_offset = torch.zeros([num_verts, 3])

            T = self.num_timesteps

            self.flame_param = {
                'shape': torch.from_numpy(meshes[0]['shape']),
                'expr': torch.zeros([T, meshes[0]['expr'].shape[1]]),
                'rotation': torch.zeros([T, 3]),
                'neck_pose': torch.zeros([T, 3]),
                'jaw_pose': torch.zeros([T, 3]),
                'eyes_pose': torch.zeros([T, 6]),
                'translation': torch.zeros([T, 3]),
                'static_offset': static_offset,
                'dynamic_offset': torch.zeros([T, num_verts, 3]),
            }

            for i, mesh in pose_meshes.items():
                self.flame_param['expr'][i] = torch.from_numpy(mesh['expr'])
                self.flame_param['rotation'][i] = torch.from_numpy(mesh['rotation'])
                self.flame_param['neck_pose'][i] = torch.from_numpy(mesh['neck_pose'])
                self.flame_param['jaw_pose'][i] = torch.from_numpy(mesh['jaw_pose'])
                self.flame_param['eyes_pose'][i] = torch.from_numpy(mesh['eyes_pose'])
                self.flame_param['translation'][i] = torch.from_numpy(mesh['translation'])
                # self.flame_param['dynamic_offset'][i] = torch.from_numpy(mesh['dynamic_offset'])
            
            for k, v in self.flame_param.items():
                self.flame_param[k] = v.float().cuda()
            
            self.flame_param_orig = {k: v.clone() for k, v in self.flame_param.items()}
            self.flame_param_orig = {k: v.clone() for k, v in self.flame_param.items()}
            self.update_cano_mesh()
        else:
            # NOTE: not sure when this happens
            import ipdb; ipdb.set_trace()
            pass
        bound_upper = torch.max(self.verts_cano[0].detach(), dim = 0)[0] + 0.005
        bound_lower = torch.min(self.verts_cano[0].detach(), dim = 0)[0] - 0.005
        self.trip_spatial_bounds = torch.cat([bound_lower, bound_upper], dim = 0).reshape((2, 3))
        self.trip_spatial_bounds[0, 1] -= 0.025
        self.trip_spatial_bounds[0, 0] = -0.25
        self.trip_spatial_bounds[1, 0] = 0.25

        self.triplane = TriPlaneNetwork(self.trip_spatial_bounds).cuda()
        self.low_triplane = LowRankTriPlaneNetwork(self.trip_spatial_bounds).cuda()
    
    def update_cano_mesh(self):
        flame_param = self.flame_param

        verts, verts_cano = self.flame_model(
            flame_param['shape'][None, ...],
            torch.zeros_like(flame_param['expr'][[0]]).cuda(),
            torch.zeros_like(flame_param['rotation'][[0]]).cuda(),
            torch.zeros_like(flame_param['neck_pose'][[0]]).cuda(),
            torch.zeros_like(flame_param['jaw_pose'][[0]]).cuda(),
            torch.zeros_like(flame_param['eyes_pose'][[0]]).cuda(),
            torch.zeros_like(flame_param['translation'][[0]]).cuda(),
            zero_centered_at_root_node=False,
            return_landmarks=False,
            return_verts_cano=True,
            static_offset=flame_param['static_offset'],
            dynamic_offset=None,
        )

        self.verts_cano = verts
        self.cano_face_max = torch.max(self.verts_cano[0].detach()[self.vert_in_face], dim = 0)[0]
        self.cano_face_min = torch.min(self.verts_cano[0].detach()[self.vert_in_face], dim = 0)[0]
        self.cano_head_max = torch.max(self.verts_cano[0].detach(), dim = 0)[0]
        self.cano_head_min = torch.min(self.verts_cano[0].detach(), dim = 0)[0]

        self.face_center_cano = verts[:, self.flame_model.faces].mean(dim=-2).squeeze(0)
        self.face_orien_mat_cano, self.face_scaling_cano = compute_face_orientation(verts_cano.squeeze(0), self.flame_model.faces.squeeze(0), return_scale=True)

    def update_mesh_by_param_dict(self, flame_param):
        if 'shape' in flame_param:
            shape = flame_param['shape']
        else:
            shape = self.flame_param['shape']

        if 'static_offset' in flame_param:
            static_offset = flame_param['static_offset']
        else:
            static_offset = self.flame_param['static_offset']

        verts, verts_cano = self.flame_model(
            shape[None, ...],
            flame_param['expr'].cuda(),
            flame_param['rotation'].cuda(),
            flame_param['neck'].cuda(),
            flame_param['jaw'].cuda(),
            flame_param['eyes'].cuda(),
            flame_param['translation'].cuda(),
            zero_centered_at_root_node=False,
            return_landmarks=False,
            return_verts_cano=True,
            static_offset=static_offset,
        )
        self.update_mesh_properties(verts, verts_cano)

    def select_mesh_by_timestep(self, timestep, original=False):
        self.timestep = timestep
        flame_param = self.flame_param_orig if original and self.flame_param_orig != None else self.flame_param
        self.xyz_cano = self.get_xyz_cano

        verts, verts_cano = self.flame_model(
            flame_param['shape'][None, ...],
            flame_param['expr'][[timestep]],
            flame_param['rotation'][[timestep]],
            flame_param['neck_pose'][[timestep]],
            flame_param['jaw_pose'][[timestep]],
            flame_param['eyes_pose'][[timestep]],
            flame_param['translation'][[timestep]],
            zero_centered_at_root_node=False,
            return_landmarks=False,
            return_verts_cano=True,
            static_offset=flame_param['static_offset'],
            dynamic_offset=flame_param['dynamic_offset'][[timestep]],
        )
        self.update_mesh_properties(verts, verts_cano)
    
    def update_mesh_properties(self, verts, verts_cano):
        faces = self.flame_model.faces
        triangles = verts[:, faces]

        # position
        self.face_center = triangles.mean(dim=-2).squeeze(0)

        # orientation and scale
        self.face_orien_mat, self.face_scaling = compute_face_orientation(verts.squeeze(0), faces.squeeze(0), return_scale=True)
        # self.face_orien_quat = matrix_to_quaternion(self.face_orien_mat)  # pytorch3d (WXYZ)
        self.face_orien_quat = quat_xyzw_to_wxyz(rotmat_to_unitquat(self.face_orien_mat))  # roma

        # for mesh rendering
        self.verts = verts
        self.faces = faces

        # for mesh regularization
        self.verts_cano = verts_cano
    
    def compute_dynamic_offset_loss(self):
        # loss_dynamic = (self.flame_param['dynamic_offset'][[self.timestep]] - self.flame_param_orig['dynamic_offset'][[self.timestep]]).norm(dim=-1)
        loss_dynamic = self.flame_param['dynamic_offset'][[self.timestep]].norm(dim=-1)
        return loss_dynamic.mean()
    
    def compute_laplacian_loss(self):
        # offset = self.flame_param['static_offset'] + self.flame_param['dynamic_offset'][[self.timestep]]
        offset = self.flame_param['dynamic_offset'][[self.timestep]]
        verts_wo_offset = (self.verts_cano - offset).detach()
        verts_w_offset = verts_wo_offset + offset

        L = self.flame_model.laplacian_matrix[None, ...].detach()  # (1, V, V)
        lap_wo = L.bmm(verts_wo_offset).detach()
        lap_w = L.bmm(verts_w_offset)
        diff = (lap_wo - lap_w) ** 2
        diff = diff.sum(dim=-1, keepdim=True)
        return diff.mean()
    
    def training_setup(self, training_args):
        super().training_setup(training_args)
        
        # Add triplane network parameters to optimizer
        if self.triplane is not None:
            # Get all triplane parameters
            triplane_params = list(self.triplane.parameters())
            if triplane_params:
                param_triplane = {'params': triplane_params, 'lr': 1e-4, "name": "triplane"}
                self.optimizer.add_param_group(param_triplane)
                print(f"Added triplane parameters to optimizer ({len(triplane_params)} params)")
        

        # Add low-rank triplane network parameters to optimizer
        if self.low_triplane is not None:
            # Get all triplane parameters
            triplane_params = list(self.low_triplane.parameters())
            if triplane_params:
                param_low_triplane = {'params': triplane_params, 'lr': 1e-4, "name": "low_triplane"}
                self.optimizer.add_param_group(param_low_triplane)
                print(f"Added low-rank triplane parameters to optimizer ({len(param_low_triplane)} params)")

        if self.not_finetune_flame_params:
            return

        # # shape
        # self.flame_param['shape'].requires_grad = True
        # param_shape = {'params': [self.flame_param['shape']], 'lr': 1e-5, "name": "shape"}
        # self.optimizer.add_param_group(param_shape)

        # pose
        self.flame_param['rotation'].requires_grad = True
        self.flame_param['neck_pose'].requires_grad = True
        self.flame_param['jaw_pose'].requires_grad = True
        self.flame_param['eyes_pose'].requires_grad = True
        params = [
            self.flame_param['rotation'],
            self.flame_param['neck_pose'],
            self.flame_param['jaw_pose'],
            self.flame_param['eyes_pose'],
        ]
        param_pose = {'params': params, 'lr': training_args.flame_pose_lr, "name": "pose"}
        self.optimizer.add_param_group(param_pose)

        # translation
        self.flame_param['translation'].requires_grad = True
        param_trans = {'params': [self.flame_param['translation']], 'lr': training_args.flame_trans_lr, "name": "trans"}
        self.optimizer.add_param_group(param_trans)
        
        # expression
        self.flame_param['expr'].requires_grad = True
        param_expr = {'params': [self.flame_param['expr']], 'lr': training_args.flame_expr_lr, "name": "expr"}
        self.optimizer.add_param_group(param_expr)

        # # static_offset
        # self.flame_param['static_offset'].requires_grad = True
        # param_static_offset = {'params': [self.flame_param['static_offset']], 'lr': 1e-6, "name": "static_offset"}
        # self.optimizer.add_param_group(param_static_offset)

        # # dynamic_offset
        # self.flame_param['dynamic_offset'].requires_grad = True
        # param_dynamic_offset = {'params': [self.flame_param['dynamic_offset']], 'lr': 1.6e-6, "name": "dynamic_offset"}
        # self.optimizer.add_param_group(param_dynamic_offset)

    def load_triplane_weights(self, parent_path):
        parent_path = Path(parent_path)

        triplane_path = parent_path / "triplane.pth"
        if triplane_path.exists() and self.triplane is not None:
            try:
                state_dict = torch.load(str(triplane_path))
                self.triplane.load_state_dict(state_dict)
                print(f"Loaded triplane weights from {triplane_path}")
            except Exception as e:
                print(f"Warning: Failed to load triplane weights from {triplane_path}: {e}")

        low_triplane_path = parent_path / "low_triplane.pth"
        if low_triplane_path.exists() and self.low_triplane is not None:
            try:
                state_dict = torch.load(str(low_triplane_path))
                self.low_triplane.load_state_dict(state_dict)
                print(f"Loaded low_triplane weights from {low_triplane_path}")
            except Exception as e:
                print(f"Warning: Failed to load low_triplane weights from {low_triplane_path}: {e}")

    def save_ply(self, path):
        super().save_ply(path)

        npz_path = Path(path).parent / "flame_param.npz"
        flame_param = {k: v.cpu().numpy() for k, v in self.flame_param.items()}
        
        # Save mesh_translation_offset if it exists
        if self.mesh_translation_offset is not None:
            flame_param['mesh_translation_offset'] = self.mesh_translation_offset.cpu().numpy()
        
        np.savez(str(npz_path), **flame_param)
        
        # Save triplane network weights
        if self.triplane is not None:
            triplane_path = Path(path).parent / "triplane.pth"
            torch.save(self.triplane.state_dict(), str(triplane_path))

        # Save low_triplane network weights
        if self.low_triplane is not None:
            low_triplane_path = Path(path).parent / "low_triplane.pth"
            torch.save(self.low_triplane.state_dict(), str(low_triplane_path))

    def load_ply(self, path, **kwargs):
        super().load_ply(path)
        
        self.load_triplane_weights(Path(path).parent)

        if not kwargs['has_target']:
            # When there is no target motion specified, use the finetuned FLAME parameters.
            # This operation overwrites the FLAME parameters loaded from the dataset.
            npz_path = Path(path).parent / "flame_param.npz"
            flame_param = np.load(str(npz_path))
            flame_param = {k: torch.from_numpy(v).cuda() for k, v in flame_param.items()}
            
            # Load mesh_translation_offset if it exists
            if 'mesh_translation_offset' in flame_param:
                self.mesh_translation_offset = flame_param.pop('mesh_translation_offset')

            self.flame_param = flame_param
            self.num_timesteps = self.flame_param['expr'].shape[0]  # required by viewers
        
        if 'motion_path' in kwargs and kwargs['motion_path'] is not None:
            # When there is a motion sequence specified, load only dynamic parameters.
            motion_path = Path(kwargs['motion_path'])
            flame_param = np.load(str(motion_path))
            flame_param = {k: torch.from_numpy(v).cuda() for k, v in flame_param.items() if v.dtype == np.float32}

            self.flame_param = {
                # keep the static parameters
                'shape': self.flame_param['shape'],
                'static_offset': self.flame_param['static_offset'],
                # update the dynamic parameters
                'translation': flame_param['translation'],
                'rotation': flame_param['rotation'],
                'neck_pose': flame_param['neck_pose'],
                'jaw_pose': flame_param['jaw_pose'],
                'eyes_pose': flame_param['eyes_pose'],
                'expr': flame_param['expr'],
                'dynamic_offset': flame_param['dynamic_offset'],
            }
            self.num_timesteps = self.flame_param['expr'].shape[0]  # required by viewers
        
        if 'disable_fid' in kwargs and len(kwargs['disable_fid']) > 0:
            mask = (self.binding[:, None] != kwargs['disable_fid'][None, :]).all(-1)

            self.binding = self.binding[mask]
            self._xyz = self._xyz[mask]
            self._features_dc = self._features_dc[mask]
            self._features_rest = self._features_rest[mask]
            self._scaling = self._scaling[mask]
            self._rotation = self._rotation[mask]
            self._opacity = self._opacity[mask]

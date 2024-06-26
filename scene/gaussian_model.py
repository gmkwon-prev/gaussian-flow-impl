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
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
import math
class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.time = 0
        self._pt_poly_len =0
        self._pt_fourier_len=0
        self._rot_poly_len=0
        self._rot_fourier_len=0
        self._fdc_poly_len =0
        self._fdc_fourier_len = 0
        self._frt_poly_len=0
        self._frt_fourier_len=0
        self._position_time_parameter = torch.empty(0)
        self._rotation_time_parameter = torch.empty(0)
        self._features_dc_time_parameter = torch.empty(0)
        self._features_rest_time_parameter = torch.empty(0) # not use.
        self._lambda_s = torch.tensor(1.)
        self._lambda_b = torch.tensor(0.)
        self._lambda_optimizer = None
        self.setup_functions()

    def capture(self): # use on train.py
        parameters_len = [self._pt_poly_len, self._pt_fourier_len,
                    self._rot_poly_len,self._rot_fourier_len,
                    self._fdc_poly_len ,self._fdc_fourier_len,
                    self._frt_poly_len,self._frt_fourier_len]
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self._lambda_s,
            self._lambda_b,
            parameters_len,
            self._position_time_parameter,
            self._rotation_time_parameter,
            self._features_dc_time_parameter,
            self._features_rest_time_parameter,
        )
    
    def restore(self, model_args, training_args): # use on train.py to load.
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale,
        self._lambda_s,
        self._lambda_b,
        parameters_len,
        self._position_time_parameter,
        self._rotation_time_parameter,
        self._features_dc_time_parameter,
        self._features_rest_time_parameter,) = model_args

        self._pt_poly_len, self._pt_fourier_len, self._rot_poly_len,self._rot_fourier_len,\
                                self._fdc_poly_len ,self._fdc_fourier_len, self._frt_poly_len,self._frt_fourier_len = parameters_len
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        if self._rotation_time_parameter.shape[-1] != self._rot_poly_len+2* self._rot_fourier_len:
            print("type error")
            return self.rotation_activation(self._rotation)
        #param_len = int(self._rotation_time_parameter_len /3)
        return self.rotation_activation(self._rotation \
                + self.poly_diff(self._rotation_time_parameter[...,0:self._rot_poly_len], self.time, self._rot_poly_len) \
                + self.freq_diff(self._rotation_time_parameter[...,self._rot_poly_len:self._rot_poly_len+2* self._rot_fourier_len], self.time, self._rot_fourier_len))

    
    @property
    def get_xyz(self):
        if self._position_time_parameter.shape[-1] != self._pt_poly_len + 2*self._pt_fourier_len:
            print("type error")
            return self._xyz
        return self._xyz + self.poly_diff(self._position_time_parameter[...,0:self._pt_poly_len], self.time, self._pt_poly_len) \
                + self.freq_diff(self._position_time_parameter[...,self._pt_poly_len:self._pt_poly_len + 2*self._pt_fourier_len], self.time, self._pt_fourier_len)


    @property
    def get_features(self):

        features_dc = self._features_dc + self.poly_diff(self._features_dc_time_parameter[...,0:self._fdc_poly_len], self.time, self._fdc_poly_len) \
                        + self.freq_diff(self._features_dc_time_parameter[...,self._fdc_poly_len:self._fdc_poly_len + 2*self._fdc_fourier_len], self.time, self._fdc_fourier_len)
        features_rest = self._features_rest + self.poly_diff(self._features_rest_time_parameter[...,0:self._frt_poly_len], self.time, self._frt_poly_len) \
                        + self.freq_diff(self._features_rest_time_parameter[...,self._frt_poly_len:self._frt_poly_len + 2*self._frt_fourier_len], self.time, self._frt_fourier_len)
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def D_xyz(self):
        return self.poly_diff(self._position_time_parameter[...,0:self._pt_poly_len], self.time, self._pt_poly_len) \
                + self.freq_diff(self._position_time_parameter[...,self._pt_poly_len:self._pt_poly_len + 2* self._pt_fourier_len], self.time, self._pt_fourier_len)
    def D_rotation(self):
        return self.rotation_activation(self.poly_diff(self._rotation_time_parameter[...,0:self._rot_poly_len], self.time, self._rot_poly_len) \
                + self.freq_diff(self._rotation_time_parameter[...,self._rot_poly_len:self._rot_poly_len+2* self._rot_fourier_len], self.time, self._rot_fourier_len))
    
    def D_features(self):

        features_dc = self.poly_diff(self._features_dc_time_parameter[...,0:self._fdc_poly_len], self.time, self._fdc_poly_len) \
                        + self.freq_diff(self._features_dc_time_parameter[...,self._fdc_poly_len:self._fdc_poly_len + 2*self._fdc_fourier_len], self.time, self._fdc_fourier_len)
        features_rest = self.poly_diff(self._features_rest_time_parameter[...,0:self._frt_poly_len], self.time, self._frt_poly_len) \
                        + self.freq_diff(self._features_rest_time_parameter[...,self._frt_poly_len:self._frt_poly_len + 2*self._frt_fourier_len], self.time, self._frt_fourier_len)
        return torch.cat((features_dc, features_rest), dim=1)
    
    ### time-scale
    def setTime(self, t):
        self.time = t
        #self.time = self._lambda_s * t + self._lambda_b

    def setDDDM(self,dddm_param_len):
        self._pt_poly_len, self._pt_fourier_len = dddm_param_len[0]
        self._rot_poly_len, self._rot_fourier_len = dddm_param_len[1]
        self._fdc_poly_len, self._fdc_fourier_len = dddm_param_len[2]
        self._frt_poly_len, self._frt_fourier_len = dddm_param_len[3] # in my understand, uncessary for this paper

    def poly_diff(self, parameters, time, param_len):
        pol = torch.tensor([time ** i for i in range(1,1+param_len)], device="cuda")
        return torch.sum(parameters * pol, axis=-1 )
    
    def freq_diff(self, parameters, time, param_len):
        l = torch.tensor(range(1,1+param_len), device="cuda").repeat(2,1)
        sin = torch.sin(2 * math.pi * time * l[0])
        cos = torch.cos(2 * math.pi * time * l[1])
        return torch.sum(parameters[...,0:param_len]*sin, axis=-1) + torch.sum(parameters[...,param_len:]*cos, axis=-1)

    def get_time_smooth_loss(self, time_interval):
        cur_time = self.time

        cur_pos = self.get_xyz
        cur_rot = self.get_rotation
        cur_sh = self.get_features

        self.setTime(cur_time + time_interval)
        new_pos = self.get_xyz
        new_rot = self.get_rotation
        new_sh = self.get_features

        self.setTime(cur_time) # return origin time

        return torch.sqrt(torch.sum((cur_pos - new_pos)**2) + torch.sum((cur_rot - new_rot)**2) + torch.sum((cur_sh - new_sh)**2))


    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))

        position_time_parameter = torch.zeros([*self._xyz.shape, self._pt_poly_len + 2*self._pt_fourier_len], device ="cuda" )
        rotation_time_parameter = torch.zeros([*self._rotation.shape, self._rot_poly_len + 2*self._rot_fourier_len], device ="cuda" )
        features_dc_time_parameter = torch.zeros([*self._features_dc.shape, self._fdc_poly_len + 2* self._fdc_fourier_len], device="cuda")
        features_rest_time_parameter = torch.zeros([*self._features_rest.shape, self._frt_poly_len + 2*self._frt_fourier_len], device="cuda")

        self._lambda_s = nn.Parameter(torch.tensor(1.).requires_grad_(True))
        self._lambda_b = nn.Parameter(torch.tensor(0.).requires_grad_(True))

        self._position_time_parameter = nn.Parameter(position_time_parameter.requires_grad_(True))
        self._rotation_time_parameter = nn.Parameter(rotation_time_parameter.requires_grad_(True))
        self._features_dc_time_parameter = nn.Parameter(features_dc_time_parameter.requires_grad_(True))
        self._features_rest_time_parameter = nn.Parameter(features_rest_time_parameter.requires_grad_(True))


        self.max_radii2D = torch.zeros((self._xyz.shape[0]), device="cuda")



    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self._xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self._xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self._position_time_parameter], 'lr': 0.0, "name": "tp_pos"},
            {'params': [self._rotation_time_parameter], 'lr': 0.0, "name": "tp_rot"},
            {'params': [self._features_dc_time_parameter], 'lr': 0.0, "name": "tp_f_dc"},
            {'params': [self._features_rest_time_parameter], 'lr': 0.0, "name": "tp_f_rest"},
        ]


        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        lam = [
            {'params': [self._lambda_s], 'lr': 1e-3, "name": "lambda_s"},
            {'params': [self._lambda_b], 'lr': 1e-3, "name": "lambda_b"},
            ]
        self._lambda_optimizer = torch.optim.Adam(lam, lr=1e-5, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration, dddm_from_iter):
        ''' Learning rate scheduling per step '''
        lr_train = self.xyz_scheduler_args(iteration)
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                param_group['lr'] = lr_train
            if param_group["name"] in ["tp_pos", "tp_rot","tp_f_dc", "tp_f_rest"]:
                if iteration < dddm_from_iter:
                    param_group['lr'] = 0.0
                else:
                    param_group['lr'] = lr_train
        return lr_train

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        for i in range(self._position_time_parameter.shape[1]*self._position_time_parameter.shape[2]): ###
            l.append('tp_pos_{}'.format(i))
        for i in range(self._rotation_time_parameter.shape[1]*self._rotation_time_parameter.shape[2]): ###
            l.append('tp_rot_{}'.format(i))
        for i in range(self._features_dc_time_parameter.shape[1]*self._features_dc_time_parameter.shape[2]*self._features_dc_time_parameter.shape[3]):
            l.append('tp_f_dc_{}'.format(i))
        for i in range(self._features_rest_time_parameter.shape[1]*self._features_rest_time_parameter.shape[2]*self._features_rest_time_parameter.shape[3]):
            l.append('tp_f_rest_{}'.format(i))
        l.append('others')
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        position_time_parameter = self._position_time_parameter.flatten(start_dim=1).contiguous().detach().cpu().numpy() ###
        rotation_time_parameter = self._rotation_time_parameter.flatten(start_dim=1).contiguous().detach().cpu().numpy()
        features_dc_time_parameter = self._features_dc_time_parameter.transpose(1,2).flatten(start_dim=1).contiguous().detach().cpu().numpy()
        features_rest_time_parameter = self._features_rest_time_parameter.transpose(1,2).flatten(start_dim=1).contiguous().detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        others = np.zeros((xyz.shape[0], 1), dtype=xyz.dtype)
        others[0,0] = self._lambda_s
        others[1,0] = self._lambda_b
        others[2:10,0] = [self._pt_poly_len, self._pt_fourier_len, self._rot_poly_len,self._rot_fourier_len,
                        self._fdc_poly_len ,self._fdc_fourier_len, self._frt_poly_len,self._frt_fourier_len]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, position_time_parameter, rotation_time_parameter, features_dc_time_parameter, features_rest_time_parameter, others), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        
        others = np.asarray(plydata.elements[0]["others"])
        pt_poly_len = int(others[2])
        pt_fourier_len = int(others[3])
        rot_poly_len = int(others[4])
        rot_fourier_len = int(others[5])
        fdc_poly_len = int(others[6])
        fdc_fourier_len = int(others[7])
        frt_poly_len = int(others[8])
        frt_fourier_len = int(others[9])

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        pos_tp_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("tp_pos_")]
        pos_tp_names = sorted(pos_tp_names, key = lambda x: int(x.split('_')[-1] ))
        if len(pos_tp_names) != 3 * ( pt_poly_len + 2* pt_fourier_len): 
            print("position parmeter error")

        pos_tp = np.zeros((xyz.shape[0], len(pos_tp_names)))
        for idx, attr_name in enumerate(pos_tp_names):
            pos_tp[:, idx] = np.asarray(plydata.elements[0][attr_name])
        pos_tp = pos_tp.reshape((xyz.shape[0],3, pt_poly_len + 2* pt_fourier_len))

        rot_tp_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("tp_rot_")]
        rot_tp_names = sorted(rot_tp_names, key = lambda x: int(x.split('_')[-1] ))
        if len(rot_tp_names) != len(rot_names) * (rot_poly_len + 2 * rot_fourier_len): 
            print("rotation parmeter error")
        rot_tp = np.zeros((xyz.shape[0], len(rot_tp_names)))
        for idx, attr_name in enumerate(rot_tp_names):
            rot_tp[:, idx] = np.asarray(plydata.elements[0][attr_name])
        rot_tp = rot_tp.reshape((*rots.shape, rot_poly_len + 2 * rot_fourier_len))

        f_dc_tp_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("tp_f_dc_")]
        f_dc_tp_names = sorted(f_dc_tp_names, key = lambda x: int(x.split('_')[-1] ))
        if len(f_dc_tp_names) != 3 * (fdc_poly_len + 2*fdc_fourier_len):
            print("color parmeter error")
        f_dc_tp = np.zeros((xyz.shape[0], len(f_dc_tp_names)))
        for idx, attr_name in enumerate(f_dc_tp_names):
            f_dc_tp[:,idx] = np.asarray(plydata.elements[0][attr_name])
        f_dc_tp = f_dc_tp.reshape((*features_dc.shape, fdc_poly_len + 2*fdc_fourier_len))

        f_rest_tp_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("tp_f_rest_")]
        f_rest_tp_names = sorted(f_rest_tp_names, key = lambda x: int(x.split('_')[-1] ))
        if len(f_rest_tp_names) != len(extra_f_names) * (frt_poly_len + 2*frt_fourier_len): 
            print("sh rest parameter error")
        f_rest_tp = np.zeros((xyz.shape[0], len(f_rest_tp_names)))
        for idx, attr_name in enumerate(f_rest_tp_names):
            f_rest_tp[:, idx] = np.asarray(plydata.elements[0][attr_name])
        f_rest_tp = f_rest_tp.reshape((*features_extra.shape, frt_poly_len + 2*frt_fourier_len))

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        
        self._pt_poly_len = torch.tensor(int(others[2]), dtype=torch.int, device = "cuda")
        self._pt_fourier_len = torch.tensor(int(others[3]), dtype=torch.int, device = "cuda")
        self._rot_poly_len = torch.tensor(int(others[4]), dtype=torch.int, device = "cuda")
        self._rot_fourier_len = torch.tensor(int(others[5]), dtype=torch.int, device = "cuda")
        self._fdc_poly_len = torch.tensor(int(others[6]), dtype=torch.int, device = "cuda")
        self._fdc_fourier_len = torch.tensor(int(others[7]), dtype=torch.int, device = "cuda")
        self._frt_poly_len = torch.tensor(int(others[8]), dtype=torch.int, device = "cuda")
        self._frt_fourier_len = torch.tensor(int(others[9]), dtype=torch.int, device = "cuda")

        self._position_time_parameter = nn.Parameter(torch.tensor(pos_tp, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation_time_parameter = nn.Parameter(torch.tensor(rot_tp, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc_time_parameter = nn.Parameter(torch.tensor(f_dc_tp, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest_time_parameter = nn.Parameter(torch.tensor(f_rest_tp, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        
        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        
        self._position_time_parameter = optimizable_tensors["tp_pos"]
        self._rotation_time_parameter = optimizable_tensors["tp_rot"]
        self._features_dc_time_parameter = optimizable_tensors["tp_f_dc"]
        self._features_rest_time_parameter = optimizable_tensors["tp_f_rest"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tp_pos, new_tp_rot, new_tp_f_dc, new_tp_f_rest):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation,
        "tp_pos" : new_tp_pos,
        "tp_rot" : new_tp_rot,
        "tp_f_dc" : new_tp_f_dc,
        "tp_f_rest" : new_tp_f_rest,
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self._position_time_parameter = optimizable_tensors["tp_pos"]
        self._rotation_time_parameter = optimizable_tensors["tp_rot"]
        self._features_dc_time_parameter = optimizable_tensors["tp_f_dc"]
        self._features_rest_time_parameter = optimizable_tensors["tp_f_rest"]

        self.xyz_gradient_accum = torch.zeros((self._xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self._xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self._xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self._xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self._xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        new_tp_pos = self._position_time_parameter[selected_pts_mask].repeat(N, 1, 1)
        new_tp_rot = self._rotation_time_parameter[selected_pts_mask].repeat(N, 1, 1)
        new_tp_features_dc = self._features_dc_time_parameter[selected_pts_mask].repeat(N, 1, 1, 1)
        new_tp_features_rest = self._features_rest_time_parameter[selected_pts_mask].repeat(N, 1, 1,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_tp_pos, new_tp_rot, new_tp_features_dc, new_tp_features_rest)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        new_tp_pos = self._position_time_parameter[selected_pts_mask]
        new_tp_rot = self._rotation_time_parameter[selected_pts_mask]
        new_tp_features_dc = self._features_dc_time_parameter[selected_pts_mask]
        new_tp_features_rest = self._features_rest_time_parameter[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tp_pos, new_tp_rot, new_tp_features_dc, new_tp_features_rest)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
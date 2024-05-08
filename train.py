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
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

from scipy.spatial import KDTree
import numpy as np
import numba as nb
import shutil

@nb.njit
def find_top_p_elements(dataaa, p):
    results = np.zeros((dataaa.shape[0], p), dtype=dataaa.dtype)
    count_all = np.zeros((dataaa.shape[0], p), dtype=dataaa.dtype)
    for i in range(dataaa.shape[0]):
        counts = np.bincount(dataaa[i])
        top_5 = np.argsort(counts)[::-1][:p]
        results[i, :] = top_5
        count_all[i, :] = np.sort(counts)[::-1][:p]
    return results, count_all

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    nearest_points_index = torch.empty(0) # knn index list. 
    indices = torch.empty(0)

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    gaussians.setDDDM(opt.dddm_param_len) ### dddm setting
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration, opt.dddm_from_iter)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)                
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        
        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        ### weight decay (lasso) for time parameters. 
        #time_smooth_loss=torch.nanmean(torch.abs(gaussians._position_time_parameter)) 
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        lasso_loss = 0
        time_smooth_loss = 0
        knn_rigid_loss = 0

        if iteration > opt.dddm_from_iter:
            #print(f"{torch.nanmean(torch.abs(gaussians._position_time_parameter) )},{torch.nanmean(torch.abs(gaussians._rotation_time_parameter))} +{torch.nanmean(torch.abs(gaussians._features_dc_time_parameter) )} +{torch.nanmean(torch.abs(gaussians._features_rest_time_parameter) )}") 
            lasso_loss = torch.sum(torch.pow(gaussians._position_time_parameter,2 ) ) +torch.sum(torch.pow(gaussians._rotation_time_parameter,2)) +torch.sum(torch.pow(gaussians._features_dc_time_parameter, 2) ) #+torch.nanmean(torch.abs(gaussians._features_rest_time_parameter) ) 
            if lasso_loss >0:
                loss +=opt.lambda_lasso * lasso_loss
            time_smooth_loss = gaussians.get_time_smooth_loss(scene.scene_info.time_delta /10)
            if time_smooth_loss > 0: # sqrt(0) has inf on its gradient, so it makes Nan at backpropagation step.
                loss += time_smooth_loss
        

        if iteration == opt.densify_until_iter: # find near neighbors for calculate knn loss. it calcuated only once on here.
            data = gaussians.get_xyz.to(torch.device('cpu')).clone().detach().numpy()
            tree = KDTree(data)
            distances, indices = tree.query(data, k=opt.knn_param +1) 
            indices = torch.tensor(indices[:,1:]) # first result of KDTree.query is itself, so remove it.
        
        if iteration > opt.densify_until_iter and iteration < opt.knn_until_iter:
            gaussian_means = gaussians.get_xyz.to(torch.device('cpu'))
            points = gaussian_means[:,None,:].repeat(1,opt.knn_param, 1)
            if points.shape[0] != indices.shape[0] or points.shape[1] != indices.shape[1]:
                print("Error occur on knn rigid_loss")
            near_points =gaussians.get_xyz.to(torch.device('cpu'))[indices.squeeze()].reshape(points.shape)
            if points.shape != near_points.shape:
                print("Error occur on knn rigid_loss")
            #knn_rigid_loss =torch.sum(torch.sqrt( (near_points - points) **2  ))
            
            powersum = torch.sum((near_points - points) **2  ,dim=-1)
            knn_rigid_loss = torch.sum(torch.sqrt( powersum[torch.where(powersum > 0)] ))
            if knn_rigid_loss >0:
                loss+= opt.lambda_knn * knn_rigid_loss.to(loss.device)
        
        """ 
        if iteration > opt.densify_until_iter and iteration < opt.knn_until_iter:
        #if iteration > 1000:
            #gaussians.setTime(viewpoint_cam.time + scene.scene_info.time_delta * 0.5)
            d_xyz = gaussians.D_xyz().to(torch.device('cpu')) # N , 3, 3,
            d_rotation = gaussians.D_rotation().to(torch.device('cpu'))# N , 4, 3
            d_feature = gaussians.D_features().to(torch.device('cpu')).reshape((d_xyz.shape[0], -1))# N 1 3

            #print(d_xyz.shape)
            #print(d_rotation.shape)
            #print(d_feature.shape)

            d_all = torch.cat([d_xyz, d_rotation, d_feature], dim=1 )    
            #print(d_all.shape)
            #return
            origin_all = d_all[:,None,:].repeat(1,opt.knn_param, 1)
            near_all = d_all[indices.numpy().squeeze()]
            #print(origin_all.shape)
            #print(near_all.shape)
            near_all = near_all.reshape(origin_all.shape)

            powersum = torch.sum((origin_all - near_all) **2  ,dim=-1)
            
            knn_rigid_loss = torch.sum(torch.sqrt( powersum[torch.where(powersum > 0)] ))
            if knn_rigid_loss >0:
                loss+= opt.lambda_knn * knn_rigid_loss.to(loss.device)
            #gaussians.setTime(viewpoint_cam.time)
        
        

        knn_rigid_loss = 0
        if iteration == opt.densify_until_iter:
            current_time = gaussians.time
            indices_all = []
            for t in np.arange(0.1, 1, 0.2):
                gaussians.setTime(t)
                data = gaussians.get_xyz.to(torch.device('cpu')).clone().detach().numpy() #torch.tensor(gaussians.get_xyz, device='cpu').numpy()
                tree = KDTree(data)
                distances, indices = tree.query(data, k=opt.knn_param +1) # 논문에 안적혀있다... 임의로 7이라 정의
                indices_all.append(indices[:,1:]) # 가장 첫 번째 항은 자기자신. 따라서 배제.
            
            indices_np = np.stack(indices_all)
            indices_np = np.transpose(indices_np, (1, 0, 2))
            indices_np = indices_np.reshape((indices_np.shape[0], -1))
            nearest_points_index_np, counts = find_top_p_elements(indices_np , opt.knn_param)
            nearest_points_index = torch.tensor(nearest_points_index_np)

            gaussians.time = current_time

        if iteration > opt.densify_until_iter and iteration < opt.knn_until_iter:
            gaussian_means = gaussians.get_xyz.to(torch.device('cpu')).clone().detach()
            points = gaussian_means[:,None,:].repeat(1,opt.knn_param, 1) #torch.tensor(gaussians.get_xyz, device='cpu')[:,None,:].repeat(1,opt.knn_param, 1)
            if points.shape[0] != nearest_points_index.shape[0] or points.shape[1] != nearest_points_index.shape[1]:
                print("Error occur on knn rigid_loss")
            #print(torch.tensor(gaussians.get_xyz, device='cpu').shape)
            #print(indices.shape)
            #print(torch.tensor(gaussians.get_xyz, device='cpu')[indices.squeeze()].shape)
            near_points =torch.tensor(gaussians.get_xyz, device='cpu')[nearest_points_index.squeeze()].reshape(points.shape)
            #near_points = torch.tensor([self.get_xyz[index] for index in self._indices])
            if points.shape != near_points.shape:
                print("Error occur on knn rigid_loss")
            knn_rigid_loss =torch.sum(torch.sqrt( torch.sum((near_points - points) **2  ,dim=-1) ))
           
            if knn_rigid_loss >0:
                loss+= opt.lambda_knn * knn_rigid_loss.to(loss.device)
        """
        cur_loss= loss.item()
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()
            
            if iteration % 1000 == 0:
                print("")
                print(f"current loss : {(1.0 - opt.lambda_dssim)} * {Ll1} + {opt.lambda_dssim} * {(1.0 - ssim(image, gt_image))} + {opt.lambda_lasso * lasso_loss}+{time_smooth_loss} +{knn_rigid_loss}= {cur_loss}")
                print(f"gaussian params :{torch.sum(torch.abs(gaussians._position_time_parameter))}, {torch.sum(torch.abs(gaussians._rotation_time_parameter))}")
                print(f"time : {viewpoint_cam.time}, {gaussians.time}")
                print(f"lambdas : {gaussians._lambda_s}, {gaussians._lambda_b}")

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), gaussians.get_xyz.shape[0], time_smooth_loss, knn_rigid_loss)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                gaussians._lambda_optimizer.step()
                gaussians._lambda_optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

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

    # copy args file to output path
    shutil.copyfile("./arguments/__init__.py", args.model_path + "/arguments.py")


    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, num_gaussian, time_smooth_loss, knn_rigid_loss):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/time_smooth_loss', time_smooth_loss, iteration)
        tb_writer.add_scalar('train_loss_patches/knn_rigid_loss', knn_rigid_loss, iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('num_gaussian', num_gaussian, iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

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
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
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

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=list(range(1000, 30001, 1000)))
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1000,10_000, 20_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[10_000, 20_000, 30_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")

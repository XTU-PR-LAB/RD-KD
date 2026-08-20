
from __future__ import absolute_import, division, print_function

import cv2
import numpy as np
import time
import os

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ChainedScheduler
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from tensorboardX import SummaryWriter

import json

from litenetworks.distiller_metric import HeteroFeatureDistiller
from utils import *
from kitti_utils import *
from layers import *
from loss_functions import kd_loss
from layers import disp_to_depth
from utils import readlines
from options_lite import LiteMonoOptions
import datasets
import litenetworks
cv2.setNumThreads(0)


splits_dir = os.path.join(os.path.dirname(__file__), "splits")

import datasets
import litenetworks
from IPython import embed
from linear_warmup_cosine_annealing_warm_restarts_weight_decay import ChainedScheduler

class Trainermetric:
    def __init__(self, options):
        print("Initializing Trainer")
        self.opt = options
        self.rank = getattr(self.opt, 'rank', 0)
        self.world_size = getattr(self.opt, 'world_size', 1)
        self.is_distributed = self.world_size > 1
        self.is_main_process = self.rank == 0
        
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]},
            'vitt': {'encoder': 'vitt', 'features': 32, 'out_channels': [16, 32, 64, 128]}
        }

        self.models = {}
        
        self.device = torch.device(f"cuda:{self.rank}" if torch.cuda.is_available() and not self.opt.no_cuda else "cpu")

        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])

        if self.opt.use_stereo:
            self.opt.frame_ids.append("s")
        
        self.teacher_model = torch.hub.load('/home/aoao/n417/zhaoyajuan/monodepth2-master/', 'metric3d_vit_small', pretrain=True,
                           source='local')


        for p in self.teacher_model.parameters():
            p.requires_grad = False
        self.teacher_model.to(self.device).eval()
        self.models["encoder"] = litenetworks.LiteMono(model=self.opt.model,
                                                   drop_path_rate=self.opt.drop_path,
                                                   width=self.opt.width, height=self.opt.height)
        self.models["encoder"].to(self.device)
        self.models["depth"] = litenetworks.DepthDecoder(
            self.models["encoder"].num_ch_enc, self.opt.scales)
        self.models["depth"].to(self.device)


        t_hw = (self.opt.teacher_height // 14, self.opt.teacher_width // 14)
        self.FeatureDistiller = HeteroFeatureDistiller().to(self.device)
        

        if self.use_pose_net:
            if self.opt.pose_model_type == "separate_resnet":
                self.models["pose_encoder"] = litenetworks.ResnetEncoder(
                    self.opt.num_layers,
                    self.opt.weights_init == "pretrained",
                    num_input_images=self.num_pose_frames)
                self.models["pose_encoder"].to(self.device)

                self.models["pose"] = litenetworks.PoseDecoder(
                    self.models["pose_encoder"].num_ch_enc,
                    num_input_features=1,
                    num_frames_to_predict_for=2)
                self.models["pose"].to(self.device)

            elif self.opt.pose_model_type == "shared":
                self.models["pose"] = litenetworks.PoseDecoder(
                    self.models["encoder"].num_ch_enc, self.num_pose_frames)
                self.models["pose"].to(self.device)

            elif self.opt.pose_model_type == "posecnn":
                self.models["pose"] = litenetworks.PoseCNN(
                    self.num_input_frames if self.opt.pose_model_input == "all" else 2)

            self.models["pose"].to(self.device)
                

        if self.opt.load_weights_folder is not None:
            self.load_model()
        if self.opt.mypretrain is not None:
            self.load_pretrain()
        if self.is_distributed:
            self.models["encoder"] = DDP(self.models["encoder"], device_ids=[self.rank],
                                         output_device=self.rank, broadcast_buffers=False,
                                         find_unused_parameters=True)
            self.models["depth"] = DDP(self.models["depth"], device_ids=[self.rank], output_device=self.rank,
                                       find_unused_parameters=True)
        if self.is_distributed:
            for key in ["encoder", "pose_encoder"]:
                if key in self.models:
                    self.models[key] = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.models[key])
        if self.is_distributed:
            self.FeatureDistiller = DDP(self.FeatureDistiller, device_ids=[self.rank], output_device=self.rank,
                                        find_unused_parameters=True)


        if self.is_distributed:
            self.models["pose_encoder"] = DDP(self.models["pose_encoder"], device_ids=[self.rank],
                                              output_device=self.rank, broadcast_buffers=False,
                                              find_unused_parameters=True)
            self.models["pose"] = DDP(self.models["pose"], device_ids=[self.rank], output_device=self.rank,
                                      find_unused_parameters=True)

        self.parameters_to_train = []
        self.parameters_to_train_pose = []
        for key in ["encoder", "depth","FeatureDistiller"]:
            if key in self.models:
                self.parameters_to_train += list(self.models[key].parameters())
        for key in [ "pose_encoder", "pose"]:
            if key in self.models:
                self.parameters_to_train_pose += list(self.models[key].parameters())

        self.model_optimizer = optim.AdamW(self.parameters_to_train, self.opt.lr[0], weight_decay=self.opt.weight_decay)
        
        if self.use_pose_net:
            self.model_pose_optimizer = optim.AdamW(self.parameters_to_train_pose, self.opt.lr[3], weight_decay=self.opt.weight_decay)
        
        self.model_lr_scheduler = ChainedScheduler(
            self.model_optimizer,
            T_0=int(self.opt.lr[2]),
            T_mul=1,
            eta_min=self.opt.lr[1],
            last_epoch=-1,
            max_lr=self.opt.lr[0],
            warmup_steps=1,
            gamma=0.8
        )
        
        if self.use_pose_net :
            self.model_pose_lr_scheduler = ChainedScheduler(
                self.model_pose_optimizer,
                T_0=int(self.opt.lr[5]),
                T_mul=1,
                eta_min=self.opt.lr[4],
                last_epoch=-1,
                max_lr=self.opt.lr[3],
                warmup_steps=1,
                gamma=0.8
            )


        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)

        datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                         "kitti_odom": datasets.KITTIOdomDataset,
                         "kitti_depth": datasets.KITTIDepthDataset
                         }
        self.dataset = datasets_dict[self.opt.dataset]

        fpath_train = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")
        fpath_val = os.path.join(os.path.dirname(__file__), "splits", self.opt.eval_split, "{}_files.txt")
        train_filenames = readlines(fpath_train.format("train"))
        val_filenames = readlines(fpath_val.format("test"))
        img_ext = '.png' if self.opt.png else '.jpg'
        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

        train_dataset = self.dataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=True, img_ext=img_ext)
        
        if self.is_distributed:
            train_sampler = DistributedSampler(train_dataset, shuffle=True)
            self.train_loader = DataLoader(
                train_dataset, self.opt.batch_size, sampler=train_sampler,
                num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        else:
            self.train_loader = DataLoader(
                train_dataset, self.opt.batch_size, True,
                num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        
        val_dataset = self.dataset(
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=False, img_ext=img_ext)
        
        if self.is_distributed:
            val_sampler = DistributedSampler(val_dataset, shuffle=False)
            self.val_loader = DataLoader(
                val_dataset, self.opt.batch_size, sampler=val_sampler,
                num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        else:
            self.val_loader = DataLoader(
                val_dataset, self.opt.batch_size, True,
                num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        self.val_iter = iter(self.val_loader)

        self.writers = {}
        if self.is_main_process:
            for mode in ["train", "val"]:
                self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        self.val_history = {
            'epochs': [],
            'abs_rel': [],
            'delta1': []
        }
        self.best_delta1 = 0.0
        self.best_delta1_epoch = 0

        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))

        self.save_opts()

    def set_train(self):
        for m in self.models.values():
            m.train()

    def set_eval(self):
        for m in self.models.values():
            m.eval()

    def train(self):
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs):
            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()

    def run_epoch(self):
        self.model_lr_scheduler.step()
        if hasattr(self, 'model_pose_lr_scheduler'):
            self.model_pose_lr_scheduler.step()
        
        if self.is_distributed:
            self.train_loader.sampler.set_epoch(self.epoch)

        if self.is_main_process:
            print(f"Training epoch {self.epoch}")
        
        self.set_train()

        for batch_idx, inputs in enumerate(self.train_loader):

            before_op_time = time.time()
            outputs, losses = self.process_batch(inputs)

            self.model_optimizer.zero_grad()
            self.model_pose_optimizer.zero_grad()
            
            losses["loss"].backward()
            
            grad_stats = {}
            for name, model in self.models.items():
                grad_norms = []
                for param in model.parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.data.norm(2).item()
                        grad_norms.append(grad_norm)
                if grad_norms:
                    grad_stats[name] = {
                        'avg': np.mean(grad_norms),
                        'max': np.max(grad_norms),
                        'min': np.min(grad_norms)
                    }
                else:
                    grad_stats[name] = None
            
            self.model_optimizer.step()
            self.model_pose_optimizer.step()

            duration = time.time() - before_op_time

            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 4000
            late_phase = self.step % 1000 == 0

            if early_phase or late_phase:
                self.log_time(batch_idx, duration, losses["loss"].cpu().data,losses["self_loss"].cpu().data,
                              losses["kd_loss"].cpu().data,losses["feature_loss"].cpu().data,losses["depth_loss"].cpu().data,
                              grad_stats)
                
                if "depth_gt" in inputs:
                    self.compute_depth_losses(inputs, outputs, losses)

                self.log("train", inputs, outputs, losses)
                self.val()

            self.step += 1

    def process_batch(self, inputs):
        for key, ipt in inputs.items():
            if not torch.is_tensor(ipt):
                continue
            inputs[key] = ipt.to(self.device)

        features = self.models["encoder"](inputs["color_aug", 0, 0])
        outputs = self.models["depth"](features)
        outputs["student_features"]=features


        if self.use_pose_net:
            outputs.update(self.predict_poses(inputs, features))
        with torch.no_grad():
            from torch.cuda.amp import autocast
            with autocast(enabled=True):
                teacher_pred, confidence, output_dict = self.teacher_model(
                    {'input': inputs[("colorteacher_aug", 0, 0)]}
                )
                teacher_features = output_dict['features']
                teacher_pred_contrast, _, _ = self.teacher_model(
                    {'input': inputs[("teacher_contrast", 0, 0)]}
                )
            augmentation_consistency_mask = self.teacher_self_consistency(
                teacher_pred, teacher_pred_contrast
            )
            outputs["teacher_pred"] = teacher_pred
            outputs["teacher_features"] = teacher_features
            conf_standardized = torch.sigmoid(confidence)
            outputs['metric3d_confidence_mask'] = conf_standardized
            outputs["teacher_augmentation_consistency_mask"] = augmentation_consistency_mask

        self.generate_images_pred(inputs, outputs)
        self.generate_teacher_images_pred(inputs, outputs)

        c = outputs["metric3d_confidence_mask"]
        s = outputs["teacher_augmentation_consistency_mask"]
        p = outputs["teacher_photometric_consistency_mask"]

        c = torch.clamp(c, min=0.0, max=1.0)
        s = torch.clamp(s, min=0.0, max=1.0)
        p = torch.clamp(p, min=0.0, max=1.0)


        mu = (c + s + p) / 3.0

        variance = ((c - mu) ** 2 + (s - mu) ** 2 + (p - mu) ** 2)

        lambda_var = 0.2

        vpec_mask = torch.exp(-lambda_var * variance)


        outputs["conf_teacher_mask"] = vpec_mask

        losses = self.compute_losses(inputs, outputs)

        return outputs, losses
    def teacher_self_consistency(self, teacher_pred, teacher_pred_contrast, eps=1e-6):
        if teacher_pred.shape != teacher_pred_contrast.shape:
            teacher_pred_contrast = F.interpolate(
                teacher_pred_contrast, size=teacher_pred.shape[-2:], mode='bilinear', align_corners=False
            )

        valid_mask = (teacher_pred > 1e-3) & (teacher_pred_contrast > 1e-3)

        diff = torch.abs(teacher_pred - teacher_pred_contrast)
        denom = teacher_pred + teacher_pred_contrast + eps
        rel_diff = diff / denom
        rel_diff = rel_diff.clamp(0, 1)

        conf = 1 - 0.2 * rel_diff

        conf = conf * valid_mask.float()

        return conf

    def predict_poses(self, inputs, features):
        outputs = {}
        if self.num_pose_frames == 2:

            if self.opt.pose_model_type == "shared":
                pose_feats = {f_i: features[f_i] for f_i in self.opt.frame_ids}
            else:
                pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}

            for f_i in self.opt.frame_ids[1:]:
                if f_i != "s":
                    if f_i < 0:
                        pose_inputs = [pose_feats[f_i], pose_feats[0]]
                    else:
                        pose_inputs = [pose_feats[0], pose_feats[f_i]]

                    if self.opt.pose_model_type == "separate_resnet":
                        pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                    elif self.opt.pose_model_type == "posecnn":
                        pose_inputs = torch.cat(pose_inputs, 1)

                    axisangle, translation = self.models["pose"](pose_inputs)
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation

                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

        else:
            if self.opt.pose_model_type in ["separate_resnet", "posecnn"]:
                pose_inputs = torch.cat(
                    [inputs[("color_aug", i, 0)] for i in self.opt.frame_ids if i != "s"], 1)

                if self.opt.pose_model_type == "separate_resnet":
                    pose_inputs = [self.models["pose_encoder"](pose_inputs)]

            elif self.opt.pose_model_type == "shared":
                pose_inputs = [features[i] for i in self.opt.frame_ids if i != "s"]

            axisangle, translation = self.models["pose"](pose_inputs)

            for i, f_i in enumerate(self.opt.frame_ids[1:]):
                if f_i != "s":
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, i], translation[:, i])

        return outputs

    def val(self):
        self.set_eval()
        try:
            inputs = next(self.val_iter)
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = next(self.val_iter)

        with torch.no_grad():
            outputs, losses = self.process_batch(inputs)

            if "depth_gt" in inputs:
                self.compute_depth_losses(inputs, outputs, losses)

            self.log("val", inputs, outputs, losses)

            if self.is_main_process:
                metric_display_names = {
                        'de/abs_rel': 'abs_rel',
                        'de/sq_rel': 'sq_rel',
                        'de/rms': 'rms',
                        'de/log_rms': 'log_rms',
                        'da/a1': 'δ < 1.25',
                        'da/a2': 'δ < 1.25²',
                        'da/a3': 'δ < 1.25³'
                }
                    
                for metric in self.depth_metric_names:
                    if metric in losses:
                        value = losses[metric].item() if isinstance(losses[metric], torch.Tensor) else losses[metric]
                        display_name = metric_display_names.get(metric, metric)
                        print(f"  {display_name:20s}: {value:.6f}")
                
            del inputs, outputs, losses

        self.set_train()
    
    def full_val(self):
        self.set_eval()
        
        total_metrics = {metric: 0.0 for metric in self.depth_metric_names}
        num_batches = 0
        
        print("\n====== Performing full validation ======")
        
        with torch.no_grad():
            for inputs in self.val_loader:
                num_batches += 1
                
                outputs, losses = self.process_batch(inputs)
                
                if "depth_gt" in inputs:
                    self.compute_depth_losses(inputs, outputs, losses)
                    
                    for metric in self.depth_metric_names:
                        if metric in losses:
                            value = losses[metric].item() if isinstance(losses[metric], torch.Tensor) else losses[metric]
                            total_metrics[metric] += value
                
                del inputs, outputs, losses
        
        if self.is_main_process:
            metric_display_names = {
                    'de/abs_rel': 'abs_rel',
                    'de/sq_rel': 'sq_rel',
                    'de/rms': 'rms',
                    'de/log_rms': 'log_rms',
                    'da/a1': 'δ < 1.25',
                    'da/a2': 'δ < 1.25²',
                    'da/a3': 'δ < 1.25³'
            }
                
            print(f"\nEpoch {self.epoch} Full Validation Results:")
            print("=" * 40)
            
            current_abs_rel = 0.0
            current_delta1 = 0.0
            
            for metric, total_value in total_metrics.items():
                if num_batches > 0:
                    avg_value = total_value / num_batches
                    display_name = metric_display_names.get(metric, metric)
                    print(f"  {display_name:20s}: {avg_value:.6f}")
                    
                    if metric == 'de/abs_rel':
                        current_abs_rel = avg_value
                    elif metric == 'da/a1':
                        current_delta1 = avg_value
            print("=" * 40)
            
            self.val_history['epochs'].append(self.epoch)
            self.val_history['abs_rel'].append(current_abs_rel)
            self.val_history['delta1'].append(current_delta1)
            
            if current_delta1 > self.best_delta1:
                self.best_delta1 = current_delta1
                self.best_delta1_epoch = self.epoch
            
            print("\n====== Historical Validation Metrics ======")
            print(f"Number of validations performed: {len(self.val_history['epochs'])}")
            print(f"Current epoch's abs_rel: {current_abs_rel:.6f}")
            print(f"Current epoch's delta1: {current_delta1:.6f}")
            print(f"Best delta1: {self.best_delta1:.6f} (at epoch {self.best_delta1_epoch})")
            
            print("\nAll historical abs_rel values:")
            for i, (epoch, abs_rel, delta1) in enumerate(zip(
                    self.val_history['epochs'],
                    self.val_history['abs_rel'],
                    self.val_history['delta1'])):
                print(f"  Epoch {epoch}: abs_rel={abs_rel:.6f}, delta1={delta1:.6f}")
            
            print("=" * 40)
            
           
        self.set_train()

    def generate_images_pred(self, inputs, outputs):
        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                disp = F.interpolate(
                    disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                source_scale = 0

            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

            outputs[("depth", 0, scale)] = depth

            for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    T = outputs[("cam_T_cam", 0, frame_id)]

                if self.opt.pose_model_type == "posecnn":

                    axisangle = outputs[("axisangle", 0, frame_id)]
                    translation = outputs[("translation", 0, frame_id)]

                    inv_depth = 1 / depth
                    mean_inv_depth = inv_depth.mean(3, True).mean(2, True)

                    T = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0)

                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)

                outputs[("sample", frame_id, scale)] = pix_coords

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border")

                if not self.opt.disable_automasking:
                    outputs[("color_identity", frame_id, scale)] = \
                        inputs[("color", frame_id, source_scale)]
    def generate_teacher_images_pred(self, inputs, outputs):
        depth = outputs["teacher_pred"]
        depth = F.interpolate(depth, [self.opt.height, self.opt.width], mode='bilinear')


        for i, frame_id in enumerate(self.opt.frame_ids[1:]):
            T = outputs[("cam_T_cam", 0, frame_id)]


            cam_points = self.backproject_depth[0](
                depth, inputs[("inv_K", 0)])
            pix_coords = self.project_3d[0](
                cam_points, inputs[("K", 0)], T)

            outputs[("teacher_sample", frame_id, 0)] = pix_coords

            outputs[("teacher_color", frame_id, 0)] = F.grid_sample(
                inputs[("color", frame_id, 0)],
                outputs[("teacher_sample", frame_id, 0)],
                padding_mode="border")
        teacher_reprojection_losses=[]
        target = inputs[("color", 0, 0)]
        for frame_id in self.opt.frame_ids[1:]:
            pred = outputs[("teacher_color", frame_id, 0)]
            teacher_reprojection_losses.append(self.compute_reprojection_loss(pred, target))
        teacher_reprojection = torch.cat(teacher_reprojection_losses, 1)
        teacher_reprojection_min, _ = torch.min(teacher_reprojection, dim=1, keepdim=True)
        E = teacher_reprojection_min
        E_min = E.amin(dim=[1, 2, 3], keepdim=True)
        E_max = E.amax(dim=[1, 2, 3], keepdim=True)
        E_norm = (E - E_min) / (E_max - E_min + 1e-6)
        R = 1 - 0.2 * E_norm
        teacher_photometric_conf = R
        teacher_photometric_conf= F.interpolate(
            teacher_photometric_conf,
            size=(self.opt.teacher_height, self.opt.teacher_width),
            mode="nearest"
            )
        outputs["teacher_photometric_consistency_mask"]=teacher_photometric_conf

    def compute_reprojection_loss(self, pred, target):
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.8 * ssim_loss + 0.2 * l1_loss

        return reprojection_loss

    def compute_losses(self, inputs, outputs):
        losses = {}
        total_loss = 0
        for scale in self.opt.scales:
            loss = 0
            reprojection_losses = []

            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0

            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]

            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))

            reprojection_losses = torch.cat(reprojection_losses, 1)

            if not self.opt.disable_automasking:
                identity_reprojection_losses = []
                for frame_id in self.opt.frame_ids[1:]:
                    pred = inputs[("color", frame_id, source_scale)]
                    identity_reprojection_losses.append(
                        self.compute_reprojection_loss(pred, target))

                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

                if self.opt.avg_reprojection:
                    identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                else:
                    identity_reprojection_loss = identity_reprojection_losses


            if self.opt.avg_reprojection:
                reprojection_loss = reprojection_losses.mean(1, keepdim=True)
            else:
                reprojection_loss = reprojection_losses

            if not self.opt.disable_automasking:
                identity_reprojection_loss += torch.randn(
                    identity_reprojection_loss.shape, device=self.device) * 0.00001

                combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
            else:
                combined = reprojection_loss

            if combined.shape[1] == 1:
                to_optimise = combined
            else:
                to_optimise, idxs = torch.min(combined, dim=1)

            if not self.opt.disable_automasking:
                identity_mask = (idxs > identity_reprojection_loss.shape[1] - 1).float()
                outputs["identity_selection/{}".format(scale)] = identity_mask
                if scale == 0:
                    outputs["student_identity_selection_inverse_mask"] = 1 - identity_mask
                reprojection_loss,_=torch.min(reprojection_loss, dim=1)
                selected_photo = reprojection_loss[identity_mask > 0.5]
                q1 = torch.quantile(selected_photo, 0.3)
                q2 = torch.quantile(selected_photo, 0.7)
                norm = (reprojection_loss - q1) / (q2 - q1 + 1e-6)
                weight = torch.clamp(norm, 0.9, 1)
                outputs["student_photometric_difficulty_mask"] = weight * identity_mask
                outputs["student_mixed_difficulty_mask"]=outputs["student_identity_selection_inverse_mask"]+outputs["student_photometric_difficulty_mask"]

            loss += to_optimise.mean()
            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        total_loss /= self.num_scales

        student_features = outputs["student_features"]
        teacher_features = outputs["teacher_features"]
        feature_loss = self.FeatureDistiller(student_features, teacher_features)
        kdloss = kd_loss(outputs,losses,inputs,feature_loss,depth_loss_para=self.opt.depth_loss_para,feature_loss_para=self.opt.feature_loss_para)

        losses["self_loss"]=total_loss
        losses["feature_loss"] = feature_loss
        losses["kd_loss"] = kdloss
        losses["loss"] = total_loss + kdloss


        return losses


    def compute_depth_losses(self, inputs, outputs, losses):
        depth_pred = outputs["teacher_pred"]
        depth_pred = outputs[("disp",0)]

        depth_pred = F.interpolate(depth_pred, [375, 1242], mode="bilinear", align_corners=False)[0]
        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt"][0]
        mask = depth_gt > 0

        crop_mask = torch.zeros_like(mask)
        crop_mask[ :, 153:371, 44:1197] = 1
        mask = mask * crop_mask

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        epsilon = 1e-6
        inverse_truth = 1.0 / (depth_gt + epsilon)
        A = torch.stack([depth_pred, torch.ones_like(depth_pred)], dim=-1)

        scale, shift = torch.linalg.lstsq(A, inverse_truth, rcond=None)[0]
        depth_pred = 1.0 / (depth_pred * scale + shift + epsilon)


        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = depth_errors[i].cpu()

    def log_time(self, batch_idx, duration, loss, self_loss,kd_loss,feature_loss,depth_loss,grad_stats):
        if self.is_main_process:
            samples_per_sec = self.opt.batch_size / duration
            time_sofar = time.time() - self.start_time
            training_time_left = (
                self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
            
            print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
                " | loss: {:.5f} | self_loss: {:.5f} | kd_loss: {:.5f} | feature_loss: {:.5f} | depth_loss: {:.5f} |time elapsed: {} | time left: {}"
            print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,self_loss,kd_loss,feature_loss,depth_loss,
                                      sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))
            
            current_lr = self.model_optimizer.param_groups[0]['lr']
            print(f"learn_rate: {current_lr:.8f}")

            print("grad_information:")
            for name, stats in grad_stats.items():
                if stats is not None:
                    print(f"  {name:15s}: avg_grad={stats['avg']:.6f}")
            print("=" * 80)

    def log(self, mode, inputs, outputs, losses):
        if self.is_main_process:
            writer = self.writers[mode]
            for l, v in losses.items():
                writer.add_scalar("{}".format(l), v, self.step)

            for j in range(min(4, self.opt.batch_size)):
                for s in self.opt.scales:
                    for frame_id in self.opt.frame_ids:
                        writer.add_image(
                            "color_{}_{}/{}".format(frame_id, s, j),
                            inputs[("color", frame_id, s)][j].data, self.step)
                        if s == 0 and frame_id != 0:
                            writer.add_image(
                                "color_pred_{}_{}/{}".format(frame_id, s, j),
                                outputs[("color", frame_id, s)][j].data, self.step)

                    writer.add_image(
                        "disp_{}/{}".format(s, j),
                        normalize_image(outputs[("disp", s)][j]), self.step)

                    if self.opt.predictive_mask:
                        for f_idx, frame_id in enumerate(self.opt.frame_ids[1:]):
                            writer.add_image(
                                "predictive_mask_{}_{}/{}".format(frame_id, s, j),
                                outputs["predictive_mask"][("disp", s)][j, f_idx][None, ...],
                                self.step)

                    elif not self.opt.disable_automasking:
                        writer.add_image(
                            "automask_{}/{}".format(s, j),
                            outputs["identity_selection/{}".format(s)][j][None, ...], self.step)

    def save_opts(self):
        if self.is_main_process:
            models_dir = os.path.join(self.log_path, "models")
            if not os.path.exists(models_dir):
                os.makedirs(models_dir)
            to_save = self.opt.__dict__.copy()

            with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
                json.dump(to_save, f, indent=2)

    def save_model(self):
        if self.is_main_process:
            save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)

            for model_name, model in self.models.items():
                save_path = os.path.join(save_folder, "{}.pth".format(model_name))
                if self.is_distributed:
                    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                        to_save = model.module.state_dict()
                    else:
                        to_save = model.state_dict()
                else:
                    to_save = model.state_dict()
                
                if model_name == 'encoder':
                    to_save['height'] = self.opt.height
                    to_save['width'] = self.opt.width
                    to_save['use_stereo'] = self.opt.use_stereo
                torch.save(to_save, save_path)

            save_path = os.path.join(save_folder, "{}.pth".format("adam"))
            torch.save(self.model_optimizer.state_dict(), save_path)

    def load_pretrain(self):
        self.opt.mypretrain = os.path.expanduser(self.opt.mypretrain)
        path = self.opt.mypretrain
        model_dict = self.models["encoder"].state_dict()
        if self.is_distributed and torch.distributed.is_initialized():
            torch.distributed.barrier()
        pretrained_dict = torch.load(path, map_location="cpu")
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict and not k.startswith('norm'))}
        model_dict.update(pretrained_dict)
        self.models["encoder"].load_state_dict(model_dict)
        print('mypretrain loaded.')

    def load_model(self):
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            if self.is_distributed and torch.distributed.is_initialized():
                torch.distributed.barrier()
            pretrained_dict = torch.load(path, map_location="cpu")
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)


        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        optimizer_pose_load_path = os.path.join(self.opt.load_weights_folder, "adam_pose.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            optimizer_pose_dict = torch.load(optimizer_pose_load_path)
            self.model_optimizer.load_state_dict(optimizer_dict)
            self.model_pose_optimizer.load_state_dict(optimizer_pose_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")

import os

import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpmath import eps
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.onnx.symbolic_opset11 import unsqueeze
import torch
import torch.nn as nn

import torch
import torch.nn as nn

import torch
import torch.nn as nn

class HierarchicalNormalizationLoss(nn.Module):
    def __init__(self, scales=(1,2,4), eps=1e-6, tau=0.001):
        super().__init__()
        self.scales = scales
        self.eps = eps

    def normalize_by_segment_soft(self, d, mask, S):
        B, _, H, W = d.shape
        out = torch.zeros_like(d)

        for b in range(B):
            valid = mask[b] > 1e-5
            if valid.sum() < 10:
                continue
            vals = d[b][valid]

            q = torch.linspace(0, 1, S + 1, device=d.device)
            segs = torch.quantile(vals, q)

            medians = []
            mads = []
            seg_masks = []

            for i in range(S):
                low, high = segs[i], segs[i + 1]
                seg_mask = valid & (d[b] >= low) & (d[b] < high)
                if seg_mask.sum() < 5:
                    medians.append(vals.median())
                    mads.append(torch.abs(vals - vals.median()).mean() + self.eps)
                    seg_masks.append(seg_mask)
                    continue

                seg_vals = d[b][seg_mask]
                median = seg_vals.median()
                mad = torch.abs(seg_vals - median).mean() + self.eps

                medians.append(median)
                mads.append(mad)
                seg_masks.append(seg_mask)

            medians = torch.stack(medians)
            mads = torch.stack(mads)

            x = d[b, 0]

            x_expand = x.unsqueeze(0)
            med_expand = medians.view(S, 1, 1)
            mad_expand = mads.view(S, 1, 1)

            z = (x_expand - med_expand) / mad_expand

            dist = torch.abs(x_expand - med_expand)

            global_mad = torch.abs(vals - vals.median()).mean().detach()
            tau_raw = global_mad
            tau = torch.clamp(tau_raw, min=0.02, max=0.2)

            w = torch.softmax(-dist / tau, dim=0)

            out[b, 0] = (w * z).sum(dim=0) * valid.float()


        return out

    def normalize_global(self, d, mask):
        B, _, H, W = d.shape
        out = torch.zeros_like(d)

        for b in range(B):
            valid = mask[b] > 1e-5
            if valid.sum() < 10:
                continue

            vals = d[b][valid]
            median = vals.median()
            mad = torch.abs(vals - median).mean() + self.eps
            out[b][valid] = (vals - median) / mad

        return out


    def forward(self, student_disp, teacher_disp, conf_difficulty_mask, outputs):
        mask = (teacher_disp > 5e-3)
        mask = mask * conf_difficulty_mask

        valid_pixel_cnt = (mask > 0).sum()

        total_loss = 0.0
        cnt = 0

        for S in self.scales:
            t_norm_soft = self.normalize_by_segment_soft(teacher_disp, mask, S)
            s_norm_soft = self.normalize_by_segment_soft(student_disp, mask, S)
            diff_soft = torch.abs(t_norm_soft - s_norm_soft)
            loss = (diff_soft).sum() / (valid_pixel_cnt + self.eps)
            total_loss += loss
            cnt += 1

        return total_loss / cnt


    def build_difficulty_diff(self, diff, mask, low_q=0.1, high_q=0.9, eps=1e-6):
        q1 = torch.quantile(diff, low_q)
        q2 = torch.quantile(diff, high_q)
        norm = (diff - q1) / (q2 - q1 + eps)
        norm_mask = torch.clamp(norm, 0.9,1.05)
        focal_diff = diff* norm_mask * mask
        return norm_mask,focal_diff


def kd_loss(outputs,losses,inputs,feature_loss,depth_loss_para,feature_loss_para):
    student_pred = outputs[('disp', 0)]
    teacher_pred=outputs["teacher_pred"]
    teacher_pred = 1.0 / (teacher_pred + 10e-6)
    student_size = student_pred.shape[2:]
    teacher_pred = F.interpolate(teacher_pred, size=student_size,
                                 mode='bilinear', align_corners=False)

    raw_image=inputs[("color", 0, 0)]


    interp_mask_keys = [
        "conf_teacher_mask",
        "metric3d_confidence_mask",
        "teacher_augmentation_consistency_mask",
        "teacher_photometric_consistency_mask",
        "teacher_pred"
    ]

    for key in interp_mask_keys:
        outputs[key] = F.interpolate(
            outputs[key],
            size=student_size,
            mode="bilinear",
            align_corners=False
        )

    mask=outputs["conf_teacher_mask"]
    depth_distiller = HierarchicalNormalizationLoss()
    depth_loss = depth_distiller(student_pred, teacher_pred,mask,outputs)
    losses["depth_loss"] = depth_loss


    total_loss = 0
    total_loss += depth_loss_para*depth_loss
    total_loss += feature_loss_para*feature_loss
    return total_loss

def disp_to_depth(disp, min_depth=0.1, max_depth=80):
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth


import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap

UNIFIED_CMAP = LinearSegmentedColormap.from_list(
    "skyblue_to_orange",
    [
        "#0B46FF",
        "#007BFF",
        "#1E88E5",
        "#2196F3",
        "#26C6DA",
        "#4DD0E1",
        "#C8E6C9",
        "#FFF9C4",
        "#FFE082",
        "#FFB74D",
        "#FB8C00",
        "#FF7A00",
        "#FF6B00",
        "#f0470a",
        "#A00042"
    ],
    N=256
)

def robust_vmin_vmax(arr, p_low=2, p_high=98):
    vmin, vmax = np.percentile(arr, [p_low, p_high])
    if vmax <= vmin:
        vmin, vmax = float(arr.min()), float(arr.max())
        if vmax <= vmin:
            vmax = vmin + 1e-6
    return vmin, vmax

def robust_vmin_vmax_multi(arr_list, p_low=2, p_high=98):
    flat = np.concatenate([a.reshape(-1) for a in arr_list if a is not None])
    vmin, vmax = np.percentile(flat, [p_low, p_high])
    if vmax <= vmin:
        vmin, vmax = float(flat.min()), float(flat.max())
        if vmax <= vmin:
            vmax = vmin + 1e-6
    return vmin, vmax

def _gaussian_blur_np(img, sigma=1.2):
    if sigma is None or sigma <= 0:
        return img
    radius = int(3 * sigma + 0.5)
    if radius <= 0:
        return img

    x = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-(x * x) / (2 * sigma * sigma))
    k = k / (k.sum() + 1e-12)

    img = img.astype(np.float32)

    pad = radius
    ap = np.pad(img, ((0, 0), (pad, pad)), mode="reflect")
    out = np.zeros_like(img, dtype=np.float32)
    for j in range(img.shape[1]):
        out[:, j] = (ap[:, j:j + 2 * pad + 1] * k[None, :]).sum(axis=1)

    ap = np.pad(out, ((pad, pad), (0, 0)), mode="reflect")
    out2 = np.zeros_like(out, dtype=np.float32)
    for i in range(out.shape[0]):
        out2[i, :] = (ap[i:i + 2 * pad + 1, :] * k[:, None]).sum(axis=0)

    return out2

def _save_vertical(fig_items, raw_np, save_path, conf_blur_sigma=1.4):
    fig_items = list(fig_items)
    fig_items.append(("raw_image", raw_np))
    n = len(fig_items)

    target_ratio = 192 / 640
    row_w = 7
    row_h = row_w * target_ratio

    fig = plt.figure(figsize=(row_w, row_h * n))
    gs = fig.add_gridspec(n, 1)

    for i, (title, arr) in enumerate(fig_items):
        ax = fig.add_subplot(gs[i, 0])
        ax.set_title(title)

        if arr.ndim == 3:
            ax.imshow(arr)
            ax.axis("off")
            ax.set_box_aspect(target_ratio)
            continue

        vmin, vmax = robust_vmin_vmax(arr, p_low=2, p_high=98)

        if title == "conf_teacher_mask":
            arr_show = _gaussian_blur_np(arr, sigma=conf_blur_sigma)
        else:
            arr_show = arr

        im = ax.imshow(arr_show, cmap=UNIFIED_CMAP, vmin=vmin, vmax=vmax)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.2)
        plt.colorbar(im, cax=cax)

        ax.axis("off")
        ax.set_box_aspect(target_ratio)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[Visualize] saved → {save_path}")

def _save_diff_focal_merged(diff_dict, focal_dict, raw_np, save_path, S_list=(1, 2, 4)):
    target_ratio = 192 / 640

    fig_w = 14
    one_col_w = fig_w / 2
    row_h = one_col_w * target_ratio

    n_rows = len(S_list) + 1
    fig = plt.figure(figsize=(fig_w, row_h * n_rows))
    gs = fig.add_gridspec(n_rows, 2, width_ratios=[1, 1])

    for r, S in enumerate(S_list):
        aL = diff_dict.get(S, None)
        aR = focal_dict.get(S, None)

        axL = fig.add_subplot(gs[r, 0])
        axR = fig.add_subplot(gs[r, 1])
        axL.set_title(f"diff_soft_{S}")
        axR.set_title(f"focal_map_{S}")

        if aL is None and aR is None:
            axL.axis("off"); axR.axis("off")
            continue

        base_list = []
        if aL is not None: base_list.append(aL)
        if aR is not None: base_list.append(aR)
        vmin, vmax = robust_vmin_vmax_multi(base_list, p_low=2, p_high=98)

        im_for_cb = None
        if aL is not None:
            axL.imshow(aL, cmap=UNIFIED_CMAP, vmin=vmin, vmax=vmax)
        else:
            axL.text(0.5, 0.5, "Missing", ha="center", va="center")

        if aR is not None:
            im_for_cb = axR.imshow(aR, cmap=UNIFIED_CMAP, vmin=vmin, vmax=vmax)
        else:
            axR.text(0.5, 0.5, "Missing", ha="center", va="center")
            if aL is not None:
                im_for_cb = axL.images[0]

        divider = make_axes_locatable(axR)
        cax = divider.append_axes("right", size="3%", pad=0.2)
        plt.colorbar(im_for_cb, cax=cax)

        axL.axis("off"); axR.axis("off")
        axL.set_box_aspect(target_ratio)
        axR.set_box_aspect(target_ratio)

    ax_raw = fig.add_subplot(gs[-1, :])
    ax_raw.set_title("raw_image")
    ax_raw.imshow(raw_np)
    ax_raw.axis("off")
    ax_raw.set_box_aspect(target_ratio)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[Visualize] saved → {save_path}")

def visualize_all(outputs, raw_image, save_dir="showimgkitti", idx=0):

    if not hasattr(visualize_all, "_counter"):
        if os.path.exists(save_dir):
            for f in os.listdir(save_dir):
                fp = os.path.join(save_dir, f)
                if os.path.isfile(fp):
                    os.remove(fp)
        else:
            os.makedirs(save_dir, exist_ok=True)

        visualize_all._counter = 0
        print(f"[Visualize] Clear dir & start new training → {save_dir}")

    cnt = visualize_all._counter

    raw = raw_image[idx].detach().cpu()
    raw_np = raw.numpy().transpose(1, 2, 0)
    raw_np = (raw_np - raw_np.min()) / (raw_np.max() - raw_np.min() + 1e-6)

    teacher_keys = [
        "conf_teacher_mask",
        "metric3d_confidence_mask",
        "teacher_augmentation_consistency_mask",
        "teacher_photometric_consistency_mask",
        "teacher_pred",
    ]

    teacher_items = []
    for key in teacher_keys:
        if key in outputs:
            arr = outputs[key][idx, 0].detach().cpu().numpy()
            teacher_items.append((key, arr))

    _save_vertical(
        teacher_items,
        raw_np,
        os.path.join(save_dir, f"{cnt:06d}_teacher_masks.png"),
        conf_blur_sigma=1.4
    )

    diff_dict = {}
    focal_dict = {}

    for S in [1, 2, 4]:
        kd = f"teacher_student_normalize_diff_mask_{S}"
        kf = f"focal_diff_{S}"
        if kd in outputs:
            diff_dict[S] = outputs[kd][idx, 0].detach().cpu().numpy()
        if kf in outputs:
            focal_dict[S] = outputs[kf][idx, 0].detach().cpu().numpy()

    _save_diff_focal_merged(
        diff_dict,
        focal_dict,
        raw_np,
        os.path.join(save_dir, f"{cnt:06d}_diff_focal_merged.png"),
        S_list=(1, 2, 4)
    )

    visualize_all._counter += 1



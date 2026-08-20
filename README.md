# Reliability-Boosted and Distribution-Aligned Distillation for Monocular Depth Estimation
This is the official repository of paper *Reliability-Boosted and Distribution-Aligned Distillation for Monocular Depth Estimation*.

## Environment
Follow the installation instructions of [Lite-Mono](https://github.com/noahzn/Lite-Mono). This project uses the same student-side dependencies.

## Dataset
### KITTI
Download training sequences from [KITTI raw data](https://www.cvlibs.net/datasets/kitti/raw_data.php).  
Common zip URLs are listed in `splits/kitti_archives_to_download.txt`.

Expected layout:

```text
kitti_data/
  2011_09_26/
    calib_cam_to_cam.txt
    calib_velo_to_cam.txt
    2011_09_26_drive_0001_sync/
      image_02/data/*.jpg
      image_03/data/*.jpg
      velodyne_points/data/*.bin
  2011_09_28/
  2011_09_29/
  2011_09_30/
  2011_10_03/
```

Images are read as `.jpg` by default. Add `--png` if your files are PNG.

### Ground truth for evaluation
KITTI Eigen evaluation needs `splits/eigen/gt_depths.npz`. Generate it from velodyne the same way as Monodepth2 / Lite-Mono, then place it at:
```text
splits/eigen/gt_depths.npz
```

## Teacher weights
Training loads Metric3D ViT-Small via `torch.hub.load(..., source='local')`.
1. Download the Metric3D checkpoint, e.g.:
```text
https://huggingface.co/JUGGHM/Metric3D/resolve/main/metric_depth_vit_small_800k.pth
```
1. Put it locally, e.g.:
```text
metric3D_weights/metric_depth_vit_small_800k.pth
```
1. Point `ckpt_file` for `ViT-Small` in `hubconf.py` to that path.
2. Update the teacher load path in `trainer_metric.py` to the current repo root, e.g.:
```python
self.teacher_model = torch.hub.load(
    './', 'metric3d_vit_small', pretrain=True, source='local'
)
```

## Training
Entry point: `train_metric.py`. Options are parsed by `LiteMonoOptions` in `options_lite.py`.

Single GPU:

```bash
python train_metric.py \
  --model_name lite-mono-metric \
  --model lite-mono \
  --data_path /path/to/kitti_data \
  --log_dir ./logs \
  --split eigen_zhou \
  --dataset kitti \
  --height 192 \
  --width 640 \
  --teacher_height 392 \
  --teacher_width 672 \
  --batch_size 12 \
  --num_epochs 50 \
  --num_workers 8 \
  --lr 1e-4 5e-6 31 1e-4 1e-5 31 \
  --weight_decay 1e-2 \
  --drop_path 0.2 \
  --depth_loss_para 0.1 \
  --feature_loss_para 0.1
```

Multi-GPU: `mp.spawn` starts automatically when more than one GPU is visible. You can also use `torchrun`:

```bash
torchrun --nproc_per_node=2 train_metric.py \
  --model_name lite-mono-metric \
  --data_path /path/to/kitti_data \
  --log_dir ./logs \
  --split eigen_zhou
```

Common options:


| Argument                | Description                                                         |
| ----------------------- | ------------------------------------------------------------------- |
| `--model`               | `lite-mono` / `lite-mono-small` / `lite-mono-tiny` / `lite-mono-8m` |
| `--split`               | training split, default `eigen_zhou`                                |
| `--eval_split`          | validation split, typically `eigen` for KITTI                       |
| `--png`                 | load PNG images                                                     |
| `--mypretrain`          | Lite-Mono encoder pretrained weights                                |
| `--load_weights_folder` | resume from a checkpoint                                            |
| `--use_stereo`          | stereo supervision (default is monocular adjacent frames)           |


Logs and weights are written to:

```text
{log_dir}/{model_name}/
  train/          # tensorboard
  val/
  models/
    opt.json
    weights_{epoch}/
      encoder.pth
      depth.pth
      pose_encoder.pth
      pose.pth
      adam.pth
```

TensorBoard:

```bash
tensorboard --logdir ./logs
```

## Evaluation
Entry point: `evaluate_lite_depth.py`. Evaluates the student on the KITTI Eigen split.

```bash
python evaluate_lite_depth.py \
  --data_path /path/to/kitti_data \
  --load_weights_folder ./logs/lite-mono-metric/models/weights_49 \
  --eval_split eigen \
  --model lite-mono \
  --eval_mono
```

Reported metrics: `abs_rel / sq_rel / rmse / rmse_log / a1 / a2 / a3`.

The script loads `{load_weights_folder}/encoder.pth` and `depth.pth`. Input resolution follows the `height` / `width` stored in the checkpoint.

## Optional Metric3D inference
You can also run the teacher on its own. See the example at the bottom of `hubconf.py`, or:
```bash
python mono/tools/test_scale_cano.py \
  mono/configs/HourglassDecoder/vit.raft5.small.py \
  --load-from metric3D_weights/metric_depth_vit_small_800k.pth \
  --test_data_path /path/to/images \
  --show-dir ./show_dirs \
  --launcher None
```

## Project layout
```text
.
├── train_metric.py          # training entry
├── trainer_metric.py        # training loop, losses, distillation
├── evaluate_lite_depth.py   # KITTI evaluation
├── options_lite.py          # command-line options
├── loss_functions.py        # depth distillation loss
├── hubconf.py               # Metric3D torch.hub interface
├── datasets/                # KITTI / NYU loaders
├── litenetworks/            # Lite-Mono student
├── mono/                    # Metric3D teacher
│   ├── configs/
│   ├── model/
│   └── tools/
└── splits/                  # KITTI split files
```
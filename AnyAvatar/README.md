# AnyAvatar: High-Fidelity Gaussian Head Avatars under Uncalibrated Camera Settings


## Setup

```shell
git clone https://github.com/syncanimation/AnyAvatar.github.io.git --recursive
cd AnyAvatar/AnyAvatar

conda create --name AnyAvatar -y python=3.10
conda activate AnyAvatar

pip install -r requirements.txt
```

## Download
Our code and the pre-processed data relies on FLAME 2023. Please download [original assets](https://flame.is.tue.mpg.de/download.php) to the following paths:

- FLAME 2023 (versions w/ jaw rotation) -> `flame_model/assets/flame/flame2023.pkl`
- FLAME Vertex Masks -> `flame_model/assets/flame/FLAME_masks.pkl`

## Usage

### 1. Rendering

```shell
python render.py \
  -s /path/to/0048 \
  -m output/emo3d/0048 \
  --iteration 600000 \
  --skip_train --skip_val \
  --select_camera_id 8

python render_with_optim.py \
  -s /path/to/0048 \
  -m output/emo3d/0048 \
  --iteration 600000 \
  --skip_train --skip_val \
  --select_camera_id 8 \
  --optim_nvs_pose
```

`--optim_nvs_pose`: fine-tunes camera poses on val/test views before rendering and evaluating image quality.


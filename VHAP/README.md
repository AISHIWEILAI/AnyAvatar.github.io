# AnyAvatar: High-Fidelity Gaussian Head Avatars under Uncalibrated Camera Settings (VHAP part)

## Setup

```shell
git clone git@github.com:syncanimation/AnyAvatar.github.io.git
cd AnyAvatar/VHAP

conda create --name VHAP -y python=3.10
conda activate VHAP

# Install CUDA and ninja for compilation
conda install -c "nvidia/label/cuda-12.1.1" cuda-toolkit ninja cmake  # use the right CUDA version
ln -s "$CONDA_PREFIX/lib" "$CONDA_PREFIX/lib64"  # to avoid error "/usr/bin/ld: cannot find -lcudart"
conda env config vars set CUDA_HOME=$CONDA_PREFIX  # for compilation

# Install PyTorch (make sure that the CUDA version matches with "Step 1")
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# or
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
# make sure torch.cuda.is_available() returns True

pip install -e .
```

> [!NOTE]
> - We use an adjusted version of [nvdiffrast](https://github.com/ShenhanQian/nvdiffrast/tree/backface-culling) for backface-culling. If you have other versions installed before, you can reinstall as follows:
>    ```shell
>    pip install nvdiffrast@git+https://github.com/ShenhanQian/nvdiffrast@backface-culling --force-reinstall
>    rm -r ~/.cache/torch_extensions/*/nvdiffrast*
>    ```
> - We use [STAR](https://github.com/ShenhanQian/STAR/) for landmark detection by default. Alterntively, [face-alignment](https://github.com/1adrianb/face-alignment) is faster but less accurate.

## Download

### FLAME

Our code relies on FLAME. Please download assets from the [official website](https://flame.is.tue.mpg.de/download.php) and store them in the paths below:

- FLAME 2023 (versions w/ jaw rotation) -> `asset/flame/flame2023.pkl`
- FLAME Vertex Masks -> `asset/flame/FLAME_masks.pkl`

> [!NOTE]
> It is possible to use FLAME 2020 by download to `asset/flame/generic_model.pkl`. The `FLAME_MODEL_PATH` in `flame.py` needs to be updated accordingly.

## Usage

### Multiview
[For NeRSemble Dataset](doc/nersemble.md)


[For EmoTalk3d](doc/emotalk3d.md)

## Acknowledgments

Part of this codebase is borrowed from [VHAP](https://github.com/ShenhanQian/VHAP). We thank the authors for releasing their head tracking pipeline.


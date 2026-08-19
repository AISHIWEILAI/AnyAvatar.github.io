<div align="center">

# [ACM MM 2026 Poster] AnyAvatar: High-Fidelity Gaussian Head Avatars under Uncalibrated Camera Settings

<img src="static/images/pipeline.png" alt="AnyAvatar" width="100%"/>

[Project](https://syncanimation.github.io/AnyAvatar.github.io/) / [Paper](https://github.com/syncanimation/AnyAvatar.github.io) 

**Yujian Liu**<sup>1,2,&#42;</sup>, **Dongxu Shen**<sup>3,&#42;</sup>, **Haoran Li**<sup>1,&#42;</sup>, **Yuting Liu**<sup>1</sup>, **Chuang Chen**<sup>1</sup>, **Xinyi Jiang**<sup>1</sup>, **Zhupeng Jiang**<sup>1</sup>, **Peng Cao**<sup>4,†</sup>, **Shidang Xu**<sup>2,†</sup>, **Xiaoli Liu**<sup>1,†</sup>

<sup>1</sup> AiShiWeiLai AI Research  <sup>2</sup> South China University of Technology  
<sup>3</sup> The Hong Kong University of Science and Technology (Guangzhou)  <sup>4</sup> Northeastern University

</div>

## Usage

### Step 1. Coarse Camera Pose Initialization

Obtain coarse camera poses with [VGGT](https://github.com/facebookresearch/vggt). Please refer to the official VGGT repository for installation and inference.

In our implementation, we use the **first frame of the neutral expression** to estimate camera poses.

### Step 2. VHAP Training

We use the camera poses estimated by VGGT to train VHAP. For specific instructions, please refer to [VHAP/README.md](VHAP/README.md).

After this step, we obtain the FLAME mesh heads that will be used for Gaussian training.

### Step 3. Gaussian Training

With the mesh heads obtained from VHAP, we can start training Gaussian head avatars. For specific instructions, please refer to [AnyAvatar/README.md](AnyAvatar/README.md).


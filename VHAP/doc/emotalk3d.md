## For NeRSemble Dataset

### 1. Preprocess

This step extracts frames from video(s), then run foreground matting for each frame, which requires GPU.

```shell
SUBJECT="0048"
SEQUENCE="2-00-01"

python vhap/preprocess_video.py \
--input data/emo3d/${SUBJECT}/${SEQUENCE}* \
--matting_method background_matting_v2
```

- `--matting_method background_matting_v2`: Use BackGroundMatingV2 due to availability of background images.


### 2. Align and track faces

This step automatically detects facial landmarks if absent, then begin FLAME tracking. For multi-view calibrated data, we initialize FLAME translation at the intersection of camera viewing rays, then project it into the overlapping frustum region of all cameras. We initialize shape and appearance parameters on the first frame, then do a sequential tracking of following frames. After the sequence tracking, we conduct 30 epochs of global tracking, which optimize all the parameters on a random frame in each iteration.

```shell
SUBJECT="0048"
SEQUENCE="2-00-01"
TRACK_OUTPUT_FOLDER="output/emo3d/${SUBJECT}_${SEQUENCE}_v16_DS4_wBg_staticOffset"

python vhap/track_nersemble.py --data.root_folder "data/emo3d" \
--exp.output_folder $TRACK_OUTPUT_FOLDER \
--data.subject $SUBJECT --data.sequence $SEQUENCE \
--data.image_size_during_calibration 512 512 \
--data.no_use_color_correction
```

Optional arguments

- `--data.image_size_during_calibration 512 512`: set the image resolution used during camera calibration to 512×512, matching the EmoTalk3D / Emo3D data processed at this resolution. 

- `--data.no_use_color_correction`: disable NeRSemble-style per-camera color correction. EmoTalk3D does not provide the `color_correction/` files required by the default NeRSemble pipeline, so this flag should be enabled for EmoTalk3D tracking.

- `--model.no_use_static_offset`: disable static offset for FLAME (very stable, but less aligned facial geometry)

  > Disabling static offset will automatically triggers `--model.occluded hair`, which is crucial to prevent the head from growing too larger to align with the top of hair.

- `--exp.no_photometric`: track only with landmark (very fast, but coarse)

> [!NOTE]
> We use all 11 views for the optimization, but we only visualize 3 views for efficiency.

### 3. Export tracking results into a NeRF-style dataset

Given the tracked FLAME parameters from the above step, you can export the results to form a NeRF/3DGS style sequence, consisting of image folders and a `transforms.json`.

```shell
SUBJECT="0048"
SEQUENCE="2-00-01"
TRACK_OUTPUT_FOLDER="output/emo3d/${SUBJECT}_${SEQUENCE}_v16_DS4_wBg_staticOffset"
EXPORT_OUTPUT_FOLDER="export/emo3d/${SUBJECT}_${SEQUENCE}_v16_DS4_whiteBg_staticOffset_maskBelowLine"

python vhap/export_as_nerf_dataset.py \
--src_folder ${TRACK_OUTPUT_FOLDER} \
--tgt_folder ${EXPORT_OUTPUT_FOLDER} --background-color white \
--target-camera-id F
```

Optional arguments

- `--target-camera-id F`: set the world origin to camera `F` during export. This is required to adapt the exported dataset to the second-stage drivable avatar pipeline, which expects the coordinate system to be camera-centered rather than FLAME-centered.

### 4. Combine exported sequences of the same person as a union dataset

```shell
SUBJECT="0048"

python vhap/combine_nerf_datasets.py \
--src_folders \
  export/emo3d/${SUBJECT}_2-00-01_v16_DS4_whiteBg_staticOffset_maskBelowLine \
  export/emo3d/${SUBJECT}_2-00-02_v16_DS4_whiteBg_staticOffset_maskBelowLine \
  export/emo3d/${SUBJECT}_2-00-03_v16_DS4_whiteBg_staticOffset_maskBelowLine \
  export/emo3d/${SUBJECT}_2-00-04_v16_DS4_whiteBg_staticOffset_maskBelowLine \
  export/emo3d/${SUBJECT}_2-00-05_v16_DS4_whiteBg_staticOffset_maskBelowLine \
  export/emo3d/${SUBJECT}_2-00-06_v16_DS4_whiteBg_staticOffset_maskBelowLine \
  export/emo3d/${SUBJECT}_2-00-07_v16_DS4_whiteBg_staticOffset_maskBelowLine \
  export/emo3d/${SUBJECT}_2-00-08_v16_DS4_whiteBg_staticOffset_maskBelowLine \
  export/emo3d/${SUBJECT}_2-00-09_v16_DS4_whiteBg_staticOffset_maskBelowLine \
  export/emo3d/${SUBJECT}_2-00-10_v16_DS4_whiteBg_staticOffset_maskBelowLine \
--tgt_folder \
  export/emo3d/UNION10_${SUBJECT}_2-00-01-10_v16_DS4_whiteBg_staticOffset_maskBelowLine
```

> [!NOTE]
> The `tgt_folder` must be in the same parent folder as `src_folders` because the union dataset read from the original image files by relative paths.

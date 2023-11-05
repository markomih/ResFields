# Automatic 4D Scene Dataset Creation - Script Guide

This script provides a way you can convert your videos to compatible dynamic NeuS/IDR setup (Owlii setup from docs/data.md). 
In script `preprocess_rgb_data.py`, you can specify which videos in your setup are made with the moving/dynamic camera, and which are static. Following code below shows how to specify this in the script. 

Test dataset can be found in [shortened form in GDrive](https://drive.google.com/drive/folders/1rcwebURNmsDlcgDYkNymOWYpyD-7Vw4o?usp=sharing). `plane_dataset` is one used for input, `plane_dataset_owlii_format` is resulting format you should get after calling this script.


Before calling function `preprocess_data.py`, you should edit it and specify videos you are going to extract.

```python

    DATA_ROOT = "plane_dataset"

    static_video_paths = [
        f'{DATA_ROOT}/cam_0.mkv',
        f'{DATA_ROOT}/cam_1.mkv',
        f'{DATA_ROOT}/cam_2.mkv',
        f'{DATA_ROOT}/cam_3.mkv'
    ]    
    dynamic_video_paths = [
        # Empty, not yet supported
    ]
```



### Colmap to Owlii format:
   Currently supports only the dynamic scene format with only non-moving cameras.


   * --colmap_db : path to colmap database (default: 'colmap.db').
   * --colmap_matcher : which matcher to use (default: 'exhaustive')
   * --input_model_path : COLMAP model's name (default: 'colmap_model')
   * --run_make_dataset: calls video-to-images script for each video in dataset.            
      - --extracted_images_folder : path to folder where images from videos are extracted, and will be later used for COLMAP. Each video will have a separate folder with color and possibly depth images.
      - --begin_frame : frame microseconds from which to start extracting images (default: 0)
      - --end_frame : frame microseconds from which to end extracting images (default: -1)
      - --by_step : step in microseconds between frames (default: 33333 for 30fps)
   * --run_initial_colmap: calls colmap on initial images. If COLMAP images not provided (i.e. compatible with extracted videos, and containing charuco board), will use `--begin_frame` for each video to extract camera extrinsics/intrinsics.
      - you need to specify folder with calibration frames in the similar pattern as the --run_make_dataset does, meaning that you should follow it's image pattern `f'video_{camera_view_id}_frame_{chosen_timestep}.png'`.
   * -- run_transform_neus: call conversion of all extracted images and depths and camera intrinsics to format compatible with NeuS/IDR format (Owlii dataset from DATA_ROOT). You'll be prompted to choose whether you want to edit pointcloud with COLMAP SfM features as shown in [NeuS repo](https://github.com/Totoro97/NeuS/tree/main/preprocess_custom_data). You may skip this step if convenient.


### Example run:
    
```bash
python preprocess_rgb_data.py --colmap_db plane_colmap.db --colmap_matcher exhaustive --input_model_path plane_model --run_make_dataset --extracted_images_folder plane_dataset_owlii_format --begin_frame 0 --end_frame 969000 --by_step 33333 --run_initial_colmap --run_transform_neus
```

## TODO list:
- [ ] Add aligned depth support for training.
- [ ] Add segmentation support.
- [ ] Add SLAM/COLMAP support for moving camera capture.
- [ ] Make abstract support for video loading (Kinect's MKV, GoPro, Realsense etc.).


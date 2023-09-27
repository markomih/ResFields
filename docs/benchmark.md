# Benchmark

This document contains instructions on how reproduce results from the paper. 

# 2D video approximation

To reproduce results on the video approximation task, execute the bash script:
```bash
cd ./dyrecon
bash ./benchmark_tsdf.sh
```
To reproduce the small demo: 
```bash
export sequence="../DATA_ROOT/video/cat.mp4" # or "skvideo.datasets.bikes"
python launch.py --config ./configs/video/base.yaml --train --predict dataset.video_path=$sequence --exp_dir ../exp_video model.hidden_features=256
python launch.py --config ./configs/video/base.yaml --train --predict dataset.video_path=$sequence --exp_dir ../exp_video model.resfield_layers=[1,2,3] model.composition_rank=40 tag=ResFields model.hidden_features=128
```

![Video](../assets/cat.gif)

# Temporal SDF
To reproduce results on the Temporal SDF shape estimation task, execute:
```bash
export sequence="../DATA_ROOT/ReSynth/dress" # or path to a deforming things 4D sequence
python launch.py --exp_dir ../exp_tsdf --config ./configs/tsdf/base.yaml --train model.hidden_features=256 dataset.path=$sequence tag=Siren256
python launch.py --exp_dir ../exp_tsdf --config ./configs/tsdf/base.yaml --train model.hidden_features=256 dataset.path=$sequence model.resfield_layers=[1,2,3] model.composition_rank=10 tag=Siren256ResFields123_10
# note that ReSynth uses high-resolution meshes which makes training slow. Training on DeformingThings4D is significantly faster
```

![TSDF](../assets/tsdf.gif)

# Dynamic NeRF from 4 RGB views
We consider the following models in our benchmark: 
1. [TNeRF](https://neural-3d-video.github.io/) Li et al. (CVPR 2022) and Pumarola et al. (CVPR 2021)
2. [DyNeRF](https://neural-3d-video.github.io/) Li et al., CVPR 2022
3. [DNeRF](https://neural-3d-video.github.io/) Pumarola et al., CVPR 2021
4. [Nerfies](https://github.com/google/nerfies), Park et al., ICCV 2021
5. [HyperNeRF](https://github.com/google/hypernerf), Park et al., SIGGRAPH Asia 2021
6. [NDR](https://github.com/USTC3DV/NDR-code), Cai et al., NeurIPS 2022

To run a respective baseline execute (e.g. TNeRF on the Basketball squence from Owlii):
```bash 
cd ./dyrecon
export method=tnerf sequence=basketball
# run training of vanilla baseline
python launch.py --config ./configs/dysdf/$method.yaml dataset.scene=$sequence --exp_dir ../exp_owlii_benchmark --train
# run training of vanilla baseline + ResFields
python launch.py --config ./configs/dysdf/$method.yaml dataset.scene=$sequence --exp_dir ../exp_owlii_benchmark --train model.sdf_net.resfield_layers=[1,2,3,4,5,6,7] tag=ResFields1234567
```
Note that `method` could be any of the baselines (tnerf, dnerf, dynerf, hypernerf, ndr, nerfies) and squence could be any Owlii sequence (basketball, dancer, exercise, model).

Or to reproduce the benchmark on the Owlii dataset (Table 3 in the paper), execute the bash script
```bash
cd ./dyrecon
bash ./benchmark_owlii.sh
```
All the results will be stored in the respective directories. 

![TNeRF](../assets/tnerf.gif)

# RGB-D reconstruction
For the RGB-D reconstruction we include two additional loss terms (depth and sparsness) as mentioned in the paper. 
Until we release the paper, you could try it out on the Owlii dataset for which we release the depth maps too, e.g:

```bash
export method=tnerf sequence=basketball
# run training of vanilla baseline
python launch.py --config ./configs/dysdf/$method.yaml system.loss.depth=0.1 system.loss.sparse=[0.1,70000] dataset.scene=$sequence --exp_dir ../exp_owlii_benchmark --train
# run training of vanilla baseline + ResFields
python launch.py --config ./configs/dysdf/$method.yaml system.loss.depth=0.1 system.loss.sparse=[0.1,70000] dataset.scene=$sequence --exp_dir ../exp_owlii_benchmark --train model.sdf_net.resfield_layers=[1,2,3,4,5,6,7] tag=ResFields1234567
# note that the sparsness loss will be activated after 70k iterations
```

![TNeRF_RGBD](../assets/tnerf_rgbd.gif)

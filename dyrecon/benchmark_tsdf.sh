#!/bin/bash
set -x # print commands

SEQUENCES=(../DATA_ROOT/ReSynth/dress  ../DATA_ROOT/DeformingThings4D/bear3EP_Agression.anime  ../DATA_ROOT/DeformingThings4D/tigerD8H_Swim17.anime ../DATA_ROOT/DeformingThings4D/vampire_Breakdance1990.anime ../DATA_ROOT/DeformingThings4D/vanguard_JoyfulJump.anime)
METHODS=(128 256)

# iterate over sequences and methods
for sequence in "${SEQUENCES[@]}"; do
    for method in "${METHODS[@]}"; do
        # run training of vanilla baseline
        tag="Siren${method}"
        python launch.py --config ./configs/tsdf/base.yaml dataset.path=$sequence --exp_dir ../exp_owlii_benchmark --train model.hidden_features=$method tag=$tag
        # run training of vanilla baseline + ResFields
        python launch.py --config ./configs/tsdf/base.yaml dataset.path=$sequence --exp_dir ../exp_owlii_benchmark --train model.sdf_net.resfield_layers=[1,2,3,4,5,6,7] model.hidden_features=$method tag=ResFields1234567$tag
    done
    # Note that for the instant-ngp experiment, you would need to intall tiny-cuda-nn (`pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch`):
    # python launch.py --exp_dir ../exp_video_rnd --config ./configs/video/ngp.yaml --train --predict dataset.video_path=$sequence model.log2_hashmap_size=23 model.n_levels=8 tag=T23_L8
done

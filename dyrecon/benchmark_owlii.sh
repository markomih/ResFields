#!/bin/bash
set -x # print commands

SEQUENCES=(basketball  dancer  exercise  model)
METHODS=(tnerf dnerf dynerf hypernerf ndr nerfies)

# iterate over sequences and methods
for sequence in "${SEQUENCES[@]}"; do
    for method in "${METHODS[@]}"; do
        # run training of vanilla baseline
        python launch.py --config ./configs/dysdf/$method.yaml dataset.scene=$sequence --train
        # run training of vanilla baseline + ResFields
        python launch.py --config ./configs/dysdf/$method.yaml dataset.scene=$sequence --train model.sdf_net.resfield_layers=[1,2,3,4,5,6,7] tag=ResFields1234567
    done
done

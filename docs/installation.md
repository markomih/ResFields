# Installation

The code was tested on Ubuntu 22.04 with CUDA 11.6 and Python 3.9.

## 1. Clone the repo

```bash
git clone https://github.com/markomih/ResFields.git
cd ResFields
```

## 2. Install necessary dependencies

Create a new `conda` [environment](https://www.anaconda.com/) with all dependencies: 
```bash
conda create -n ResFields python=3.9 -y
conda activate ResFields
conda install cudatoolkit=11.6  -y
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install tqdm scikit-image opencv-python configargparse lpips imageio-ffmpeg lpips tensorboard numpy==1.22.4 sk-video trimesh wandb omegaconf pysdf pymcubes matplotlib pytorch-lightning==1.6.5 gdown
```

Now you can verify that the environment is set up correctly by running a simple experiment of learning a 2D video via a Siren+ResField MLP: 
```bash
cd dyrecon
python launch.py --config ./configs/video/base.yaml --train --predict model.resfield_layers=[1,2,3] model.composition_rank=10 tag=ResFields

# or the following command for the vanilla Siren
python launch.py --config ./configs/video/base.yaml --train --predict tag=vanilla
```

## 3. [Optional] Download the data data
See [data preparation](https://github.com/markomih/ResFields/blob/master/docs/data.md) to set up the datasets


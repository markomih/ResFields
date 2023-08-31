# Installation

The code was tested on Ubuntu 22.04 with CUDA 11.3 and Python 3.8.

## 1. Clone the repo.

```bash
git clone https://github.com/markomih/ResFields.git
cd ResFields
```

## 2. Install necessary dependencies .

Create a new `conda` [environment](https://www.anaconda.com/) with all dependencies: 
```bash
conda env create -f environment.yml
conda activate ResFields
```
or manually install dependencies specified in `requirements.txt`:
```bash
pip install -r requirements.txt
```

Now you can verify that the environment is set up correctly by running a simple experiment of learning a 2D video via a Siren+ResField MLP: 
```bash
python train ...
```

## 3. [Optional] Download the data data
See [data preparation](https://github.com/markomih/ResFields/blob/master/docs/data.md) to set up the datasets


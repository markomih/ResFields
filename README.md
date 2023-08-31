<p align="center">
  <p align="center">
    <h1 align="center">ResFields: Residual Neural Fields for Spatiotemporal Signals</h1>
  </p>
  <p align="center" style="font-size:16px">
    <a target="_blank" href="https://markomih.github.io/"><strong>Marko Mihajlovic</strong></a>
    ·
    <a target="_blank" href="https://vlg.inf.ethz.ch/team/Dr-Sergey-Prokudin.html"><strong>Sergey Prokudin</strong></a>
    ·
    <a target="_blank" href="https://scholar.google.com/citations?user=YYH0BjEAAAAJ&hl=en"><strong>Marc Pollefeys</strong></a>
    ·
    <a target="_blank" href="https://vlg.inf.ethz.ch/team/Prof-Dr-Siyu-Tang.html"><strong>Siyu Tang</strong></a>
  </p>
  <h2 align="center"></h2>
  <div align="center"></div> 

  <p align="center">
    <a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
    <a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
    <br>
    <a href='https://arxiv.org/abs/XXX.XXX'>
      <img src='https://img.shields.io/badge/Paper-PDF-green?style=for-the-badge&logo=arXiv&logoColor=green' alt='Paper PDF'>
    </a>
    <a href='https://markomih.github.io/ResFields/' style='padding-left: 0.5rem;'>
      <img src='https://img.shields.io/badge/ResFields-Page-orange?style=for-the-badge&logo=Google%20chrome&logoColor=orange' alt='Project Page'>
    </a>
  </p>
<p align="center">
<a href='https://colab.research.google.com/drive/1YRgwoRCZIrSB2e7auEWFyG10Xzjbrbno?usp=sharing'><img src='https://img.shields.io/badge/Colab Demo-ec740b.svg?logo=googlecolab' alt='Google Colab'></a>

</p>
  <p align="center"><a href='https://paperswithcode.com/sota/generalizable-novel-view-synthesis-on-zju?p=keypointnerf-generalizing-image-based'>
	<img src='https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/keypointnerf-generalizing-image-based/generalizable-novel-view-synthesis-on-zju' alt='Generalizable Novel View Synthesis'></a>
  </p>
</p>
<video width="100%" autoplay muted controls loop src="https://github.com/markomih/ResFields/assets/13746017/b708331c-b3cf-43ab-9f2f-8458fe599fdb"></video>
<p>
ResField layers incorporate time-dependent weights into MLPs to effectively represent complex temporal signals. 
</p>

## Applications


|                              ![Video](assets/cat.gif)                              |                                                ![TSDF](assets/tsdf.gif)                                                   |
| :--------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------: |
|                               2D Video Approximation                               |                                                   Temporal SDF Capture                                                    |
|                               ![TNeRF](assets/tnerf.gif)                           |                                            ![TNeRF_RGBD](assets/tnerf_rgbd.gif)                                           |
|                              Dynamic NeRFs from 4 RGB views                        |                                                   Dynamic NeRFs from 3 RGB-D                                              |


## News :triangular_flag_on_post:
- [2023/10/01] Code released. 

## Key idea of ResFields
Our key idea is to substitute one or several MLP layers with time-dependent layers whose weights are modeled as trainable residual parameters added to the existing layer weights.

<img width="30%" src='https://markomih.github.io/ResFields/images/resfield_mlp_web.png'>

We propose to implement the residual parameters as a global low-rank spanning set and a set of time-dependent coefficients. this modeling enhances the generalization properties and further reduces the memory footprint caused by maintaining additional network parameters. 

<img width="30%" src='https://markomih.github.io/ResFields/images/resfield_decomp.png'>

These residual weights are modeled as a learnable low-rank composition.

Increasing the model capacity in this way offers three key advantages:

1) **Runtime**: the underlying MLP does not increase in size and hence maintains the inference and training speed.
2) **Generalizability**: retains the implicit regularization and generalization properties of MLPs.
3) **Universality**: ResFields are versatile, easily extendable, and compatible with most MLP-based methods for spatiotemporal signals.


<details><summary>Please consider citing our work if you find it useful</summary>

```bibtex
@inproceedings{Mihajlovic:ResFields:2023,
  title = {{ResFields}: Residual Neural Fields for Spatiotemporal Signals},
  author = {Mihajlovic, Marko and Prokudin, Sergey and Pollefeys, Marc and Tang, Siyu},
  booktitle={Arxiv},
  year={2023},
}
```
</details>
<br>

<details open="open" style='padding: 10px; border-radius:5px 30px 30px 5px; border-style: solid; border-width: 1px;'>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#instructions">Instructions</a>
    </li>
    <li>
      <a href="#demos">Demos</a>
    </li>
    <li>
      <a href="#citation">Citation</a>
    </li>
  </ol>
</details>
<br/>

## Instructions


## ToDos:
- Preprocessing code for the RGB-D setup
- RGB-D data

## Installation 
Please install python dependencies specified in `environment.yml`:
```bash
conda env create -f environment.yml
conda activate ResFields
```
## Data preparation
Please see [DATA_PREP.md](DATA_PREP.md) to set up the datasets.

After this step the data directory follows the structure:
```bash
$DATA_ROOT
├── Owlii
├── Kinect4D
├── Videos
```
## Experiments
We demonstrate ResFields on four different tasks: 2D video approximation, Temporal SDF learning, and 4D NeRF reconstruction from RGB and RGBD views.
Here, we will summarize how to run respective experiments. 
Please see [BENCHMARK.md](BENCHMARK.md) on how to reproduce results for various baselines. 

### 2D video approximation 
To run vanilla siren execute:
```shell script
python train.py blabla
```
To run siren + ResFields execute:
```shell script
python train.py blabla
```
The results will be stored under the respective experiment directory.

## Train your own model on the ZJU dataset
Execute `train.py` script to train the model on the ZJU dataset.
```shell script
python train.py --config ./configs/blabla.json dataset.data_root=
```
After the training, the model checkpoint will be stored under `./exp_dir/zju/ckpts/last.ckpt`, which is equivalent to the one provided [here](https://drive.google.com/file/d/...).
The valuation will be automatically executed after the training. 


## Citation

```bibtex
@inproceedings{Mihajlovic:ResFields:2023,
  title = {{ResFields}: Residual Neural Fields for Spatiotemporal Signals},
  author = {Mihajlovic, Marko and Prokudin, Sergey and Pollefeys, Marc and Tang, Siyu},
  booktitle={Arxiv},
  year={2023},
}
```

## Acknowledgments
We thank Hongrui Cai and Ruizhi Shao for providing additional details about the baseline methods and Anpei Chen, Shaofei Wang, and Songyou Peng for proofreading the manuscript and proving useful suggestions. 

Some great prior work we benefit from:
- [Siren](https://github.com/vsitzmann/siren) for the 2D video approximation task
- [NeuS](https://github.com/Totoro97/NeuS) for data preprocessing and following their data format
- [Owlii](https://mpeg-pcc.org/index.php/pcc-content-database/owlii-dynamic-human-textured-mesh-sequence-dataset/), [DeformingThings4D](https://github.com/rabbityl/DeformingThings4D), and [ReSynth](https://pop.is.tue.mpg.de/) for datasets
- [PyTorch3D](https://github.com/facebookresearch/pytorch3d) for visualizing meshes and some evaluation scripts
- [Instant NSR](https://github.com/bennyguo/instant-nsr-pl) for inspiring the code structure

### Disclosure of Funding
This project has been supported by [the Innosuisse Flagship project PROFICIENCY](https://www.surgicalproficiency.ch/).

## License
The code and models are available for use without any restrictions. 
See the [LICENSE](LICENSE) file for details. 

## Contact
Please open a PR or contact [Marko Mihajlovic](https://markomih.github.io/) for any questions. 
We greatly appreciate everyone's feedback and insights. 
Please do not hesitate to get in touch. 

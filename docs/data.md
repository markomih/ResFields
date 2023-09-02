# Datasets

Please find the processed datasets at [here](https://drive.google.com/file/d/1KEZY3_bReZL23MNvSITtrsg8Spbh7Fo7/view?usp=sharing) or execute the following script to download all sequences:
```bash
gdown 1KEZY3_bReZL23MNvSITtrsg8Spbh7Fo7
unzip DATA_ROOT.zip
```
All datasets are organized as `<DATA_ROOT>/<DATASET>/<SEQUENCE>`.

If you need access to the [Deforming4D](https://github.com/rabbityl/DeformingThings4D) dataset, you will need to manually obtain the license and store the data under `./DATA_ROOT/DeformingThings4D`. 

After downloading the data, `DATA_ROOT` will have the following structure: 

```
DATA_ROOT/
    ├── Owlii/
    │   └── basketball/
    │   │   └── camera_*/**
    │   └── exercise/
    │   │   └── camera_*/**
    │   └── dancer/
    │   │   └── camera_*/**
    │   └── model/
    │   │   └── camera_*/**
    ├── Video/
    │   └── cat.mp4
    ├── ReSynth/**
    │   └── dress
    │       └── mesh*.ply
    │           └── <ITEM_ID>.png
    ├── DeformingThings4D/**
    │   ├── bear3EP_Agression.anime
    │   ├── tigerD8H_Swim17.anime
    │   ├── vampire_Breakdance1990.anime
    │   └── vanguard_JoyfulJump.anime
    └── Kinect4D
        └── book/
        │   └── camera_*/**
        └── hand/
        │   └── camera_*/**
        └── writing/
        │   └── camera_*/**
        └── glasses/
            └── camera_*/**
```
Note: We will release the Kinect4D data as soon as possible. 

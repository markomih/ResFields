import os
import torch
import argparse
import numpy as np
import cv2
from tqdm import tqdm
from glob import glob
from pytorch3d.structures import Meshes
from pytorch3d.renderer.cameras import FoVOrthographicCameras, look_at_view_transform

parser = argparse.ArgumentParser()
parser.add_argument('--src', 
                    # default='/media/STORAGE_4TB/projects/ResFields/datasets/public_data/mv_basketball_neurips2023_10', 
                    # default='/media/STORAGE_4TB/projects/ResFields/datasets/public_data/model', 
                    # default='/media/STORAGE_4TB/projects/ResFields/datasets/public_data/dancer_vox11', 
                    default='/media/STORAGE_4TB/projects/ResFields/datasets/public_data/exercise_vox11', 
                    help='Directory with meshes')
parser.add_argument('--dst', 
                    default='./tmp_videos', 
                    help='Directory with dst images')

def imgs2video(img_dir, out_path, fps=20):
   img_regex = os.path.join(img_dir, '*.png')
   cmd = f"ffmpeg -framerate {fps} -pattern_type glob -i '{img_regex}'  -c:v libx264 -pix_fmt yuv420p {out_path} -y"
   os.system(cmd)

def main(args):
    src_dir = args.src
    dst_dir = args.dst
    for cam in CAMERAS:
        dir_path = os.path.join(src_dir, cam)
        rgbs = sorted(glob(os.path.join(dir_path, 'rgb/*.png')))
        masks = sorted(glob(os.path.join(dir_path, 'mask/*.png')))
        assert len(rgbs) == len(masks), f'len(rgbs)={len(rgbs)} != len(masks)={len(masks)}'
        # load images 
        rgbs = np.stack([cv2.imread(rgb) for rgb in rgbs])
        masks = np.stack([cv2.imread(mask) for mask in masks]) #> 0
        rgbs = (rgbs.astype(np.uint32) + (255 - masks.astype(np.uint32))).astype(np.uint8)
        dst_dir_imgs = os.path.join(dst_dir, f"{os.path.basename(src_dir)}_{cam}")
        os.system(f'mkdir -p {dst_dir_imgs}')
        for ind in tqdm(range(len(rgbs))):
            cv2.imwrite(os.path.join(dst_dir_imgs, f"{ind:05d}.png"), rgbs[ind])
        imgs2video(dst_dir_imgs, dst_dir_imgs + '_video.mp4')
         
    os.system(f'mkdir -p {dst_dir}')

if __name__ == '__main__':
    CAMERAS = ['cam_train_1','cam_train_3','cam_train_6','cam_train_8',]
    device = torch.device('cuda')
    main(parser.parse_args())

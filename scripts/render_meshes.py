import os
import torch
import argparse
import trimesh
import cv2
from tqdm import tqdm
from glob import glob
from pytorch3d.structures import Meshes
from pytorch3d.renderer.cameras import FoVOrthographicCameras, look_at_view_transform

parser = argparse.ArgumentParser()
parser.add_argument('--src', 
                    default='/home/marko/remote_euler/projects/ResFields/exp_tsdf250k_mape/tsdf/resynth/baseSiren256ResFields123_40/save/meshes/it200000/', 
                    help='Directory with meshes')
parser.add_argument('--dst', 
                    default='./tmp_meshes', 
                    help='Directory with dst images')

from pytorch3d.renderer import (
    PerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    SoftSilhouetteShader,
    HardPhongShader,
    AmbientLights,
    PointLights,
    BlendParams,
    Materials,
    TexturesUV,
    Textures
)

def get_torch_mesh(v, f):
  v = torch.Tensor(v).reshape([-1, 3]).unsqueeze(0)
  f = torch.Tensor(f).reshape([-1, 3]).unsqueeze(0)
  mesh = Meshes(v, f)
  return mesh


def get_mesh_renderer(cameras, image_size=512, device='cuda'):
  materials = Materials(device=device, specular_color=[[0.0, 0.0, 0.0]], shininess=100.0)
  raster_settings = RasterizationSettings(image_size=image_size, blur_radius=0.0, faces_per_pixel=1, bin_size=None, cull_backfaces=True)
  blend_params = BlendParams(background_color=(0, 0, 0))
  lights = AmbientLights(ambient_color=(1, 1, 1), device=device)
  phong_renderer = MeshRenderer(
      rasterizer=MeshRasterizer(
          cameras=cameras,
          raster_settings=raster_settings
      ),
      shader=HardPhongShader(device=device, cameras=cameras, blend_params=blend_params, lights=lights,materials=materials)
  )
  return phong_renderer

def normals_to_rgb(n):
  return torch.abs(n*0.5 + 0.5)

def get_normals_as_textures(mesh):
  textures = Textures(verts_rgb=normals_to_rgb(mesh.verts_normals_packed()).unsqueeze(0))
  verts = mesh.verts_packed()
  mesh_new = Meshes(verts.unsqueeze(0), mesh.faces_packed().unsqueeze(0), textures)
  return mesh_new

def get_textured_mesh(v, f, c):
  v = torch.Tensor(v).reshape([-1, 3]).unsqueeze(0)
  f = torch.Tensor(f).reshape([-1, 3]).unsqueeze(0)
  c = torch.Tensor(c).reshape([-1, 3]).unsqueeze(0)
  textures = Textures(verts_rgb=c)
  mesh = Meshes(v, f, textures)
  return mesh

def render_mesh(mesh, dist=1, elev=0, azim=0, image_size=512, scale_val=1.0):
  R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim)
  cam = FoVOrthographicCameras(R=R, T=T, scale_xyz=((scale_val, scale_val, scale_val),)).to(device)
  #cam = PerspectiveCameras(R=R, T=T, scale_xyz=((scale_val, scale_val, scale_val),)).to(device)
  renderer = get_mesh_renderer(cam, image_size=image_size)
  img = renderer(mesh, cameras=cam)[0]
  img = (img*255.0).clamp(0, 255).byte().cpu().numpy()
  trans_mask = img[:,:,3] == 0
  img[trans_mask] = [255, 255, 255, 255]
  return img

def imgs2video(img_dir, out_path, fps=15):
   img_regex = os.path.join(img_dir, '*.png')
   cmd = f"ffmpeg -framerate {fps} -pattern_type glob -i '{img_regex}'  -c:v libx264 -pix_fmt yuv420p {out_path} -y"
   os.system(cmd)

def main(args):
    # load meshes
    src_dir = args.src
    dst_dir = args.dst
    os.system(f'mkdir -p {dst_dir}')
    files = sorted(glob(src_dir + '/*.obj') + glob(src_dir + '/*.ply'))
    for ind, f in tqdm(enumerate(files)):
        mesh = trimesh.load(f)
        mesh = get_torch_mesh(mesh.vertices, mesh.faces)
        mesh = get_normals_as_textures(mesh).to(device)
        img = render_mesh(mesh)
        fname = f"{ind:05d}.png" # os.path.basename(f).split('.')[0] + '.png'
        img = cv2.cvtColor(img[..., :3], cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(dst_dir, fname), img)

    imgs2video(dst_dir, dst_dir + '_video.mp4')
    print('Saved video to', dst_dir + '_video.mp4')

if __name__ == '__main__':
    device = torch.device('cuda')
    main(parser.parse_args())

# python scripts/render_meshes.py --src /home/marko/remote_euler/projects/ResFields/exp_tsdf250k_mape/tsdf/resynth/baseSiren256ResFields123_40/save/meshes/it200000 --dst /home/marko/remote_euler/projects/ResFields/exp_tsdf250k_mape/tsdf/resynth/baseSiren256ResFields123_40/save/meshes/it200000_rnd
# python scripts/render_meshes.py --src /home/marko/remote_euler/projects/ResFields/exp_tsdf250k_mape/tsdf/resynth/baseSiren256/save/meshes/it200000 --dst /home/marko/remote_euler/projects/ResFields/exp_tsdf250k_mape/tsdf/resynth/baseSiren256/save/meshes/it200000_rnd
# python scripts/render_meshes.py --src /media/STORAGE_4TB/projects/ResFields/datasets/resynth --dst /media/STORAGE_4TB/projects/ResFields/datasets/resynth_rnd

# ffmpeg \
# -i /home/marko/remote_euler/projects/ResFields/exp_tsdf250k_mape/tsdf/resynth/baseSiren256/save/meshes/it200000_rnd_video.mp4 \
# -i /home/marko/remote_euler/projects/ResFields/exp_tsdf250k_mape/tsdf/resynth/baseSiren256ResFields123_40/save/meshes/it200000_rnd_video.mp4 \
# -i /media/STORAGE_4TB/projects/ResFields/datasets/resynth_rnd_video.mp4 \
# -c:v libx264 -preset slow  -profile:v high -level:v 4.0 -pix_fmt yuv420p -crf 22 -codec:a aac  -filter_complex hstack=inputs=2  ./tmp_resynth.mp4 -y

# ffmpeg -i /home/marko/remote_euler/projects/ResFields/exp_tsdf250k_mape/tsdf/resynth/baseSiren256/save/meshes/it200000_rnd_video.mp4 -i /home/marko/remote_euler/projects/ResFields/exp_tsdf250k_mape/tsdf/resynth/baseSiren256ResFields123_40/save/meshes/it200000_rnd_video.mp4 -i /media/STORAGE_4TB/projects/ResFields/datasets/resynth_rnd_video.mp4 -c:v libx264 -preset slow  -profile:v high -level:v 4.0 -pix_fmt yuv420p -crf 22 -codec:a aac  -filter_complex hstack=inputs=3  ./tmp_resynth.mp4 -y
# ffmpeg -i tmp_resynth.mp4 -filter_complex "[0:v]reverse,fifo[r];[0:v][r] concat=n=2:v=1 [v]" -map "[v]" tmp_resynth.mp4.mp4 -y

import torch
import numpy as np
import torch.nn.functional as F
import yaml
import systems
from systems.base import BaseSystem
import torch.nn.functional as F
from models.utils import extract_geometry, mape_loss

try:
    from pytorch3d.structures.meshes import Meshes
    from pytorch3d.loss.chamfer import chamfer_distance
    from pytorch3d.ops import sample_points_from_meshes

    from pytorch3d import renderer
    from pytorch3d.structures import Meshes
    from pytorch3d.renderer.mesh import Textures
    from pytorch3d.renderer import look_at_rotation
    PYTORCH3D_AVAILABLE = True
except Exception:
    PYTORCH3D_AVAILABLE = False
    print('Warning: pytorch3d not installed.', 'Needed for Chamfer loss and mesh sampling.')
    print('See https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md for installation instructions.')

@systems.register('tsdf_system')
class TSDFSystem(BaseSystem):
    def prepare(self):
        self.n_frames = self.config.model.metadata.n_frames
        if self.config.model.capacity == 'n_frames':
            self.config.model.capacity = self.n_frames
        elif self.config.model.capacity.startswith('fraction'):
            fraction = float(self.config.model.capacity.split('_')[-1])
            self.config.model.capacity = int(fraction * self.n_frames)
            self.config.model.mode = 'interpolation'

        rnd_res = self.config.model.metadata.get('rnd_resolution', 512)
        if PYTORCH3D_AVAILABLE:
            self.mesh_renderer = MeshRenderer(rnd_res)

    def frame2time_step(self, frame_id):
        time_step = 2*(frame_id / (self.n_frames+2) - 0.5) #[-1.0,1.0]
        return time_step

    def preprocess_data(self, batch, stage):
        for key, val in batch.items():
            if torch.is_tensor(val):
                batch[key] = val.squeeze(0).to(self.device)

    def forward(self, coords, frame_id):
        # coords: (T, S, 4) txyz
        # frame_id: (T)
        # input_time: (T)
        # return: (T, S, 3))
        input_time = self.frame2time_step(frame_id)
        pred = self.model(coords, frame_id, input_time)
        return pred

    def training_step(self, batch, batch_idx):
        pred = self(batch['coords'], batch['frame_id'])
        loss = mape_loss(pred.reshape(-1, 1), batch['data'].reshape(-1, 1))
        self.log('train/loss', loss, prog_bar=True, rank_zero_only=True, sync_dist=True)
        return dict(loss=loss)

    def _isosurface(self, mesh_path, frame_id, resolution):
        time_step = self.frame2time_step(frame_id)
        assert time_step.numel() == 1 and frame_id.numel() == 1, 'Only support single time_step and frame_id'

        def _query_sdf(pts): # pts: Tensor with shape (n_pts, 3). Dtype=float32.
            pts = pts.view(1, -1, 3).to(frame_id.device)
            pts_time = torch.full_like(pts[..., :1], time_step)
            coords = torch.cat((pts_time, pts), dim=-1)
            BENCHMARK = False
            if BENCHMARK:
                # measure inference time 
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)

                with torch.no_grad():
                    B = 100
                    coord_tensor = torch.rand((B, 1000000, 4)).cuda()
                    inf_time = []
                    for b_ind in range(B):
                        _tensor = coord_tensor[b_ind].unsqueeze(0)
                        torch.cuda.synchronize()
                        start.record()
                        self.model(_tensor, frame_id=frame_id, input_time=time_step)
                        end.record()
                        torch.cuda.synchronize()
                        time_end = start.elapsed_time(end) #time.time() - time_start
                        inf_time.append(time_end)
                    inf_time = float(np.mean(inf_time[3:]))
                    print('Inference time: %f' % inf_time)

                exit(0)

            sdf = self.model(coords, frame_id=frame_id, input_time=time_step)
            return -sdf.view(-1)
        
        bound_min = torch.tensor([-1.0, -1.0, -1.0], dtype=torch.float32)
        bound_max = torch.tensor([ 1.0,  1.0,  1.0], dtype=torch.float32)
        mesh = extract_geometry(bound_min, bound_max, resolution=resolution, threshold=0, query_func=_query_sdf)
        mesh.export(mesh_path)
        return mesh

    def validation_step(self, batch, batch_idx, prefix='val'):
        frame_id = batch['frame_id']
        file_name = f"{int(frame_id.item()):06d}"
        mesh_path = self.get_save_path(f"meshes/it{self.global_step:06d}/{file_name}.ply")
        # extract and evaluate mesh
        pred_mesh = self._isosurface(mesh_path, frame_id, self.config.model.isosurface.resolution)

        torch_pred_mesh = Meshes(
            torch.from_numpy(pred_mesh.vertices).float()[None].to(self.device),
            torch.from_numpy(pred_mesh.faces).long()[None].to(self.device),
        )
        torch_gt_mesh = Meshes(batch['gt_vertices'][None].float(), batch['gt_faces'][None].float())

        metrics_dict = {}
        if PYTORCH3D_AVAILABLE:
            CD, ND = self.mesh_renderer.eval_mesh(torch_pred_mesh, torch_gt_mesh)

            metrics_dict = dict(metric_CD=CD, metric_ND=ND)

            # render mesh for visualization
            rnd_gt = self.mesh_renderer.render_mesh(torch_gt_mesh, mode='n').squeeze(0)[..., :3]
            rnd_pred = self.mesh_renderer.render_mesh(torch_pred_mesh, mode='n').squeeze(0)[..., :3]
            _log_imgs = [
                {'type': 'rgb', 'img': rnd_gt, 'kwargs': {'data_format': 'HWC'}},
                {'type': 'rgb', 'img': rnd_pred, 'kwargs': {'data_format': 'HWC'}},
            ]
            file_name = f"{batch['index'].squeeze().item():06d}"
            if prefix in ['test', 'predict']:
                self.save_image_grid(f"rgb_gt/it{self.global_step:06d}-{prefix}/{file_name}.png", [_log_imgs[-2]])
                self.save_image_grid(f"rgb/it{self.global_step:06d}-{prefix}/{file_name}.png", [_log_imgs[-1]])
            img = self.save_image_grid(f"it{self.global_step:06d}-{prefix}/{file_name}.png", _log_imgs)
            # log images to tensorboard
            if self.trainer.is_global_zero:
                if self.logger is not None:
                    if 'WandbLogger' in str(type(self.logger)):
                        self.logger.log_image(key=f'{prefix}/renderings', images=[img/255.], caption=["renderings"])
                    else:
                        self.logger.experiment.add_image(f'{prefix}/renderings', img/255., self.global_step, dataformats='HWC')

        metrics_dict['index'] = batch['index']
        return metrics_dict

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx, prefix='test')

    def predict_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx, prefix='predict')

    def on_predict_epoch_end(self, *args, **kwargs):
        if self.trainer.is_global_zero:
            prefix = 'predict'
            idir = f"it{self.global_step:06d}-{prefix}"
            self.save_img_sequence(idir, idir, '(\d+)\.png', save_format='mp4', fps=20)

    def on_validation_epoch_end(self, prefix='val'):
        out = self.all_gather(self.validation_step_outputs)
        if self.trainer.is_global_zero:
            metrics_dict = self._get_metrics_dict(out, prefix)
            return metrics_dict

    def on_test_epoch_end(self, prefix='test'):
        out = self.all_gather(self.test_step_outputs)
        if self.trainer.is_global_zero:
            metrics_dict = self._get_metrics_dict(out, prefix)
            res_path = self.get_save_path(f'results_it{self.global_step:06d}-{prefix}.yaml')
            with open(res_path, 'w') as file:
                yaml.dump(metrics_dict, file)

            idir = f"it{self.global_step:06d}-{prefix}"
            self.save_img_sequence(idir, idir, '(\d+)\.png', save_format='mp4', fps=20)

class MeshRenderer:
    """ Adapted from COAP [CVPR 2022]"""

    @torch.no_grad()
    def __init__(self, device, image_size=512):
        super().__init__()
        self.device = device
        camera_position = torch.from_numpy(np.array([1.0,  -1.0, 0.0])).float()
        R = look_at_rotation(camera_position[None], up=((0, 0, 1),)) # (1,3,3)
        t = -torch.bmm(R.transpose(1, 2), camera_position[None, :, None])[:, :, 0]   # (1, 3)

        cameras = renderer.FoVOrthographicCameras(R=R, T=t)
        lights = renderer.PointLights(
            location=[[0.0, 0.0, 3.0]],
            ambient_color=((1, 1, 1),), diffuse_color=((0, 0, 0),),
            specular_color=((0, 0, 0),)
        )
        raster_settings = renderer.RasterizationSettings(image_size=image_size, faces_per_pixel=100, blur_radius=0)
        rasterizer = renderer.MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
        shader = renderer.HardPhongShader(cameras=cameras, lights=lights)
        shader = renderer.SoftPhongShader(cameras=cameras, lights=lights)
        self.renderer = renderer.MeshRenderer(rasterizer=rasterizer, shader=shader)

    @torch.no_grad()
    def render_mesh(self, mesh, colors=None, mode='npat', flip_normal=False):
        """
        mode: normal, phong, texture
        """
        # mesh = Meshes(verts, faces)
        verts = mesh.verts_padded()
        faces = mesh.faces_padded()

        normals = torch.stack(mesh.verts_normals_list())
        front_light = torch.tensor([0, 0, -1]).float().to(verts.device)
        shades = (normals * front_light.view(1, 1, 3)).sum(-1).clamp(min=0).unsqueeze(-1).expand(-1, -1, 3)

        self.renderer.to(verts.device)
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=False):
            # normal
            if 'n' in mode:
                if flip_normal:
                    normals = -normals
                normals_vis = normals * 0.5 + 0.5
                mesh_normal = Meshes(verts, faces, textures=Textures(verts_rgb=normals_vis))
                image_normal = self.renderer(mesh_normal)
                return image_normal

            # shading
            if 'p' in mode:
                mesh_shading = Meshes(verts, faces, textures=Textures(verts_rgb=shades))
                image_phong = self.renderer(mesh_shading)
                return image_phong

            # albedo
            if 'a' in mode:
                assert (colors is not None)
                mesh_albido = Meshes(verts, faces, textures=Textures(verts_rgb=colors))
                image_color = self.renderer(mesh_albido)
                return image_color

            # albedo*shading
            if 't' in mode:
                assert (colors is not None)
                mesh_teture = Meshes(verts, faces, textures=Textures(verts_rgb=colors * shades))
                image_color = self.renderer(mesh_teture)
                return image_color

            raise NotImplementedError
    
    @torch.no_grad()
    def eval_mesh(self, pred_mesh, gt_mesh, num_samples:int = 100000): # pytorch3d meshes
        pred_pts, pred_normals = sample_points_from_meshes(pred_mesh, num_samples, return_normals=True)
        gt_pts, gt_normals = sample_points_from_meshes(gt_mesh, num_samples, return_normals=True)

        ch_dist, normal_dist = chamfer_distance(
            x=pred_pts.to(self.device),
            y=gt_pts.to(self.device),
            x_normals=pred_normals.to(self.device),
            y_normals=gt_normals.to(self.device),
            batch_reduction = "mean",
            point_reduction = "mean",
            norm=1,
        )
        return ch_dist, normal_dist

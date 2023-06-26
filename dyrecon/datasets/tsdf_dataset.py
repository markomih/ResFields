import os
import glob
import trimesh
import numpy as np
import pysdf
import torch
import pytorch_lightning as pl
import datasets

def anime_read(filename, normalize=True):
    """
    filename: .anime file
    return:
        nf: number of frames in the animation
        nv: number of vertices in the mesh (mesh topology fixed through frames)
        nt: number of triangle face in the mesh
        vert_data: [nv, 3], vertice data of the 1st frame (3D positions in x-y-z-order)
        face_data: [nt, 3], riangle face data of the 1st frame
        offset_data: [nf-1,nv,3], 3D offset data from the 2nd to the last frame
    """
    f = open(filename, 'rb')
    nf = np.fromfile(f, dtype=np.int32, count=1)[0]
    nv = np.fromfile(f, dtype=np.int32, count=1)[0]
    nt = np.fromfile(f, dtype=np.int32, count=1)[0]
    vert_data = np.fromfile(f, dtype=np.float32, count=nv * 3)
    face_data = np.fromfile(f, dtype=np.int32, count=nt * 3)
    offset_data = np.fromfile(f, dtype=np.float32, count=-1)
    vert_data = vert_data.reshape((-1, 3))
    face_data = face_data.reshape((-1, 3))
    offset_data = offset_data.reshape((nf - 1, nv, 3))
    def to_torch(x):
        x = x.astype(np.float32)
        return torch.from_numpy(x)
    nf, nv, nt, vert_data, face_data, offset_data = nf, nv, nt, to_torch(vert_data), to_torch(face_data), to_torch(offset_data)
    all_vertices = torch.stack([vert_data + offset_data[i] for i in range(offset_data.shape[0])])
    # normalize
    if normalize:
        vmin, vmax = all_vertices.min() - 0.01, all_vertices.max() + 0.01
        all_vertices = (all_vertices-vmin)/(vmax-vmin) * 2 - 1.
    meshes = [trimesh.Trimesh(all_vertices[_i].numpy(), face_data.numpy()) for _i in range(all_vertices.shape[0])] #Meshes(all_vertices, face_data[None].expand(all_vertices.shape[0], -1, -1))
    return meshes


class TSDFDatasetBase:
    def setup(self, config, split, load_time_steps=100000):
        path = config.path
        self.split = split
        self.clip_sdf = config.get('clip_sdf', None)
        meshes = self.load_meshes(path)[:load_time_steps]
        self.n_frames = len(meshes)
        frame_id = torch.arange(self.n_frames)
        print('Frames', self.n_frames)

        test_every = config.get('test_every', None)
        if test_every is not None:
            train_meshes, val_meshes = [], []
            train_frame_id, val_frame_id = [], []
            for i in range(len(meshes)):
                if (i + 1) % test_every == 0:
                    val_meshes.append(meshes[i])
                    val_frame_id.append(frame_id[i])
                else:
                    train_meshes.append(meshes[i])
                    train_frame_id.append(frame_id[i])

            train_meshes, train_frame_id = train_meshes, torch.tensor(train_frame_id)
            test_meshes, test_frame_id = val_meshes, torch.tensor(val_frame_id)
        else:
            train_meshes, train_frame_id = meshes, frame_id
            test_meshes, test_frame_id = meshes, frame_id

        if self.split == 'train':
            self.meshes, self.frame_id = train_meshes, train_frame_id
            self.sdf_fn = [pysdf.SDF(mesh.vertices, mesh.faces) for mesh in self.meshes]
            self.per_mesh_samples = (config.num_samples//train_frame_id.numel()) - (config.num_samples//train_frame_id.numel()) % 8
        elif self.split == 'test':
            self.meshes, self.frame_id = test_meshes, test_frame_id
        elif self.split == 'predict':
            self.meshes, self.frame_id = meshes, frame_id
        else:
            raise NotImplementedError
        print(self.split, self.frame_id)

    @staticmethod
    def load_meshes(path):
        if path.endswith('.anime'):
            return anime_read(path)
        elif os.path.isdir(path):
            mesh_paths = sorted(glob.glob(os.path.join(path, '*')))
            return [trimesh.load(_path) for _path in mesh_paths]
        else:
            raise NotImplementedError

    def frame2time_step(self, frame_id):
        time_step = 2*(frame_id / (self.n_frames+2) - 0.5) #[-1.0,1.0]
        return time_step

    def time_step2frame(self, time_step):
        frame_id = (time_step/2.+ 0.5)*(self.n_frames-1)
        return frame_id

    @staticmethod
    def collate_fn(batch):
        batch = torch.utils.data.default_collate(batch)
        for key in batch.keys():
            batch[key] = batch[key].view(batch[key].shape[0]*batch[key].shape[1], *batch[key].shape[2:])
        return batch

    def sample_data(self):
        assert self.split in ['train']
        # 1/2 points on the surface, 3/8 perturbed surface points, 1/8 uniform points
        n_samp = self.per_mesh_samples
        points_surface = np.stack([mesh.sample(n_samp*7//8) for mesh in self.meshes])
        points_surface[:, n_samp // 2:] += 0.01 * np.random.randn(*points_surface[:, n_samp // 2:].shape)

        # random
        points_uniform = np.random.rand(self.frame_id.shape[0], n_samp // 8, 3) * 2 - 1
        points = np.concatenate([points_surface, points_uniform], axis=1).astype(np.float32) # N_frames, n_samp, 3

        sdfs = np.zeros((points.shape[0], n_samp, 1))
        sdfs[:, n_samp // 2:] = np.stack([-self.sdf_fn[fid](points[fid, n_samp // 2:]).reshape(-1, 1) for fid in range(points.shape[0])]).astype(np.float32)

        # clip sdf
        if self.clip_sdf is not None:
            sdfs = sdfs.clip(-self.clip_sdf, self.clip_sdf)

        sdfs = torch.from_numpy(sdfs)
        points = torch.from_numpy(points)
        
        time_steps = self.frame2time_step(self.frame_id.view(-1, 1, 1).expand(-1, points.shape[1], -1))
        coords = torch.cat((time_steps, points), dim=-1)
        to_ret = {
            'coords': coords.float(), # T, S, 4
            'frame_id': self.frame_id.long(), # T
            'data': sdfs.float(), # T, S, 1
        }
        return  to_ret

class TSDFDataset(torch.utils.data.Dataset, TSDFDatasetBase):
    def __init__(self, config, split, load_time_steps=100000):
        self.setup(config, split, load_time_steps)

    def __len__(self):
        return len(self.meshes)
    
    def __getitem__(self, index):
        mesh = self.meshes[index]
        return dict(
            index=index,
            frame_id=torch.arange(self.n_frames)[index].long(),
            gt_vertices=torch.from_numpy(mesh.vertices).float(),
            gt_faces=torch.from_numpy(mesh.faces).long(),
        )


class TSDFIterableDataset(torch.utils.data.IterableDataset, TSDFDatasetBase):
    def __init__(self, config, split, load_time_steps=100000):
        self.setup(config, split, load_time_steps)

    def __iter__(self):
        while True:
            batch = self.sample_data()
            yield batch

@datasets.register('tsdf_dataset')
class TSDFDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    def setup(self, stage=None):
        load_time_steps = self.config.get('load_time_steps', 100000)
        if stage in [None, 'fit']:
            self.train_dataset = TSDFIterableDataset(self.config, 'train', load_time_steps)
        if stage in [None, 'fit', 'validate']:
            self.val_dataset = TSDFDataset(self.config, 'test', self.config.get('val_load_time_steps', load_time_steps))
        if stage in [None, 'test']:
            self.test_dataset = TSDFDataset(self.config, 'test', self.config.get('test_load_time_steps', load_time_steps))
        if stage in [None, 'predict']:
            self.predict_dataset = TSDFDataset(self.config, 'predict', self.config.get('test_load_time_steps', load_time_steps))

    @staticmethod
    def get_metadata(config):
        aabb = [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]
        load_time_steps = config.get('load_time_steps', 100000)
        n_frames = len(TSDFDatasetBase.load_meshes(config.path))
        return {
            'scene_aabb': aabb,
            'n_frames': min(n_frames, load_time_steps),
        }

    def prepare_data(self):
        pass
    
    def general_loader(self, dataset, batch_size):
        sampler = None
        return torch.utils.data.DataLoader(
            dataset, 
            num_workers=8,#os.cpu_count(), 
            batch_size=batch_size,
            pin_memory=True,
            sampler=sampler
        )
    
    def train_dataloader(self):
        return self.general_loader(self.train_dataset, batch_size=1)

    def val_dataloader(self):
        return self.general_loader(self.val_dataset, batch_size=1)

    def test_dataloader(self):
        return self.general_loader(self.test_dataset, batch_size=1) 

    def predict_dataloader(self):
        return self.general_loader(self.predict_dataset, batch_size=1)       

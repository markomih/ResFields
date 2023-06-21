import numpy as np
import skvideo.io
import torch
import datasets
from utils.misc import get_rank
import pytorch_lightning as pl
import skvideo.datasets

def load_video(path_to_video):
    if 'skvideo.datasets' in path_to_video:
        path_to_video = eval(path_to_video)()
    if 'npy' in path_to_video:
        vid = np.load(path_to_video)
    elif 'mp4' in path_to_video:
        vid = skvideo.io.vread(path_to_video).astype(np.float64) / 255.
    return vid # T,H,W,C

class VideoDatasetBase:

    def setup(self, config, split):
        assert split in ['train', 'test', 'predict']
        self.split = split
        vid = load_video(config.video_path)
        test_fraction = config.test_fraction

        sidelength = vid.shape[:-1] # T x H x W
        self.sidelength = sidelength
        all_mgrid = self._get_mgrid(sidelength, dim=3).view(sidelength[0], -1, 3) # Tx H*W x 3
        all_data = torch.from_numpy(vid).float().view(vid.shape[0], -1, 3) # T x H*W x 3
        n_val_samples_per_frame = int(test_fraction * all_mgrid.shape[1])
        self.n_frames = all_mgrid.shape[0]

        val_ids = np.stack([np.random.choice(np.arange(all_data.shape[1]), size=n_val_samples_per_frame, replace=False) for _ in range(self.n_frames)])
        train_ids = np.stack([np.delete(np.arange(all_data.shape[1]), val_ids[_i]) for _i in range(self.n_frames)])
        
        t_ind = np.arange(0, self.n_frames)[:, None]
        np.random.seed(42)
        val_tind, train_tind = t_ind.repeat(val_ids.shape[1], axis=1).flatten(), t_ind.repeat(train_ids.shape[1], axis=1).flatten()
        val_ids, train_ids = val_ids.reshape(-1), train_ids.reshape(-1)

        print(train_ids[77], train_ids[777], train_ids[7777], val_ids[77], val_ids[777], val_ids[7777])

        # split into train and val
        train_coords = all_mgrid[train_tind, train_ids].reshape(self.n_frames, -1, all_mgrid.shape[-1]) # T,-1,3
        train_data = all_data[train_tind, train_ids].reshape(self.n_frames, -1, all_data.shape[-1]) # T,-1,3

        val_coords = all_mgrid[val_tind, val_ids].reshape(self.n_frames, -1, all_mgrid.shape[-1]) # T,-1,3
        val_data = all_data[val_tind, val_ids].reshape(self.n_frames, -1, all_data.shape[-1]) # T,-1,3

        all_coords, all_data = all_mgrid, all_data

        # set parameters
        self.train_data = train_data.float()
        self.train_coords = train_coords.float()
        if split == 'train':
            self.sampling = config.sampling
            self.n_samples = self.sampling.n_samples // all_coords.shape[0]
            self.data = self.train_data
            self.coords = self.train_coords
        elif split == 'test':
            self.data = val_data
            self.coords = val_coords
        elif split == 'predict':
            self.data = all_data
            self.coords = all_coords

    @staticmethod
    def _get_mgrid(sidelen, dim=2):
        '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.'''
        if isinstance(sidelen, int):
            sidelen = dim * (sidelen,)

        if dim == 2:
            pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1]], axis=-1)[None, ...].astype(np.float32)
            pixel_coords[0, :, :, 0] = pixel_coords[0, :, :, 0] / (sidelen[0] - 1)
            pixel_coords[0, :, :, 1] = pixel_coords[0, :, :, 1] / (sidelen[1] - 1)
        elif dim == 3:
            pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1], :sidelen[2]], axis=-1)[None, ...].astype(np.float32)
            pixel_coords[..., 0] = pixel_coords[..., 0] / max(sidelen[0] - 1, 1)
            pixel_coords[..., 1] = pixel_coords[..., 1] / (sidelen[1] - 1)
            pixel_coords[..., 2] = pixel_coords[..., 2] / (sidelen[2] - 1)
        else:
            raise NotImplementedError('Not implemented for dim=%d' % dim)

        pixel_coords -= 0.5
        pixel_coords *= 2.
        pixel_coords = torch.Tensor(pixel_coords).view(-1, dim)
        return pixel_coords

    def sample_data(self):
        coords = self.coords # T,S,3
        data = self.data # T,S,3
        batch = dict()

        n_frames = coords.shape[0]
        frame_ids = torch.arange(n_frames, device=coords.device)
        if self.split in ['train']:
            t = frame_ids.unsqueeze(-1).repeat(1, self.n_samples).view(-1)
            y = torch.randint(0, coords.shape[1], size=(t.shape[0],), device=coords.device)
            coords = coords[t, y] # (F*S, 3)
            data = data[t, y]
        else:
            # add training coordinates to the batch to log the overfitting loss
            batch.update({
                'train_coords': self.train_coords.view(n_frames, -1, self.train_coords.shape[-1]).float(),
                'train_data': self.train_data.view(n_frames, -1, self.train_data.shape[-1]).float(),
            })
        batch.update({
            'coords': coords.view(n_frames, -1, coords.shape[-1]).float(), # T,S,3
            'data': data.view(n_frames, -1, data.shape[-1]).float(), # T,S,3
            'frame_ids': frame_ids.long(), # T
        })
        return batch

# torch wrappers
class VideoDataset(torch.utils.data.Dataset, VideoDatasetBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __len__(self):
        return 1
    
    def __getitem__(self, index):
        batch = self.sample_data()
        batch['index'] = index
        return batch

class VideoIterableDataset(torch.utils.data.IterableDataset, VideoDatasetBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __iter__(self):
        while True:
            batch = self.sample_data()
            yield batch

# pl wrapper
@datasets.register('video_dataset')
class VideoDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    def setup(self, stage=None):
        if stage in [None, 'fit']:
            self.train_dataset = VideoIterableDataset(self.config, 'train')
        if stage in [None, 'fit', 'validate']:
            self.val_dataset = VideoDataset(self.config, 'test')
        if stage in [None, 'test']:
            self.test_dataset = VideoDataset(self.config, 'test')
        if stage in [None, 'predict']:
            self.predict_dataset = VideoDataset(self.config, 'predict')

    @staticmethod
    def get_metadata(config):
        n_frames = load_video(config.video_path).shape[0]
        return {'n_frames': n_frames}

    def prepare_data(self):
        pass
    
    def general_loader(self, dataset, batch_size):
        sampler = None
        return torch.utils.data.DataLoader(
            dataset, 
            num_workers=5,
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

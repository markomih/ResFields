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
        vid = skvideo.io.vread(path_to_video).astype(np.single) / 255.
    return vid # T,H,W,C

class VideoDatasetBase:

    def setup(self, config, split):
        assert split in ['train', 'test', 'predict']

        vid = load_video(config.video_path)
        test_every = config.test_every

        sidelength = vid.shape[:-1] # T x H x W
        all_mgrid = self._get_mgrid(sidelength, dim=3).view(*sidelength, 3) # TxHxW x 3
        all_data = torch.from_numpy(vid) # TxHxW x 3

        val_ids = np.arange(0, all_mgrid.shape[2], test_every)
        train_ids = np.delete(np.arange(0, all_mgrid.shape[2]), val_ids)

        train_coords, train_data = all_mgrid[:, :, train_ids], all_data[:, :, train_ids]
        val_coords, val_data = all_mgrid[:, :, val_ids], all_data[:, :, val_ids]
        all_coords, all_data = all_mgrid, all_data

        rank = get_rank()

        # set parameters
        self.train_data = train_data.to(rank)
        self.train_coords = train_coords.to(rank)

        if split == 'train':
            self.data = train_data.to(rank)
            self.coords = train_coords.to(rank)
        elif split == 'test':
            self.data = val_data.to(rank)
            self.coords = val_coords.to(rank)
        elif split == 'predict':
            self.data = all_data.to(rank)
            self.coords = all_coords.to(rank)

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

# torch wrappers
class VideoDataset(torch.utils.data.Dataset, VideoDatasetBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __len__(self):
        return 1
    
    def __getitem__(self, index):
        return {
            'index': index
        }

class VideoIterableDataset(torch.utils.data.IterableDataset, VideoDatasetBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __iter__(self):
        while True:
            yield {}

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
            num_workers=0,
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

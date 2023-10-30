import torch
import torch.nn as nn

from utils.misc import get_rank
from models.utils import chunk_batch, Updateable

class BaseModel(nn.Module, Updateable):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.rank = get_rank()
        self.setup()
        if self.config.get('weights', None):
            self.load_state_dict(torch.load(self.config.weights))
        self.register_buffer("_dummy", torch.empty(0), persistent=False)

    @property
    def device(self) -> torch.device:
        return self._dummy.device

    def setup(self):
        raise NotImplementedError
    
    def log_variables(self):
        return dict()
    # def train(self, mode=True):
    #     self.randomized = mode and self.config.model.sampling.randomized
    #     return super().train(mode=mode)
    
    # def eval(self):
    #     self.randomized = False
    #     return super().eval()

    def regularizations(self, out):
        to_ret = {}
        for attr in self.__dir__():
            module = getattr(self, attr)
            if isinstance(module, BaseModel):
                to_ret.update(module.regularizations(out))
        return to_ret

    def forward(self, rays, **kwargs):
        if self.training:
            out = self.forward_(rays, **kwargs)
        else:
            out = chunk_batch(self.forward_, self.config.ray_chunk, rays, **kwargs)
        return {
            **out,
        }

    def forward_(self, rays, **kwargs):
        raise NotImplementedError
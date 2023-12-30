import pyro
import torch
import numpy as np
from abc import abstractmethod

class BaseModel(pyro.nn.PyroModule):
    """
    Base class for all models
    """
    @abstractmethod
    def forward(self, *inputs):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def resume_from_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        checkpoint = torch.load(resume_path)
        self.load_state_dict(checkpoint['state_dict'])

class DensityKernel(BaseModel, pyro.distributions.Distribution):
    def __init__(self):
        self.has_enumerate_support = self.has_enumerate_support
        self.has_rsample = self.density.has_rsample

    @property
    @abstractmethod
    def density(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self):
        raise NotImplementedError

    def log_prob(self, x, *args, **kwargs):
        return self.density.log_prob(x, *args, **kwargs)

    def sample(self, *args, **kwargs):
        return self.density.sample(*args, **kwargs)

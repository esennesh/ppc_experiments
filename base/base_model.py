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

class MarkovKernel(pyro.nn.PyroModule):
    """
    Base class for Markov kernels that output a Pyro distribution
    """
    @property
    def event_dim(self):
        raise NotImplementedError

    @abstractmethod
    def forward(self, *args, **kwargs) -> pyro.distributions.Distribution:
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

class PartialMarkovKernel:
    def __init__(self, kernel: MarkovKernel, *args, **kwargs):
        self._args = args
        self._kernel = kernel
        self._kwargs = kwargs
        self.batch_shape = ()

    @property
    def event_dim(self):
        return self.kernel.event_dim

    def __call__(self, *args, **kwargs) -> pyro.distributions.Distribution:
        self.kernel.batch_shape = self.batch_shape
        kwargs = {**self._kwargs, **kwargs}
        return self.kernel(*self._args, *args, **kwargs)

    @property
    def kernel(self):
        return self._kernel

    def __str__(self):
        return str(self.kernel)

import functools
import lightning as L
import math
import numpy as np
import pyro
from pyro.infer import Importance, Predictive, SVI, JitTraceGraph_ELBO, TraceGraph_ELBO
import torch
from torchvision.utils import make_grid
from model import metric
from model.inference import ParticleDict, PpcGraphicalModel
from utils import inf_loop, MetricTracker
import utils

class LightningSvi(L.LightningModule):
    def __init__(self, importance, data: L.LightningDataModule, jit=False,
                 lr=1e-3, num_particles=4):
        super().__init__()
        self.importance = importance
        self.lr = lr
        self.num_particles = num_particles

        if jit:
            elbo = JitTraceGraph_ELBO(num_particles=self.num_particles,
                                      max_plate_nesting=1,
                                      vectorize_particles=True)
        else:
            elbo = TraceGraph_ELBO(num_particles=self.num_particles,
                                   max_plate_nesting=1,
                                   vectorize_particles=True)
        self.elbo = elbo(self.importance.model, self.importance.guide)
        self.predictive = Predictive(self.importance.model,
                                     guide=self.importance.guide,
                                     num_samples=self.num_particles)

    def configure_optimizers(self):
        return torch.optim.Adam(self.elbo.parameters(), amsgrad=True,
                                lr=self.lr, weight_decay=0.)

    def forward(self, *args, **kwargs):
        return self.predictive(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        """
        Training logic for an epoch

        :param batch: Batch of training data for current training epoch.
        :return: Loss in this epoch.
        """
        data, target = batch
        loss = self.elbo(data)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation logic for an epoch

        :param batch: Batch of training data for current validation epoch.
        :return: Loss in this epoch.
        """
        data, target = batch
        loss = self.elbo(data)
        self.log("valid/loss", loss)
        return loss

class LightningPpc(L.LightningModule):
    """
    Lightning module for Population Predictive Coding (PPC)
    """
    def __init__(self, graph: PpcGraphicalModel, data: L.LightningDataModule,
                 cooldown=50, factor=0.9, lr=1e-3, num_particles=4,
                 num_sweeps=1, patience=100):
        super().__init__()
        self.save_hyperparameters(ignore=["data", "graph"])
        self.cooldown = cooldown
        self.factor = factor
        self.lr = lr
        self.graph = graph
        self.num_particles = num_particles
        self.num_sweeps = num_sweeps
        self.patience = patience
        self.predictive = Predictive(self.graph.model, guide=self.graph.guide,
                                     num_samples=self.num_particles)

        num_train = len(data.train_dataloader().dataset)
        num_valid = len(data.val_dataloader().dataset)
        self.particles = {"train": ParticleDict(num_train, num_particles),
                          "valid": ParticleDict(num_valid, num_particles)}
        for batch_idx, batch in enumerate(data.train_dataloader()):
            self._initialize_particles(batch, batch_idx)
        for batch_idx, batch in enumerate(data.val_dataloader()):
            self._initialize_particles(batch, batch_idx, False)

    def _initialize_particles(self, batch, batch_idx, train=True):
        data, target, indices = batch
        self.graph(data, lr=self.lr, P=self.num_particles, prior=True)
        self._save_particles(indices, train)

    def _load_particles(self, indices, train=True):
        particles = self.particles["train" if train else "valid"]
        for site in particles:
            self.graph.update(site, particles.get_particles(site, indices))

    def _save_particles(self, indices, train=True):
        particles = self.particles["train" if train else "valid"]
        for site in self.graph.stochastic_nodes:
            particles.set_particles(site, indices,
                                    self.graph.nodes[site]['value'].detach())

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.graph.parameters(), amsgrad=True,
                                     lr=self.lr, weight_decay=0.)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, cooldown=self.cooldown, factor=self.factor,
            patience=self.patience
        )
        return {"lr_scheduler": lr_scheduler, "monitor": "valid/loss",
                "optimizer": optimizer}

    def on_load_checkpoint(self, checkpoint):
        self.particles = checkpoint["particle_dicts"]

    def on_save_checkpoint(self, checkpoint):
        checkpoint["particle_dicts"] = self.particles

    def ppc_step(self, data):
        for _ in range(self.num_sweeps - 1):
            self.graph(data, lr=self.lr, P=self.num_particles)
        return self.graph(data, lr=self.lr, P=self.num_particles)

    def training_step(self, batch, batch_idx):
        data, _, indices = batch
        self._load_particles(indices, train=True)
        trace, log_weight = self.ppc_step(data)
        loss = -log_weight.mean()
        self._save_particles(indices, train=True)

        self.log("train/ess", metric.ess(trace, log_weight.detach()))
        self.log("train/log_joint", metric.log_joint(trace,
                                                     log_weight.detach()))
        self.log("train/log_marginal", metric.log_marginal(trace,
                                                           log_weight.detach()))
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        data, _, indices = batch
        self._load_particles(indices, train=False)
        trace, log_weight = self.ppc_step(data)
        loss = -log_weight.mean()
        self._save_particles(indices, train=False)

        self.log("valid/ess", metric.ess(trace, log_weight.detach()),
                 sync_dist=True)
        self.log("valid/log_joint", metric.log_joint(trace,
                                                     log_weight.detach()),
                 sync_dist=True)
        self.log("valid/log_marginal", metric.log_marginal(trace,
                                                           log_weight.detach()),
                 sync_dist=True)
        self.log("valid/loss", loss, sync_dist=True)
        return loss
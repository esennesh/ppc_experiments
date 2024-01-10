import functools
import math
import networkx as nx
import pyro
import pyro.distributions as dist
import pyro.nn as pnn
import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel, MarkovKernel

class DigitPositions(MarkovKernel):
    def __init__(self, z_where_dim=2):
        super().__init__()
        self.register_buffer('loc', torch.zeros(z_where_dim))
        self.register_buffer('scale', torch.ones(z_where_dim) * 0.2)

    def forward(self, z_where, K=3, batch_shape=()) -> dist.Distribution:
        scale = self.scale
        if z_where is None:
            scale = scale * 5
        prior = dist.Normal(self.loc, scale).expand([
            *batch_shape, K, *self.loc.shape
        ])
        return prior.to_event(2)

class DigitFeatures(MarkovKernel):
    def __init__(self, z_what_dim=10):
        super().__init__()
        self.register_buffer('loc', torch.zeros(z_what_dim))
        self.register_buffer('scale', torch.ones(z_what_dim))

    def forward(self, K=3, batch_shape=()) -> dist.Distribution:
        prior = dist.Normal(self.loc, self.scale).expand([
            *batch_shape, K, *self.loc.shape
        ])
        return prior.to_event(2)

class DigitsDecoder(MarkovKernel):
    def __init__(self, digit_side=28, hidden_dim=400, x_side=96, z_what_dim=10):
        super().__init__()
        self._digit_side = digit_side
        self._x_side = x_side
        self.decoder = nn.Sequential(
            nn.Linear(z_what_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, digit_side ** 2), nn.Sigmoid()
        )
        scale = torch.diagflat(torch.ones(2) * x_side / digit_side)
        self.register_buffer('scale', scale)
        self.translate = (x_side - digit_side) / digit_side

    def blit(self, digits, z_where):
        P, B, K, _ = z_where.shape
        affine_p1 = self.scale.repeat(P, B, K, 1, 1)
        affine_p2 = z_where.unsqueeze(-1) * self.translate
        affine_p2[:, :, :, 0, :] = -affine_p2[:, :, :, 0, :]
        grid = F.affine_grid(
            torch.cat((affine_p1, affine_p2), -1).view(P*B*K, 2, 3),
            torch.Size((P*B*K, 1, self._x_side, self._x_side)),
            align_corners=True
        )

        digits = digits.view(P*B*K, self._digit_side, self._digit_side)
        frames = F.grid_sample(digits.unsqueeze(1), grid, mode='nearest',
                               align_corners=True).squeeze(1)
        return frames.view(P, B, K, self._x_side, self._x_side)

    def forward(self, what, where) -> dist.Distribution:
        P, B, K, _ = where.shape
        digits = self.decoder(what)
        frame = torch.clamp(self.blit(digits, where).sum(-3), 0., 1.)
        return dist.ContinuousBernoulli(frame).to_event(2)

class DigitDecoder(MarkovKernel):
    def __init__(self, digit_side=28, hidden_dim=400, z_dim=10):
        super().__init__()
        self._digit_side = 28
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, digit_side ** 2), nn.Sigmoid()
        )

    def forward(self, what, x=None) -> dist.Distribution:
        P, B, _, _ = what.shape
        estimate = self.decoder(what).view(P, B, 1, self._digit_side,
                                           self._digit_side)
        return dist.ContinuousBernoulli(estimate).to_event(3)

class GraphicalModel(BaseModel, pnn.PyroModule):
    def __init__(self):
        super().__init__()
        self._graph = nx.DiGraph()

    def add_node(self, site, parents, kernel):
        self._graph.add_node(site, event_dim=0, is_observed=False,
                             kernel=kernel, kwargs={}, value=None)
        for parent in parents:
            self._graph.add_edge(parent, site)

    def child_sites(self, site):
        return self._graph.successors(site)

    def get_kwargs(self, site):
        return self.nodes[site]['kwargs']

    def kernel(self, site):
        return functools.partial(self.nodes[site]['kernel'],
                                 **self.get_kwargs(site))

    def log_prob(self, site, value, *args, **kwargs):
        density = self.kernel(site)(*args, **kwargs)
        return density.log_prob(value)

    @property
    def nodes(self):
        return self._graph.nodes

    def parent_sites(self, site):
        return self._graph.predecessors(site)

    def parent_vals(self, site):
        return tuple(self.nodes[p]['value'] for p in self.parent_sites(site))

    def set_kwargs(self, site, **kwargs):
        self.nodes[site]['kwargs'] = kwargs

    def forward(self, **kwargs):
        results = []
        for site in nx.lexicographical_topological_sort(self._graph):
            density = self.kernel(site)(*self.parent_vals(site))
            self.nodes[site]['event_dim'] = density.event_dim
            self.nodes[site]['value'] = pyro.sample(site, density,
                                                    obs=kwargs.get(site, None))
            self.nodes[site]['is_observed'] = site in kwargs

            if len(list(self.child_sites(site))) == 0:
                results.append(self.nodes[site]['value'])
        return results[0] if len(results) == 1 else tuple(results)
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
    def __init__(self, num_digits=3, z_where_dim=2):
        super().__init__()
        self.register_buffer('loc', torch.zeros(z_where_dim))
        self.register_buffer('scale', torch.ones(z_where_dim) * 0.2)
        self.batch_shape = ()
        self._num_digits = num_digits

    def forward(self, z_where, batch_shape=()) -> dist.Distribution:
        param_shape = (*self.batch_shape, self._num_digits, *self.loc.shape)
        scale = self.scale.expand(param_shape)
        if z_where is None:
            z_where = self.loc.expand(param_shape)
            scale = scale * 5
        return dist.Normal(z_where, scale).to_event(2)

class DigitFeatures(MarkovKernel):
    def __init__(self, num_digits=3, z_what_dim=10):
        super().__init__()
        self.register_buffer('loc', torch.zeros(z_what_dim))
        self.register_buffer('scale', torch.ones(z_what_dim))
        self.batch_shape = ()
        self._num_digits = num_digits

    def forward(self, batch_shape=()) -> dist.Distribution:
        dist_shape = (*self.batch_shape, self._num_digits, *self.loc.shape)
        return dist.Normal(self.loc, self.scale).expand(dist_shape).to_event(2)

class DigitsDecoder(MarkovKernel):
    def __init__(self, digit_side=28, hidden_dim=400, x_side=96, z_what_dim=10):
        super().__init__()
        self.batch_shape = ()
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
        self.batch_shape = ()
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

    def clamp(self, site, value):
        self.nodes[site]['is_observed'] = True
        return self.update(site, value)

    def clear(self):
        for site in self.nodes:
            for key in self.nodes[site]:
                if key != "kernel":
                    self.nodes[site][key] = None

    def kernel(self, site):
        return self.nodes[site]['kernel']

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

    @functools.cached_property
    def stochastic_nodes(self):
        return [site for site in self.nodes
                if not self.nodes[site]['is_observed']]

    @functools.cache
    def topological_sort(self, reverse=False):
        nodes = list(nx.lexicographical_topological_sort(self._graph))
        if reverse:
            nodes = list(reversed(nodes))
        return nodes

    def update(self, site, value):
        self.nodes[site]['value'] = value
        return self.nodes[site]['value']

    def forward(self, batch_shape=(), **kwargs):
        results = ()
        for site in self.topological_sort():
            kernel = self.kernel(site)
            density = kernel(*self.parent_vals(site))
            self.nodes[site]['event_dim'] = density.event_dim
            self.nodes[site]['value'] = pyro.sample(site, density,
                                                    obs=kwargs.get(site, None))
            self.nodes[site]['is_observed'] = site in kwargs

            if len(list(self.child_sites(site))) == 0:
                results = results + (self.nodes[site]['value'],)
        return results[0] if len(results) == 1 else results

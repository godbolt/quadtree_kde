
import numpy as n
from kde.cudakde import bootstrap_kde, gaussian_kde

from .qfakde.code.common import Point
from .qfakde.code.quadtree import DynamicQuadTree

__license__ = """
MIT License

Copyright (c) 2020 Bryan Godbolt

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

class quadtree_kde(gaussian_kde):
    def __init__(self, data,
                 quadtree_granularity=20,
                 quadtree_max_depth=10,
                 bandwidth_factor_power=0.5,
                 use_cuda=False):

        gaussian_kde.__init__(self, data, use_cuda=use_cuda)

        x_max = n.max(data[0, :])
        x_min = n.min(data[0, :])
        y_max = n.max(data[1, :])
        y_min = n.min(data[1, :])
        x_center = (x_max - x_min) / 2 + x_min
        y_center = (y_max - y_min) / 2 + y_min
        dimension = max((x_max - x_min) / 2, (y_max - y_min) / 2)
        num_data_points = len(data[0])
        max_points_per_node = pow(num_data_points, 1 / 2)
        qt = DynamicQuadTree(centerPt=Point(x_center, y_center, 'center'),
                             dimension=dimension,  max_points=max_points_per_node / quadtree_granularity,  max_depth=quadtree_max_depth)
        for i in range(len(data[0])):
            qt.insert(Point(data[0,i], data[1,i], i))

        bandwidth_scale_factor = {}
        bandwidth_scale_factor = self.get_lambdas_from_quadtree(
            qt.root,
            bandwidth_scale_factor)
        
        self.lambdas = pow(n.array([bandwidth_scale_factor[k] for k
                                    in sorted(bandwidth_scale_factor)]),
                           bandwidth_factor_power)

    def get_lambdas_from_quadtree(self, node, lambdas):
        d = node.boundary.dimension
        pts = list(node._points)
        for pt in pts:
            lambdas[pt.key] = d
        for region in node._nodes:
            lambdas = self.get_lambdas_from_quadtree(
                node._nodes[region],
                lambdas)
        return lambdas

    def __call__(self, grid_points):
        return gaussian_kde.__call__(self, grid_points)


class quadtree_bootstrap_kde(bootstrap_kde):
    def __init__(self, data, num_bootstraps, **kwargs):
        assert int(num_bootstraps) == float(num_bootstraps)
        num_bootstraps = int(num_bootstraps)

        self.kernels = []
        self.bootstrap_indices = []

        self.data = n.atleast_2d(data)
        self.d, self.n = self.data.shape
        self.weighted = False

        for _ in range(num_bootstraps):
            indices = n.array(self.get_bootstrap_indices())
            self.bootstrap_indices.append(indices)
            kernel = quadtree_kde(data[..., indices], **kwargs)
            self.kernels.append(kernel)

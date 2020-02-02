import numpy as np
from pykdtree.kdtree import KDTree


class KDResampler:

    def __init__(self, distance_limit=0.05, leafsize=32):
        self.distance_limit = distance_limit
        self.leafsize = leafsize
        self._invalid_mask = None
        self._indices = None

    @staticmethod
    def make_target_coords(georange, width, height, pad=0., ratio=1.02):
        latmin, latmax, lonmin, lonmax = georange
        image_width = int(width * ratio)
        image_height = int(height * ratio)
        ix = np.linspace(lonmin-pad, lonmax+pad, image_width)
        iy = np.linspace(latmax+pad, latmin-pad, image_height)
        return np.meshgrid(ix, iy), (lonmin-pad, lonmax+pad, latmin-pad, latmax+pad)

    def build_tree(self, lons, lats):
        self.tree = KDTree(np.dstack((lons.ravel(), lats.ravel()))[0], leafsize=self.leafsize)

    def resample(self, data, target_x, target_y):
        if self._indices is None:
            target_coords = np.dstack((target_x.ravel(), target_y.ravel()))[0]
            _, self._indices = self.tree.query(target_coords,
                distance_upper_bound=self.distance_limit)
            self._invalid_mask = self._indices == self.tree.n # beyond distance limit
            self._indices[self._invalid_mask] = 0
        remapped = np.array(data.reshape((-1, 3))[self._indices])
        remapped[self._invalid_mask] = 0
        remapped = remapped.reshape((*target_x.shape, 3))
        return remapped

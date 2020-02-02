import os

import dask.array as da
import h5py
import numpy as np
from geotiepoints.multilinear import MultilinearInterpolator


_url = 'https://zenodo.org/record/1288441/files/pyspectral_atm_correction_luts_marine_clean_aerosol.tgz'


class RayleighCorrector:

    _lut_dir = 'lut'

    def __init__(self, atmosphere='us-standard',
            aerosol_type='marine_clean_aerosol'):
        self.atmosphere = atmosphere
        self.aerosol_type = aerosol_type
        self._rayl = None
        self._wvl_coord = None
        self._azid_coord = None
        self._satz_sec_coord = None
        self._sunz_sec_coord = None

    def set_coord(self, lat, lon, utc_time):
        self.lat = lat
        self.lon = lon
        self.time = utc_time
        self.get_angles()

    def get_angles(self):
        from astronomy import get_alt_az, get_observer_look, sun_zenith_angle
        from hsd import GEOS_HEIGHT, HIM8_SUBLON
        lons = da.where(self.lon >= 1e30, np.nan, self.lon)
        lats = da.where(self.lat >= 1e30, np.nan, self.lat)
        sunalt, suna = get_alt_az(self.time, lons, lats)
        suna = da.rad2deg(suna)
        self.sunz = sun_zenith_angle(self.time, lons, lats)
        sata, satel = get_observer_look(HIM8_SUBLON, 0, GEOS_HEIGHT / 1000.0,
            self.time, lons, lats, 0)
        self.satz = 90 - satel
        sata = sata % 360.
        suna = suna % 360.
        self.ssadiff = da.absolute(suna - sata)
        self.ssadiff = da.minimum(self.ssadiff, 360 - self.ssadiff)

    def correct(self, data, wavelength, redband=None):
        ref = self.get_reflectance(wavelength, redband=redband)
        return data - ref

    def prepare_lut(self):
        filename = f'rayleigh_lut_{self.atmosphere}.h5'
        self.lutfile = os.path.join(self._lut_dir, filename)
        if not os.path.exists(self.lutfile):
            import tarfile
            filename = f'pyspectral_atm_correction_luts_{self.aerosol_type}.tgz'
            filename = os.path.join(self._lut_dir, filename)
            if not os.path.exists(filename):
                raise FileNotFoundError(f'LUT file not found. Please download it from {_url} and put it in lut/ directory')
            tar = tarfile.open(filename)
            tar.extractall(self._lut_dir)
            tar.close()

    def get_lut(self):
        if self._rayl is None:
            self.prepare_lut()
            h5f = h5py.File(self.lutfile, 'r')
            tab = h5f['reflectance']
            wvl = h5f['wavelengths']
            azidiff = h5f['azimuth_difference']
            satellite_zenith_secant = h5f['satellite_zenith_secant']
            sun_zenith_secant = h5f['sun_zenith_secant']
            self._rayl = da.from_array(tab, chunks=(10, 10, 10, 10))
            self._wvl_coord = wvl[:]  # no benefit to dask-ifying this
            self._azid_coord = da.from_array(azidiff, chunks=(1000,))
            self._satz_sec_coord = da.from_array(satellite_zenith_secant,
                chunks=(1000,))
            self._sunz_sec_coord = da.from_array(sun_zenith_secant,
                chunks=(1000,))

    def get_reflectance(self, wavelength, redband=None):
        self.get_lut()
        clip_angle = da.rad2deg(da.arccos(1. / self._sunz_sec_coord.max()))
        sun_zenith = da.clip(self.sunz, 0, clip_angle)
        sunzsec = 1. / da.cos(da.deg2rad(sun_zenith))
        clip_angle = da.rad2deg(da.arccos(1. / self._satz_sec_coord.max()))
        sat_zenith = da.clip(self.satz, 0, clip_angle)
        satzsec = 1. / da.cos(da.deg2rad(sat_zenith))
        shape = sun_zenith.shape
        if not (self._wvl_coord.min() < wavelength < self._wvl_coord.max()):
            return da.zeros(shape, chunks=sun_zenith.chunks)
        idx = np.searchsorted(self._wvl_coord, wavelength)
        wvl1 = self._wvl_coord[idx - 1]
        wvl2 = self._wvl_coord[idx]
        fac = (wvl2 - wavelength) / (wvl2 - wvl1)
        raylwvl = fac * self._rayl[idx - 1, :, :, :] + (1 - fac) * \
            self._rayl[idx, :, :, :]
        smin = [self._sunz_sec_coord[0], self._azid_coord[0],
            self._satz_sec_coord[0]]
        smax = [self._sunz_sec_coord[-1], self._azid_coord[-1],
            self._satz_sec_coord[-1]]
        orders = [len(self._sunz_sec_coord), len(self._azid_coord),
            len(self._satz_sec_coord)]
        f_3d_grid = da.atleast_2d(raylwvl.ravel())
        smin, smax, orders, f_3d_grid = da.compute(smin, smax, orders, f_3d_grid)
        minterp = MultilinearInterpolator(smin, smax, orders)
        minterp.set_values(f_3d_grid)
        ipn = da.map_blocks(self._do_interp, minterp, sunzsec, self.ssadiff,
            satzsec, dtype=raylwvl.dtype, chunks=self.ssadiff.chunks)
        ipn *= 100
        res = ipn
        if redband is not None:
            res = da.where(redband < 0.2, res, (1 - (redband - 0.2) / 0.8) * res)
        res = da.clip(res, 0, 100) / 100
        return res

    @staticmethod
    def _do_interp(minterp, sunzsec, azidiff, satzsec):
        interp_points2 = da.vstack((sunzsec.ravel(), 180 - azidiff.ravel(),
            satzsec.ravel()))
        res = minterp(interp_points2)
        return res.reshape(sunzsec.shape)


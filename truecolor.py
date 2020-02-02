import datetime
import logging
import os
import sys

import numpy as np
from PIL import Image, ImageEnhance

from correction import sun_zenith_correction
from hsd import MutilSegmentHimawariFormat, get_segno, get_standard_filename
from kdresampler import KDResampler
from rayleigh import RayleighCorrector


class TrueColor:

    __bands__ = [1, 2, 3, 4]
    __wavelengths__ = {
        1: 470.63,
        2: 510.00,
        3: 639.14,
        4: 856.70
    } # unit: nm, refer: https://www.data.jma.go.jp/mscweb/en/himawari89/space_segment/spsg_ahi.html

    def __init__(self, filedir=None, time=None, region=None, georange=None,
            imsize=None, projection=None, downsample=True, nir_fraction=0.07,
            export_filename=None, enh_gamma=0.45, enh_color=1.5,
            enh_contrast=1.35):
        """True color compositor for Himawari-8.
        
        Parameters
        ----------
        filedir : str, optional
            location of source data, by default None
        time : `datetime.datetime` object, optional
            utc time of image, by default None
        region : str, optional
            ** currently not implemented **
            region of satellite data, by default None (full disk)
        georange : tuple, optional
            tuple of (latmin, latmax, lonmin, lonmax), by default None
        imsize : int, optional
            width of exported image, by default None
        projection : dict, optional
            ** currently not implemented **
            projection information for exported image, by default None
        downsample : bool, optional
            ** currently not implemented **
            whether downsample red channel or upsample other channels,
            by default True
        nir_fraction : float, optional
            fraction of nir channel in combined green channel, by default 0.07
        export_filename : str, optional
            filename of exported image, by default None
        enh_gamma : float, optional
            gamma coefficient, by default 0.45
        enh_color : float, optional
            color (saturation) coefficient, by default 1.5
        enh_contrast : float, optional
            contrast coefficient, by default 1.35
        """
        self.filedir = filedir
        self.time = time
        self.region = region
        self.georange = georange
        self.imsize = imsize
        self.projection = projection
        self.downsample = downsample
        self.nir_fraction = nir_fraction
        self.export_filename = export_filename
        self.enh_gamma = enh_gamma
        self.enh_color = enh_color
        self.enh_contrast = enh_contrast
        #init
        self._arrays = {}
        self._rayleigh = RayleighCorrector()
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter('[%(asctime)s] %(message)s'))
        self.logger.addHandler(handler)

    def make(self):
        try:
            self.check_file_exists()
            self.extract_raw()
            self.sun_zen_corr()
            self.combine_green()
            self.rayleigh_corr()
            self.concat_bands()
            xy = self.establish_proj()
            self.resample(xy)
            self.enhance_image()
            self.export_image()
        except Exception as exp:
            self.logger.exception('Fatal error!')

    def check_file_exists(self):
        self.logger.info('Check if all files are ready...')
        self.segnos, self.vlines, self.vcols = get_segno(self.georange)
        for band in self.__bands__:
            for segno in self.segnos:
                filename = get_standard_filename(self.time, self.region,
                    band, segno)
                filename = os.path.join(self.filedir, filename)
                if not os.path.exists(filename):
                    raise FileNotFoundError(f'Missing file: {filename}')

    def extract_raw(self):
        # Blue = Band1
        self.logger.info('Extract blue channel...')
        hf = self.get_hf(1)
        raw = hf.extract(vline=self.vlines, vcol=self.vcols)
        self._arrays['b'] = raw
        aline = hf.linenos
        acol = hf.colnos
        lons, lats = hf.get_geocoord(aline=aline, acol=acol)
        self._arrays['lons'] = lons
        self._arrays['lats'] = lats
        # Green
        # = 0.85 * Band2 + 0.15 * Band4
        self.logger.info('Extract green channel...')
        hf = self.get_hf(2)
        raw = hf.extract(aline=aline, acol=acol)
        self._arrays['g'] = raw
        self.logger.info('Extract nir channel (Band 04)...')
        hf = self.get_hf(4)
        raw = hf.extract(aline=aline, acol=acol)
        self._arrays['g4'] = raw
        # Red = Band3
        self.logger.info('Extract red channel...')
        aline = aline[0] * 2, aline[1] * 2
        acol = acol[0] * 2, acol[1] * 2
        hf = self.get_hf(3)
        raw = hf.extract(aline=aline, acol=acol)
        # Downsample by 2×2 mean, hack!
        # And that's why we use aline/acol instead of vline/vcol, we need to
        # ensure array can be downsampled and exactly aligned to other bands.
        self.logger.info('Downsample red channel...')
        raw = raw.reshape((raw.shape[0]//2, 2, raw.shape[1]//2, 2))
        raw = raw.mean(axis=(1, 3))
        self._arrays['r'] = raw

    def sun_zen_corr(self):
        self.logger.info('Process sun zenith correction on blue channel...')
        self._arrays['b'] = sun_zenith_correction(self._arrays['b'],
            self.time, self._arrays['lons'], self._arrays['lats'])
        self.logger.info('Process sun zenith correction on green channel...')
        self._arrays['g'] = sun_zenith_correction(self._arrays['g'],
            self.time, self._arrays['lons'], self._arrays['lats'])
        self.logger.info('Process sun zenith correction on nir channel...')
        self._arrays['g4'] = sun_zenith_correction(self._arrays['g4'],
            self.time, self._arrays['lons'], self._arrays['lats'])
        self.logger.info('Process sun zenith correction on red channel...')
        self._arrays['r'] = sun_zenith_correction(self._arrays['r'],
            self.time, self._arrays['lons'], self._arrays['lats'])

    def combine_green(self):
        self.logger.info('Process rayleigh correction on green channel...')
        self._rayleigh.set_coord(self._arrays['lats'], self._arrays['lons'],
            self.time)
        self._arrays['g'] = self._rayleigh.correct(self._arrays['g'],
            self.__wavelengths__[2], redband=self._arrays['r'])
        self.logger.info('Combine nir channel into green channel...')
        self._arrays['g'] = self._arrays['g'] * (1 - self.nir_fraction) + \
            self._arrays['g4'] * self.nir_fraction
        del self._arrays['g4']

    def rayleigh_corr(self):
        self.logger.info('Process rayleigh correction on red channel...')
        self._arrays['r'] = self._rayleigh.correct(self._arrays['r'],
            self.__wavelengths__[3])
        self.logger.info('Process rayleigh correction on blue channel...')
        self._arrays['b'] = self._rayleigh.correct(self._arrays['b'],
            self.__wavelengths__[1], redband=self._arrays['r'])

    def concat_bands(self):
        import dask.array as da
        self.logger.info('Concatenate RGB bands...')
        self._arrays['rgb'] = da.dstack((self._arrays['r'], self._arrays['g'],
            self._arrays['b']))
        self._arrays['rgb'] = da.clip(self._arrays['rgb'], 0, 1)
        del self._arrays['r'], self._arrays['g'], self._arrays['b']

    def establish_proj(self):
        if self.projection is not None:
            raise NotImplementedError
        self.logger.info('Establish projection of exported image...')
        if self.imsize is None:
            self.imsize = 1200
        self.imsize = int(self.imsize)
        self.imheight = int(self.imsize / (self.georange[3] - \
            self.georange[2]) * (self.georange[1] - self.georange[0]))
        xy, _ = KDResampler.make_target_coords(self.georange,
            self.imsize, self.imheight)
        return xy

    def resample(self, xy):
        self.logger.info('Resample using KDTree...')
        self.resampler = KDResampler()
        self.resampler.build_tree(self._arrays['lons'], self._arrays['lats'])
        self._arrays['rgb'] = self.resampler.resample(self._arrays['rgb'],
            xy[0], xy[1])

    def enhance_image(self):
        self.logger.info(f'Process gamma correction: {self.enh_gamma}')
        self._arrays['rgb'] = self._arrays['rgb'] ** self.enh_gamma
        self._arrays['rgb'] = np.array(self._arrays['rgb'] * 255).astype('uint8')
        self._arrays['rgb'] = np.clip(self._arrays['rgb'], 0, 255)
        self.image = Image.fromarray(self._arrays['rgb'], 'RGB')
        del self._arrays['rgb']
        self.logger.info(f'Process color enhancement: {self.enh_color}')
        converter = ImageEnhance.Color(self.image)
        self.image = converter.enhance(self.enh_color)
        self.logger.info(f'Process contrast enhancement: {self.enh_contrast}')
        converter = ImageEnhance.Contrast(self.image)
        self.image = converter.enhance(self.enh_contrast)

    def export_image(self):
        if self.export_filename is None:
            self.export_filename = 'export.png'
        self.image.save(self.export_filename)

    def get_hf(self, band):
        filenames = [os.path.join(self.filedir, get_standard_filename(
            self.time, self.region, band, segno)) for segno in self.segnos]
        hf = MutilSegmentHimawariFormat(filenames)
        return hf


def debug():
    tc = TrueColor(filedir='data', time=datetime.datetime(2020, 1, 30, 7),
        georange=(20.2, 35, 100, 120), imsize=2000, export_filename='ph.png')
    tc.make()


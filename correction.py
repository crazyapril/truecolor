import numpy as np
import xarray as xr

from astronomy import cos_zen


def sun_zenith_correction(data, utc_time, lon, lat, limit=88., max_sza=95.):
    """Perform Sun zenith angle correction.

    The correction is based on the provided cosine of the zenith
    angle (``cos_zen``).  The correction is limited
    to ``limit`` degrees (default: 88.0 degrees).  For larger zenith
    angles, the correction is the same as at the ``limit`` if ``max_sza``
    is `None`. The default behavior is to gradually reduce the correction
    past ``limit`` degrees up to ``max_sza`` where the correction becomes
    0. Both ``data`` and ``cos_zen`` should be 2D arrays of the same shape.

    """
    cos_zenith = cos_zen(utc_time, lon, lat)
    cos_zenith = xr.DataArray(cos_zenith, dims=['y', 'x'])
    if max_sza is not None:
        max_sza_cos = np.cos(np.deg2rad(max_sza))
        cos_zenith = cos_zenith.where(cos_zenith >= max_sza_cos)
    # Convert the zenith angle limit to cosine of zenith angle
    limit_rad = np.deg2rad(limit)
    limit_cos = np.cos(limit_rad)
    max_sza_rad = np.deg2rad(max_sza) if max_sza is not None else max_sza

    # Cosine correction
    corr = 1. / cos_zenith
    if max_sza is not None:
        # gradually fall off for larger zenith angle
        grad_factor = (np.arccos(cos_zenith) - limit_rad) / (max_sza_rad - limit_rad)
        # invert the factor so maximum correction is done at `limit` and falls off later
        grad_factor = 1. - np.log(grad_factor + 1) / np.log(2)
        # make sure we don't make anything negative
        grad_factor = grad_factor.clip(0.)
    else:
        # Use constant value (the limit) for larger zenith angles
        grad_factor = 1.
    corr = corr.where(cos_zenith > limit_cos, grad_factor / limit_cos)
    # Force "night" pixels to 0 (where SZA is invalid)
    corr = corr.where(cos_zenith.notnull(), 0)

    return data * corr

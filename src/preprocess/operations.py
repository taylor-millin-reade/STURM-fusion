import numpy as np
from scipy.ndimage import uniform_filter

def remove_angle(data, profile, cfg):
    """
    Remove the angle band from the data if it exists.
    """
    if data.shape[0] == 3:
        data = data[:2]  # VV, VH
        profile.update(count=2)
    return data, profile

def crop(data, profile, cfg):
    """
    Crop the data to a square of size cfg.S1_CROP_SIZE.
    """
    size = cfg.S1_CROP_SIZE
    _, H, W = data.shape
    if H > size or W > size:
        data = data[:, :size, :size]
        profile.update(height=size, width=size)
    return data, profile

def clip_bands(data, profile, cfg):
    """
    Clip the data to the specified min and max values for each band.
    """
    mins = np.array(cfg.S1_BAND_MINS)[:, None, None]
    maxs = np.array(cfg.S1_BAND_MAXS)[:, None, None]

    return np.clip(data, mins, maxs), profile

def normalise_per_band(data, profile, cfg, eps=1e-6):
    """
    Normalize the data per band.
    """
    means = np.nanmean(data, axis=(1, 2), keepdims=True)
    stds  = np.nanstd(data, axis=(1, 2), keepdims=True)

    return (data - means) / (stds + eps), profile

def remove_nana(data, profile, cfg):
    """
    Replace NaN values with zeros.
    """
    return np.nan_to_num(data, nan=0.0), profile


def lee_filter_band(band, size=5, eps=1e-8):
    """
    Apply a Lee filter to a single 2D band.
    """
    band = band.astype(np.float32)

    # local mean and variance
    local_mean = uniform_filter(band, size=size, mode="nearest")
    local_sq_mean = uniform_filter(band ** 2, size=size, mode="nearest")
    local_var = local_sq_mean - local_mean ** 2
    local_var = np.maximum(local_var, 0.0)

    # overall noise variance
    noise_var = np.mean(local_var)

    # Lee weight
    weight = local_var / (local_var + noise_var + eps)

    # filtered result
    filtered = local_mean + weight * (band - local_mean)

    return filtered.astype(np.float32)


def lee_filter_per_band(data, profile, cfg):
    """
    Apply a Lee filter separately to each band.

    data: np.ndarray [C, H, W]
    """
    size = cfg.LEE_FILTER_SIZE
    out = np.empty_like(data, dtype=np.float32)

    for b in range(data.shape[0]):
        out[b] = lee_filter_band(data[b], size=size)

    return out, profile
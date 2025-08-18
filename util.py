from numba import njit, float64, void
import numpy as np
import math
import os
import re
import subprocess
from PIL import Image

@njit(void(float64[:, :], float64[:], float64[:], float64, float64[:]))
def numba_unstructured_gaussian_kernel(positions, center, volumes, fwhm, out):
    sigma = fwhm / (2 * math.sqrt(2 * math.log(2)))
    size = out.shape[0]
    norm = 0.0
    for i in range(size):
        r2 = 0.0
        for j in range(positions.shape[1]):
            r2 += (positions[i, j] - center[j]) ** 2
        val = math.exp(-r2 / (2 * sigma ** 2)) * volumes[i]
        out[i] = val
        norm += val
    for i in range(size):
        out[i] /= norm

@njit
def numba_fill_smooth_scratch(
    turb_kernel_indices,
    neighbors_arr,
    neighbor_counts,
    positions,
    volume,
    dataset_slice,
    fwhm,
    kernel_scratch,
    turb_scratch
):
    ncenters = len(turb_kernel_indices)
    for ci in range(ncenters):
        center_index = turb_kernel_indices[ci]
        end = neighbor_counts[ci]

        numba_unstructured_gaussian_kernel(
            positions[neighbors_arr[ci, :end]],
            positions[center_index],
            volume[neighbors_arr[ci, :end]],
            fwhm,
            kernel_scratch[:end]
        )

        for field_idx in range(dataset_slice.shape[0]):
            s = 0.0
            for ni in range(end):
                s += dataset_slice[field_idx, neighbors_arr[ci, ni]] * kernel_scratch[ni]
            turb_scratch[field_idx, ci] = s

@njit(float64(float64[:], float64[:]))
def numba_weighted_std(values, weights):
    '''
    Assumes weights are normalized to sum to 1.
    '''
    size = values.shape[0]
    mean = 0.0
    for i in range(size):
        mean += values[i] * weights[i]
    variance = 0.0
    for i in range(size):
        variance += weights[i] * (values[i] - mean) ** 2
    return math.sqrt(variance)

@njit(void(float64[:], float64[:], float64[:], float64[:], float64[:]))
def numba_get_mach(velx, vely, velz, cs, mach):
    size = velx.shape[0]
    for i in range(size):
        vel = math.sqrt(velx[i]**2 + vely[i]**2 + velz[i]**2)
        mach[i] = vel / cs[i]

@njit(void(float64[:], float64[:], float64[:]))
def numba_get_rho_by_rho0(ln_rho, gauss_weights, out):
    size = ln_rho.shape[0]
    mean_rho = 0.0
    for i in range(size):
        out[i] = math.exp(ln_rho[i])
        mean_rho += gauss_weights[i] * out[i]
    for i in range(size):
        out[i] /= mean_rho

def create_movie(path, pattern_string, outdir, outname, framerate=10):
    pattern = re.compile(pattern_string)
    files = [f for f in os.listdir(path) if pattern.match(f)]
    files.sort(key=lambda f: int(pattern.match(f).group(1)))
    ffmpeg_cmd = [
        'ffmpeg',
        '-y',  # overwrite
        '-f', 'image2pipe',
        '-vcodec', 'png',
        '-framerate', str(framerate),
        '-i', '-',  # input from stdin
        os.path.join(outdir, f'{outname}.mp4')
    ]
    process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
    # Feed images to ffmpeg
    for file in files:
        img = Image.open(os.path.join(path, file)).convert("RGB")  # convert to RGB just in case
        img.save(process.stdin, format='PNG')
    process.stdin.close()
    process.wait()
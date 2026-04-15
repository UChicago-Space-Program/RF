#!/usr/bin/env python3
"""
Standalone renderer for the Doppler × Attenuation FER surface plot.

Usage
-----
    conda activate gr
    python render_doppler_surface.py

Edit the CONFIG block below to change colours, angle, smoothing, etc.
No other part of the file needs to be touched.
"""

import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')                            # headless — no GUI window needed
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D          # noqa — registers 3-D projection
from scipy.ndimage import gaussian_filter, zoom as nd_zoom
from scipy.interpolate import griddata

# =============================================================================
# CONFIG — edit freely
# =============================================================================

CSV_FILE   = 'doppler_attn_results.csv'   # path to the sweep data
OUT_FILE   = 'doppler_attn_surface_final.png'

# --- colour map ---
# Common choices:
#   'coolwarm'   blue trench / red plateau  (current)
#   'RdYlGn_r'   green trench / red plateau
#   'plasma'     dark purple → yellow
#   'inferno_r'  dark → bright
#   'turbo'      vivid rainbow
#   'viridis_r'  dark purple → yellow-green
COLORMAP   = 'coolwarm'

# --- viewing angle ---
# elev : degrees above the horizontal plane (0 = side-on, 90 = top-down)
# azim : degrees of azimuthal rotation (0 = front, 90 = right, 45 = diagonal)
ELEV       = 45
AZIM       = 45

# --- interpolation smoothing ---
# Higher sigma = smoother surface, less faithful to raw data points
# Recommended range: 3 (sharp) to 8 (very smooth)
SIGMA      = 5

# --- surface resolution (higher = slower but crisper saved image) ---
GRID_N     = 400

# --- output DPI (180 = presentation, 300 = print-ready) ---
DPI        = 220

# --- annotations: (x=attn, y=doppler_kHz, z=FER, text, colour) ---
# Set to an empty list [] to disable all annotations.
ANNOTATIONS = [
    (12,   0.8, 0.04, 'Operating\ntrench',     '#1a237e'),
    ( 8,   3.6, 0.88, 'Doppler cliff\n(≈±2 kHz)', '#D3D3D3'),
    (20,  -3.2, 1.06, 'Link failure\nplateau', '#D3D3D3'),
    (49,   2.2, 0.55, 'Attenuation\ncliff',    '#4a148c'),
]

# =============================================================================
# Interpolation pipeline  (no need to edit below here)
# =============================================================================

def load_and_interpolate(csv_file, sigma, grid_n):
    with open(csv_file) as f:
        rows = list(csv.DictReader(f))

    attns_arr   = np.array(sorted(set(int(r['attn_db'])        for r in rows)), float)
    offsets_arr = np.array(sorted(set(int(r['freq_offset_hz']) for r in rows)), float) / 1000

    fer_grid = np.zeros((len(offsets_arr), len(attns_arr)))
    for r in rows:
        i = list((offsets_arr * 1000).astype(int)).index(int(r['freq_offset_hz']))
        j = list(attns_arr.astype(int)).index(int(r['attn_db']))
        fer_grid[i, j] = float(r['fer'])

    # Quintic zoom × 40 in index space
    Zz      = np.clip(nd_zoom(fer_grid, 40, order=5), 0, 1)
    off_idx = np.linspace(0, len(offsets_arr) - 1, Zz.shape[0])
    atn_idx = np.linspace(0, len(attns_arr)   - 1, Zz.shape[1])
    off_phys = np.interp(off_idx, np.arange(len(offsets_arr)), offsets_arr)
    atn_phys = np.interp(atn_idx, np.arange(len(attns_arr)),   attns_arr)
    Xz, Yz  = np.meshgrid(atn_phys, off_phys)
    pts_z   = np.column_stack([Xz.ravel(), Yz.ravel()])

    # Resample onto uniform fine grid + Gaussian smoothing
    attn_fine   = np.linspace(attns_arr.min(),   attns_arr.max(),   grid_n)
    offset_fine = np.linspace(offsets_arr.min(), offsets_arr.max(), grid_n)
    Xf, Yf = np.meshgrid(attn_fine, offset_fine)
    Zraw   = griddata(pts_z, Zz.ravel(), (Xf, Yf), method='linear')
    Zf     = np.clip(gaussian_filter(np.clip(Zraw, 0, 1), sigma=sigma), 0, 1)

    return Xf, Yf, Zf


def render(Xf, Yf, Zf, colormap, elev, azim, annotations, out_file, dpi):
    plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 11})

    fig = plt.figure(figsize=(14, 9))
    fig.patch.set_alpha(0)                        # transparent figure background
    ax  = fig.add_subplot(111, projection='3d')
    ax.patch.set_alpha(0)                         # transparent axes background
    ax.set_facecolor((0, 0, 0, 0))               # transparent pane fill

    surf = ax.plot_surface(Xf, Yf, Zf, cmap=colormap, alpha=0.95,
                           linewidth=0, antialiased=True)

    ax.set_xlabel('TX Attenuation (dB)', labelpad=12)
    ax.set_ylabel('Doppler Offset (kHz)', labelpad=12)
    ax.set_zlabel('FER', labelpad=8)
    ax.set_zlim(0, 1)
    ax.set_xticks([0, 10, 20, 30, 35, 42, 46, 50, 52])
    ax.set_yticks([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
    ax.set_title(
        'Frame Error Rate — TX Attenuation × Doppler Offset\n'
        'PlutoSDR → RTL-SDR, GFSK 9600 baud, 433 MHz',
        fontsize=12, fontweight='bold', pad=20)

    cbar = fig.colorbar(surf, ax=ax, shrink=0.4, aspect=12, pad=0.08)
    cbar.set_label('FER', labelpad=8)
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(['0%', '25%', '50%', '75%', '100%'])

    for (x, y, z, text, colour) in annotations:
        ax.text(x, y, z, text, color=colour,
                fontsize=9, fontweight='bold', ha='center')

    ax.view_init(elev=elev, azim=azim)
    plt.tight_layout()
    plt.savefig(out_file, dpi=dpi, bbox_inches='tight', transparent=True)
    print(f'Saved {out_file}')
    plt.close()


if __name__ == '__main__':
    Xf, Yf, Zf = load_and_interpolate(CSV_FILE, SIGMA, GRID_N)
    render(Xf, Yf, Zf, COLORMAP, ELEV, AZIM, ANNOTATIONS, OUT_FILE, DPI)

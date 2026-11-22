#!/usr/bin/env python
"""Build a Gazebo heightmap (8-bit PNG), a hill-shaded texture and an elevation lookup (npz) from open elevation data
(AWS 'terrarium' tiles: SRTM/NASADEM-derived, public). The plan coordinates (0..16 km) are placed on a real 16 km x 16 km area."""
import io, json, math, os, sys, urllib.request
import numpy as np
from PIL import Image

lat0, lon0 = float(sys.argv[1]), float(sys.argv[2]); out = sys.argv[3]; SIZE = 24000.0; Z = 11
os.makedirs(out, exist_ok=True)
def tile_xy(lat, lon, z):
    n = 2 ** z; x = (lon + 180) / 360 * n; y = (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n; return x, y
def fetch(z, x, y):
    cache = os.path.join(out, f"t_{z}_{x}_{y}.png")
    if not os.path.exists(cache):
        urllib.request.urlretrieve(f"https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png", cache)
    a = np.asarray(Image.open(cache).convert("RGB")).astype(np.float64); return a[..., 0] * 256 + a[..., 1] + a[..., 2] / 256 - 32768
m_per_px = 40075016.686 * math.cos(math.radians(lat0)) / (2 ** Z * 256); span_px = SIZE / m_per_px
cx, cy = tile_xy(lat0, lon0, Z); x0, x1 = cx - span_px / 512, cx + span_px / 512; y0, y1 = cy - span_px / 512, cy + span_px / 512
tx0, tx1, ty0, ty1 = int(x0), int(x1), int(y0), int(y1)
mos = np.vstack([np.hstack([fetch(Z, tx, ty) for tx in range(tx0, tx1 + 1)]) for ty in range(ty0, ty1 + 1)])
px0, py0 = (x0 - tx0) * 256, (y0 - ty0) * 256; N = 1025
yy, xx = np.meshgrid(np.linspace(py0, py0 + span_px, N), np.linspace(px0, px0 + span_px, N), indexing="ij")
from scipy.ndimage import map_coordinates
dem = map_coordinates(mos, [yy, xx], order=1, mode="nearest")        # row 0 = north (PNG); dem.npz is indexed by y from the south edge
# Gazebo reads image row 0 as +y (north), so keep row 0 = north in the PNG
lo, hi = float(dem.min()), float(dem.max()); relief = hi - lo
png = np.round((dem - lo) / max(relief, 1e-6) * 255).astype(np.uint8)
Image.fromarray(png).save(os.path.join(out, "heightmap.png"))
q = lo + png.astype(np.float64) / 255 * relief                         # what the simulator will actually use
np.savez(os.path.join(out, "dem.npz"), z=(q - lo)[::-1], size=SIZE, relief=relief, base=lo)
gy, gx = np.gradient(q, SIZE / (N - 1)); sh = np.clip(0.5 + 0.5 * (gx * math.cos(math.radians(315)) + gy * math.sin(math.radians(315))) * 6, 0, 1)
sand = np.array([218, 196, 150]) / 255.0; tex = np.clip(sand[None, None, :] * (0.62 + 0.45 * sh[..., None]) * (0.92 + 0.12 * (q - lo)[..., None] / max(relief, 1)), 0, 1)
Image.fromarray((tex * 255).astype(np.uint8)).resize((2048, 2048), Image.BILINEAR).save(os.path.join(out, "diffuse.png"))
Image.fromarray(np.full((8, 8, 3), (128, 128, 255), np.uint8)).save(os.path.join(out, "normal.png"))
json.dump(dict(lat=lat0, lon=lon0, size=SIZE, relief=relief, base=lo, N=N, cell=SIZE / (N - 1)), open(os.path.join(out, "info.json"), "w"))
print(f"lat {lat0} lon {lon0}: elevation {lo:.0f}..{hi:.0f} m, relief {relief:.0f} m, cell {SIZE/(N-1):.1f} m, max slope {np.degrees(np.arctan(np.hypot(gx,gy).max())):.1f} deg, mean slope {np.degrees(np.arctan(np.hypot(gx,gy))).mean():.1f} deg")

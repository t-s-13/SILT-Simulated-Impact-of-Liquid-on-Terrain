"""
show_real_mountain.py - download REAL elevation data and show it. Nothing is generated.

  python show_real_mountain.py                  Mount Everest
  python show_real_mountain.py --peak k2          (see --list-peaks)
  python show_real_mountain.py --lat 28.6 --lon 83.82 --extent-km 40

What it does
  1. Downloads elevation tiles from AWS Terrain Tiles (free, no API key):
         https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png
     elevation_m = R*256 + G + B/256 - 32768      (tiles cached in ~/.cache/erosion_dem)
  2. Prints the highest point it found, so you can check it against the real mountain
     (Everest summit is 8,849 m).
  3. Shows a 3D view and a map, saves them as a PNG, and saves the elevations as
         <peak>_elevation_m.npy
     which erosion_himalaya.py can load with  --dem-file <peak>_elevation_m.npy
Needs only numpy + matplotlib (Pillow comes with matplotlib).
"""
import argparse
import math
import os
import urllib.request

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LightSource, LinearSegmentedColormap, Normalize

PEAKS = {
    "everest":       ("Mount Everest",   27.9881, 86.9250, 8849),
    "lhotse":        ("Lhotse",          27.9617, 86.9331, 8516),
    "makalu":        ("Makalu",          27.8897, 87.0886, 8485),
    "cho_oyu":       ("Cho Oyu",         28.0942, 86.6609, 8188),
    "ama_dablam":    ("Ama Dablam",      27.8617, 86.8614, 6812),
    "kangchenjunga": ("Kangchenjunga",   27.7025, 88.1475, 8586),
    "annapurna":     ("Annapurna I",     28.5961, 83.8203, 8091),
    "dhaulagiri":    ("Dhaulagiri I",    28.6967, 83.4875, 8167),
    "manaslu":       ("Manaslu",         28.5497, 84.5597, 8163),
    "nanda_devi":    ("Nanda Devi",      30.3753, 79.9708, 7816),
    "k2":            ("K2",              35.8825, 76.5133, 8611),
    "nanga_parbat":  ("Nanga Parbat",    35.2375, 74.5892, 8126),
}
TILE_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "erosion_dem")
M_PER_DEG = 111320.0

CMAP = LinearSegmentedColormap.from_list("mountain", [
    (0.00, "#3b2a1a"), (0.30, "#6b4a2b"), (0.55, "#8a6a4a"),
    (0.75, "#9b9690"), (0.90, "#d9d9d9"), (1.00, "#ffffff")])


def tile_pixels(lat, lon, z):
    px = (np.asarray(lon, float) + 180.0) / 360.0 * 256 * 2**z
    py = (1 - np.arcsinh(np.tan(np.radians(np.asarray(lat, float)))) / np.pi) / 2 * 256 * 2**z
    return px, py


def get_tile(z, x, y):
    from PIL import Image
    path = os.path.join(CACHE_DIR, f"{z}_{x}_{y}.png")
    if not os.path.exists(path):
        os.makedirs(CACHE_DIR, exist_ok=True)
        url = TILE_URL.format(z=z, x=x, y=y)
        print("  downloading", url)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "erosion-sim/1.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                data = r.read()
        except Exception as e:
            raise SystemExit(f"\nDOWNLOAD FAILED: {url}\n  {e}\n"
                             "This computer cannot reach the elevation server. Check the internet "
                             "connection / firewall / proxy.")
        with open(path, "wb") as f:
            f.write(data)
    else:
        print("  cached     ", f"{z}/{x}/{y}")
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64)
    return rgb[..., 0] * 256 + rgb[..., 1] + rgb[..., 2] / 256 - 32768


def sample_bilinear(grid, u, v):
    H, W = grid.shape
    i0 = np.clip(np.floor(u).astype(int), 0, W - 2)
    j0 = np.clip(np.floor(v).astype(int), 0, H - 2)
    fu, fv = np.clip(u - i0, 0, 1)[None, :], np.clip(v - j0, 0, 1)[:, None]
    g = lambda j, i: grid[np.ix_(j, i)]
    return (g(j0, i0) * (1 - fu) * (1 - fv) + g(j0, i0 + 1) * fu * (1 - fv)
            + g(j0 + 1, i0) * (1 - fu) * fv + g(j0 + 1, i0 + 1) * fu * fv)


def real_elevation(lat, lon, extent_km, n):
    half = extent_km * 1000 / 2
    dlat = half / M_PER_DEG
    dlon = half / (M_PER_DEG * math.cos(math.radians(lat)))
    lats = np.linspace(lat + dlat, lat - dlat, n)          # row 0 = north
    lons = np.linspace(lon - dlon, lon + dlon, n)          # column 0 = west
    cell_m = extent_km * 1000 / n
    z = int(min(max(math.ceil(math.log2(156543.03392 * math.cos(math.radians(lat)) / cell_m)), 6), 14))
    px, _ = tile_pixels(lat, lons, z)
    _, py = tile_pixels(lats, lon, z)
    tx0, tx1, ty0, ty1 = int(px.min() // 256), int(px.max() // 256), int(py.min() // 256), int(py.max() // 256)
    print(f"Fetching real elevation tiles (zoom {z}, {(tx1 - tx0 + 1) * (ty1 - ty0 + 1)} tiles)...")
    mosaic = np.zeros(((ty1 - ty0 + 1) * 256, (tx1 - tx0 + 1) * 256))
    for ty in range(ty0, ty1 + 1):
        for tx in range(tx0, tx1 + 1):
            mosaic[(ty - ty0) * 256:(ty - ty0 + 1) * 256, (tx - tx0) * 256:(tx - tx0 + 1) * 256] = get_tile(z, tx, ty)
    elev = sample_bilinear(mosaic, px - tx0 * 256, py - ty0 * 256)
    return elev, lats, lons, cell_m


def main():
    ap = argparse.ArgumentParser(description="Download and show REAL elevation data")
    ap.add_argument("--peak", default="everest", choices=sorted(PEAKS))
    ap.add_argument("--list-peaks", action="store_true")
    ap.add_argument("--lat", type=float); ap.add_argument("--lon", type=float)
    ap.add_argument("--extent-km", type=float, default=30.0)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--vert-exag", type=float, default=1.5, help="vertical exaggeration of the 3D view")
    a = ap.parse_args()
    if a.list_peaks:
        for k, (name, lat, lon, h) in sorted(PEAKS.items()):
            print(f"  {k:14s} {name:16s} {lat:8.4f}N {lon:8.4f}E  {h} m")
        return

    if a.lat is not None and a.lon is not None:
        label, lat, lon, listed = f"{a.lat:.3f}N {a.lon:.3f}E", a.lat, a.lon, None
    else:
        label, lat, lon, listed = PEAKS[a.peak]

    elev, lats, lons, cell_m = real_elevation(lat, lon, a.extent_km, a.size)

    r, c = np.unravel_index(elev.argmax(), elev.shape)
    print(f"\n{label}: {a.extent_km:g} km square, {cell_m:.0f} m per cell")
    print(f"Lowest point : {elev.min():8.0f} m")
    print(f"Highest point: {elev.max():8.0f} m  at {lats[r]:.4f}N {lons[c]:.4f}E")
    if listed:
        print(f"Published summit height: {listed} m  (this grid averages ~{cell_m:.0f} m cells, "
              "so it reads a little lower)")
    print("Source: AWS Terrain Tiles (SRTM and other open DEMs). Nothing generated.\n")

    stem = a.peak if a.lat is None else f"lat{a.lat:.2f}_lon{a.lon:.2f}"
    np.save(f"{stem}_elevation_m.npy", elev)
    print(f"Saved {stem}_elevation_m.npy  ->  python erosion_himalaya.py --dem-file {stem}_elevation_m.npy "
          f"--extent-km {a.extent_km:g}")

    # ---- draw it
    zmin, zmax = float(elev.min()), float(elev.max())
    norm = Normalize(zmin, zmax)
    ls = LightSource(azdeg=315, altdeg=45)
    rgb = ls.shade(elev, cmap=CMAP, norm=norm, vert_exag=a.vert_exag, dx=cell_m, dy=cell_m, blend_mode="soft")

    fig = plt.figure(figsize=(15, 7.5), facecolor="#111111")
    fig.text(0.5, 0.965, f"{label} - REAL elevation data  |  {zmin:,.0f} to {zmax:,.0f} m",
             color="w", ha="center", fontsize=13)

    ax3 = fig.add_axes([0.0, 0.02, 0.58, 0.9], projection="3d", facecolor="#111111")
    ekm = np.linspace(0, a.extent_km, a.size)
    X, Y = np.meshgrid(ekm, ekm[::-1])                      # x = east, y = north (row 0 = north)
    s = 2
    ax3.plot_surface(X[::s, ::s], Y[::s, ::s], elev[::s, ::s] / 1000, facecolors=rgb[::s, ::s],
                     rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False)
    ax3.set_box_aspect((1, 1, (zmax - zmin) / 1000 / a.extent_km * a.vert_exag), zoom=1.25)
    ax3.view_init(elev=35, azim=235)
    ax3.set_axis_off()

    ax2 = fig.add_axes([0.60, 0.12, 0.37, 0.76])
    ax2.imshow(rgb, extent=[0, a.extent_km, 0, a.extent_km], origin="upper")
    cs = ax2.contour(ekm, ekm[::-1], elev, levels=np.arange(math.ceil(zmin / 250) * 250, zmax, 250),
                     colors="white", linewidths=0.3, alpha=0.5)
    ax2.set_xlabel("east (km)", color="w"); ax2.set_ylabel("north (km)", color="w")
    ax2.tick_params(colors="w")
    for sp in ax2.spines.values():
        sp.set_color("#555")
    ax2.set_title("map (contours every 250 m, north up)", color="w", fontsize=10)
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=CMAP), ax=ax2, fraction=0.04, pad=0.03)
    cb.set_label("elevation (m)", color="w"); cb.ax.yaxis.set_tick_params(color="w")
    plt.setp(cb.ax.get_yticklabels(), color="w")

    out = f"real_{stem}.png"
    fig.savefig(out, dpi=110, facecolor="#111111")
    print("Saved", out)
    plt.show()


if __name__ == "__main__":
    main()

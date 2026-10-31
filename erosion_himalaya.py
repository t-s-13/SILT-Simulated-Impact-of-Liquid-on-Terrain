"""
Real-time rainfall + evaporation erosion on a REAL mountain (NumPy + Matplotlib only).
Same erosion physics and same window as erosion_realtime_fast.py, but the starting
terrain is real elevation data (AWS Terrain Tiles: SRTM and other open DEMs).
LANDSLIDES are now drawn in RED on the map and the 3D terrain.

PHYSICS (particle / raindrop method)
  Rain      thousands of raindrops land at random places every step.
  Flow      each drop rolls downhill, steered by the slope plus some inertia.
  Erosion   fast, deep drops on steep ground pick up sediment.
  Deposit   slow or uphill drops drop it (fans, plains, filled basins).
  Evaporate each drop loses water as it moves; a dried-up drop leaves its sediment.
  Landslide slopes steeper than the talus angle collapse downhill (shown in red).
  Edge      real mountains have no sea: drops that reach the map edge leave the map,
            and the edge cells stay fixed at their real height (open boundary).

REAL DATA
  * Elevation tiles are downloaded once from
        https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png
    (free, no API key) and cached in ~/.cache/erosion_dem, so later runs are instant.
    elevation_m = R*256 + G + B/256 - 32768.
  * A square window (default 30 km) around the chosen peak is resampled to the grid.
  * Heights are converted to model units with 1 cell = 1 unit wide, so slopes are the
    real slopes (times --vert-exag). Model height = (elevation - lowest point) / cell size.

RUN
  python erosion_himalaya.py                           opens a drop-down to pick the mountain
  python erosion_himalaya.py --peak everest            skip the menu and go straight to a mountain
  python erosion_himalaya.py --peak k2                 another peak (see --list-peaks)
  python erosion_himalaya.py --lat 28.6 --lon 83.82    any place on Earth
  python erosion_himalaya.py --extent-km 50            wider area (coarser cells)
  python erosion_himalaya.py --run-years 1000 10000 50000   forecast: terrain after those many years
  python erosion_himalaya.py --rate-mm-yr 0.5          slower erosion (default 1 mm/yr, see below)
  python erosion_himalaya.py --video out.mp4           render a video
  python erosion_himalaya.py --help                    all options

YEARS SLIDER
  The slider under the window is a time machine. While the simulation runs it follows along
  (press LIVE to go back to following it). Drag it to any year up to --max-years: the window
  shows the terrain at that year, and if that year has not been simulated yet it computes
  ahead at high speed until it gets there.

REALISM SETTINGS
  Water that evaporates slowly and may travel far (lifetime 150, evaporation 0.008) makes
  long coherent rivers that cut deep valleys, instead of short streams that dump their sand
  halfway down the slope. Rain only changes how FAST things happen, not what they look like.
  Try --lifetime 300 --evaporation 0.004 for even longer rivers (slower to compute).

MODEL YEARS (rough!)
  The particle model has no clock of its own, so time is defined through the amount of
  rock removed: model years = (average lowering of the mountain, in metres) / (average
  denudation rate). The default 1 mm/yr is typical of the Everest region (catchment-averaged
  rates across the Himalaya span roughly 0.15-3 mm/yr). Treat the years as an
  order-of-magnitude estimate. Only rain erosion and landslides act: no glaciers, no
  rock uplift, no climate change.

Elevation data: AWS Terrain Tiles (https://registry.opendata.aws/terrain-tiles/);
see that page for the required attribution of the underlying sources.
"""
import argparse
import bisect
import dataclasses
import math
import os
import queue
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import LightSource, LinearSegmentedColormap, Normalize
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button, Slider

HEIGHT = 80.0            # max terrain height in model units (1 cell = 1 unit wide)
                         # (replaced by the real mountain's height when real data is loaded)


# ============================ SETTINGS ====================================
@dataclasses.dataclass
class Params:
    """Settings the sim thread reads every step (the sliders change them live)."""
    drops: int = 1500            # raindrops released per step
    lifetime: int = 150          # max moves per drop (how far water can travel)
    evaporation: float = 0.008   # fraction of a drop's water lost per move
    inertia: float = 0.30        # 0 = follows slope exactly, 1 = ignores slope
    capacity: float = 4.0        # sediment a fast, deep drop can carry
    min_slope: float = 0.02      # capacity floor so drops still carry on flat ground
    erode_rate: float = 0.0002   # fraction of spare capacity picked up per move
    deposit_rate: float = 0.20   # fraction of excess sediment dropped per move
    gravity: float = 4.0         # how strongly slopes accelerate drops
    talus: float = 1.3           # max stable height difference between neighbours
    thermal_rate: float = 0.10   # fraction of excess moved per step
    wet_decay: float = 0.894     # rivers fade by this factor each step


WET_SCALE = 260.0        # flow needed for a river to show fully blue (at REF_DROPS rain)
REF_DROPS = 3000.0       # river display is normalised to this rain amount, so the rain slider
                         # changes the speed of erosion but not how visible the rivers are
SEA_LEVEL = 0.5          # cells below this height are drawn as sea (off for real mountains)
MIN_DROPS_PER_WORKER = 20000  # don't split rain into chunks smaller than this
SLIDE_DECAY = 0.9        # red landslide marks fade by this factor each step
SLIDE_GAIN = 4.0         # how easily landslide red shows up (raise for more red)
# ==========================================================================


# ------------------------- real mountains: elevation data -------------------
PEAKS = {                       # name: (label, latitude, longitude)
    "everest":       ("Mount Everest",   27.9881, 86.9250),
    "lhotse":        ("Lhotse",          27.9617, 86.9331),
    "makalu":        ("Makalu",          27.8897, 87.0886),
    "cho_oyu":       ("Cho Oyu",         28.0942, 86.6609),
    "ama_dablam":    ("Ama Dablam",      27.8617, 86.8614),
    "kangchenjunga": ("Kangchenjunga",   27.7025, 88.1475),
    "annapurna":     ("Annapurna I",     28.5961, 83.8203),
    "dhaulagiri":    ("Dhaulagiri I",    28.6967, 83.4875),
    "manaslu":       ("Manaslu",         28.5497, 84.5597),
    "nanda_devi":    ("Nanda Devi",      30.3753, 79.9708),
    "k2":            ("K2",              35.8825, 76.5133),
    "nanga_parbat":  ("Nanga Parbat",    35.2375, 74.5892),
}

PEAK_HEIGHT_M = {"everest": 8849, "k2": 8611, "kangchenjunga": 8586, "lhotse": 8516, "makalu": 8485,
                 "cho_oyu": 8188, "dhaulagiri": 8167, "manaslu": 8163, "nanga_parbat": 8126,
                 "annapurna": 8091, "nanda_devi": 7816, "ama_dablam": 6812}

TILE_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "erosion_dem")
M_PER_DEG_LAT = 111320.0
DEM_META = {}                     # elev_min (m), cell_m, vexag of the loaded mountain


def configure_for_dem(h):
    """Re-scale the colour range, contours, 3D box... to the real mountain's height."""
    global HEIGHT, NORM, SEA_LEVEL
    HEIGHT = float(np.ceil(h.max() * 1.05))
    NORM = Normalize(0, HEIGHT)
    SEA_LEVEL = -1.0                                    # no sea: don't paint valleys blue


def _merc_px(lon, z):
    return (np.asarray(lon, float) + 180.0) / 360.0 * 256 * 2**z


def _merc_py(lat, z):
    phi = np.radians(np.asarray(lat, float))
    return (1 - np.arcsinh(np.tan(phi)) / np.pi) / 2 * 256 * 2**z


def plan_tiles(lat, lon, extent_km, n):
    """Which tiles cover the window, and where each output cell falls in the mosaic."""
    half_m = extent_km * 1000 / 2
    dlat = half_m / M_PER_DEG_LAT
    dlon = half_m / (M_PER_DEG_LAT * math.cos(math.radians(lat)))
    lats = np.linspace(lat + dlat, lat - dlat, n)        # row 0 = north
    lons = np.linspace(lon - dlon, lon + dlon, n)        # column 0 = west
    cell_m = extent_km * 1000 / n
    z = math.ceil(math.log2(156543.03392 * math.cos(math.radians(lat)) / cell_m))
    z = int(min(max(z, 6), 14))                          # tile pixels at least as fine as a cell
    px, py = _merc_px(lons, z), _merc_py(lats, z)
    return dict(z=z, px=px, py=py, cell_m=cell_m,
                tx=(int(px.min() // 256), int(px.max() // 256)),
                ty=(int(py.min() // 256), int(py.max() // 256)))


def get_tile(z, x, y, cache_dir=CACHE_DIR):
    """One 256x256 tile of elevations in metres (downloaded once, then cached)."""
    from PIL import Image
    path = os.path.join(cache_dir, f"{z}_{x}_{y}.png")
    if not os.path.exists(path):
        os.makedirs(cache_dir, exist_ok=True)
        url = TILE_URL.format(z=z, x=x, y=y)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "erosion-sim/1.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                data = r.read()
        except Exception as e:
            raise RuntimeError(
                f"Could not download elevation tile {url}\n  ({e})\n"
                "Check your internet connection, or pass your own elevation array "
                "with --dem-file my_elevation.npy.") from e
        with open(path, "wb") as f:
            f.write(data)
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64)
    return rgb[..., 0] * 256 + rgb[..., 1] + rgb[..., 2] / 256 - 32768


def sample_bilinear(grid, u, v):
    """Sample a 2D array at fractional columns u and rows v (separable: u, v are 1D)."""
    H, W = grid.shape
    i0 = np.clip(np.floor(u).astype(int), 0, W - 2)
    j0 = np.clip(np.floor(v).astype(int), 0, H - 2)
    fu = np.clip(u - i0, 0, 1)[None, :]
    fv = np.clip(v - j0, 0, 1)[:, None]
    a, b = grid[np.ix_(j0, i0)], grid[np.ix_(j0, i0 + 1)]
    c, d = grid[np.ix_(j0 + 1, i0)], grid[np.ix_(j0 + 1, i0 + 1)]
    return a * (1 - fu) * (1 - fv) + b * fu * (1 - fv) + c * (1 - fu) * fv + d * fu * fv


def download_elevation(lat, lon, extent_km, n):
    """Real elevation in metres for a square window centred on (lat, lon), north up."""
    plan = plan_tiles(lat, lon, extent_km, n)
    (tx0, tx1), (ty0, ty1) = plan["tx"], plan["ty"]
    count = (tx1 - tx0 + 1) * (ty1 - ty0 + 1)
    if count > 36:
        raise RuntimeError(f"That window needs {count} tiles; reduce --extent-km.")
    print(f"Elevation: zoom {plan['z']}, {count} tile(s), cache {CACHE_DIR}")
    mosaic = np.zeros(((ty1 - ty0 + 1) * 256, (tx1 - tx0 + 1) * 256))
    for ty in range(ty0, ty1 + 1):
        for tx in range(tx0, tx1 + 1):
            mosaic[(ty - ty0) * 256:(ty - ty0 + 1) * 256,
                   (tx - tx0) * 256:(tx - tx0 + 1) * 256] = get_tile(plan["z"], tx, ty)
    return sample_bilinear(mosaic, plan["px"] - tx0 * 256, plan["py"] - ty0 * 256)


def load_mountain(a):
    """Returns (model-unit heightmap, label, info text)."""
    n = a.size
    if a.dem_file:
        elev = np.load(a.dem_file).astype(float)
        if elev.shape != (n, n):                           # resample to the grid
            elev = sample_bilinear(elev, np.linspace(0, elev.shape[1] - 1, n),
                                   np.linspace(0, elev.shape[0] - 1, n))
        label = os.path.basename(a.dem_file)
    else:
        if a.lat is not None and a.lon is not None:
            label, lat, lon = f"{a.lat:.3f}N {a.lon:.3f}E", a.lat, a.lon
        else:
            label, lat, lon = PEAKS[a.peak]
        elev = download_elevation(lat, lon, a.extent_km, n)

    cell_m = a.extent_km * 1000 / n
    relief = float(elev.max() - elev.min())
    h = (elev - elev.min()) / cell_m * a.vert_exag         # model units: 1 cell = 1 unit wide
    h += np.random.default_rng(a.seed).normal(0, 1e-4 * h.max(), h.shape)    # ~0.5 m, only breaks dead-flat ties
    h = np.clip(h, 0, None)
    DEM_META.update(elev_min=float(elev.min()), cell_m=cell_m, vexag=a.vert_exag,
                    rate_mm_yr=a.rate_mm_yr)
    info = (f"{a.extent_km:g} km square, {cell_m:.0f} m per cell, "
            f"{elev.min():.0f}-{elev.max():.0f} m (relief {relief:.0f} m)")
    print(f"{label}: {info}")
    print("Elevation data: AWS Terrain Tiles (SRTM and other open DEMs).")
    return h, label, info


def years_elapsed(h0, h):
    """Model years: metres of average lowering divided by the assumed denudation rate."""
    m = DEM_META
    lowering_m = (h0.mean() - h.mean()) * m["cell_m"] / m["vexag"]
    return max(lowering_m, 0.0) / (m["rate_mm_yr"] / 1000.0)


def highest_point_m(h):
    m = DEM_META
    return h.max() / m["vexag"] * m["cell_m"] + m["elev_min"]


def report_slopes(h, talus):
    gy, gx = np.gradient(h)
    steep = float((np.hypot(gx, gy) > talus).mean()) * 100
    print(f"Initial slopes: {steep:.1f}% of cells are steeper than talus={talus:g}.")
    if steep > 5:
        print("  Those slopes will slump quickly. Raise --talus (e.g. 1.8) to keep more "
              "of the real cliffs, or lower --vert-exag.")


# --------------------------- one chunk of raindrops --------------------------
BRUSH = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], float) / 16.0
BRUSH_W = BRUSH.ravel()[:, None]          # (9, 1) erosion weights
_OFF3 = np.arange(-1, 2)


def rain_chunk(h_in, count, rng, P):
    """Simulate `count` raindrops on a private copy of the terrain.
    Returns (change in height, visit counts, sediment eroded, sediment lost to sea).
    Safe to run in several threads at once: it never writes to h_in."""
    n = h_in.shape[0]
    nn = n * n
    h = h_in.copy()
    hf = h.reshape(-1)                                   # flat view of h (edits show in h)
    corner_off = np.array([0, 1, n, n + 1])[:, None]
    brush_off = (_OFF3[:, None] * n + _OFF3[None, :]).ravel()[:, None]

    x = rng.uniform(3, n - 4, count)
    y = rng.uniform(3, n - 4, count)
    dx = np.zeros(count); dy = np.zeros(count)
    speed = np.ones(count); water = np.ones(count); sed = np.zeros(count)
    vis_idx, vis_w = [], []
    eroded = lost = 0.0

    for _ in range(P.lifetime):
        ix = np.clip(x.astype(np.intp), 0, n - 2); iy = np.clip(y.astype(np.intp), 0, n - 2)
        fx, fy = x - ix, y - iy
        base = iy * n + ix
        h00 = hf.take(base); h10 = hf.take(base + 1)
        h01 = hf.take(base + n); h11 = hf.take(base + n + 1)
        gx = (h10 - h00) * (1 - fy) + (h11 - h01) * fy       # slope in x
        gy = (h01 - h00) * (1 - fx) + (h11 - h10) * fx       # slope in y
        hcur = (h00 * (1 - fx) * (1 - fy) + h10 * fx * (1 - fy)
                + h01 * (1 - fx) * fy + h11 * fx * fy)

        # steer: blend the old direction with the downhill direction
        dx = dx * P.inertia - gx * (1 - P.inertia)
        dy = dy * P.inertia - gy * (1 - P.inertia)
        norm = np.hypot(dx, dy)
        flat = norm < 1e-6
        norm[flat] = 1.0
        dx /= norm; dy /= norm
        xn, yn = x + dx, y + dy

        # drops leaving the safe area reach the sea (or leave the map) and are removed;
        # flat drops stop. Dead drops are dropped from every array.
        sea = (xn < 3) | (xn > n - 4) | (yn < 3) | (yn > n - 4)
        lost += sed[sea].sum()
        keep = ~(sea | flat)
        if not keep.all():
            x, y, xn, yn, dx, dy, speed, water, sed, fx, fy, base, hcur = [
                a[keep] for a in (x, y, xn, yn, dx, dy, speed, water, sed, fx, fy, base, hcur)]
            if x.size == 0:
                break
        xn = np.clip(xn, 0, n - 1.001); yn = np.clip(yn, 0, n - 1.001)

        jx = np.clip(xn.astype(np.intp), 0, n - 2); jy = np.clip(yn.astype(np.intp), 0, n - 2)
        gx2, gy2 = xn - jx, yn - jy
        jb = jy * n + jx
        hnew = (hf.take(jb) * (1 - gx2) * (1 - gy2) + hf.take(jb + 1) * gx2 * (1 - gy2)
                + hf.take(jb + n) * (1 - gx2) * gy2 + hf.take(jb + n + 1) * gx2 * gy2)
        dh = hnew - hcur

        # carrying capacity: fast + deep + steep = lots of sediment
        cap = np.maximum(-dh, P.min_slope) * speed * water * P.capacity
        up = dh > 0
        depositing = (sed > cap) | up
        dep = np.where(up, np.minimum(dh, sed), (sed - cap) * P.deposit_rate)
        ero = np.minimum((cap - sed) * P.erode_rate, np.maximum(-dh, 0))
        dep = np.where(depositing, dep, 0.0)
        ero = np.where(depositing, 0.0, ero)
        sed = sed + ero - dep
        eroded += ero.sum()

        # ONE bincount: deposit (bilinear onto 4 corners, +) and erode (3x3 brush, -)
        cidx = (np.clip(np.rint(y).astype(np.intp), 1, n - 2) * n
                + np.clip(np.rint(x).astype(np.intp), 1, n - 2))
        cw = np.stack(((1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy))
        idx = np.concatenate(((base + corner_off).ravel(), (cidx + brush_off).ravel()))
        w = np.concatenate(((cw * dep).ravel(), (-BRUSH_W * ero).ravel()))
        hf += np.bincount(idx, weights=w, minlength=nn)

        vis_idx.append(cidx); vis_w.append(water)         # counted once, after the loop

        # move, accelerate downhill, EVAPORATE
        speed = np.sqrt(np.maximum(speed**2 - dh * P.gravity, 0.01))
        water = water * (1 - P.evaporation)
        x, y = xn, yn

    # drops that survive to the end have evaporated: they leave their sediment
    cidx = (np.clip(np.rint(y).astype(np.intp), 1, n - 2) * n
            + np.clip(np.rint(x).astype(np.intp), 1, n - 2))
    hf += np.bincount(cidx, weights=sed, minlength=nn)
    visits = (np.bincount(np.concatenate(vis_idx), weights=np.concatenate(vis_w), minlength=nn)
              if vis_idx else np.zeros(nn))
    return h - h_in, visits.reshape(n, n), eroded, lost


# --------------------------------- world -----------------------------------
class World:
    def __init__(self, size, seed, workers, dem):
        self.size, self.seed, self.workers, self.dem = size, seed, workers, dem
        self.reset()

    def reset(self):
        self.h = self.dem.copy()                          # the real mountain, nothing generated
        self.h0 = self.h.copy()
        self.wet = np.zeros_like(self.h)
        self.deposit_map = np.zeros_like(self.h)      # recent sediment deposition
        self.landslide_map = np.zeros_like(self.h)    # recent landslide activity (red)
        # map edge stays fixed at its real height (water simply leaves the map there)
        self.border = np.zeros(self.h.shape, bool)
        self.border[0, :] = self.border[-1, :] = self.border[:, 0] = self.border[:, -1] = True
        self.border_h = self.h0[self.border].copy()
        self.t = 0
        self.eroded = 0.0
        self.lost = 0.0
        self.rngs = [np.random.default_rng(s)
                     for s in np.random.SeedSequence(self.seed + 1).spawn(self.workers)]

    def step(self, pool, P):
        # split the raindrops between the worker threads
        # threads only pay off when each one gets a big batch (small numpy arrays
        # hold the GIL), so use at most one worker per MIN_DROPS_PER_WORKER drops
        k = max(1, min(self.workers, P.drops // MIN_DROPS_PER_WORKER))
        base, extra = divmod(P.drops, k)
        counts = [base + (i < extra) for i in range(k)]
        futures = [pool.submit(rain_chunk, self.h, c, r, P)
                   for c, r in zip(counts, self.rngs) if c > 0]
        delta = np.zeros_like(self.h)
        visits = np.zeros_like(self.h)
        for f in futures:                      # workers are done reading self.h now
            d, v, e, l = f.result()
            delta += d; visits += v
            self.eroded += e; self.lost += l
        h = self.h
        h_old = h.copy()
        h += delta
        step_deposit = np.maximum(0, h - h_old)
        self.deposit_map = self.deposit_map * 0.98 + step_deposit

        # landslides on over-steep slopes
        ph = np.pad(h, 1, mode="edge")
        diffs = [h - ph[1:-1, :-2], h - ph[1:-1, 2:], h - ph[:-2, 1:-1], h - ph[2:, 1:-1]]
        mv = [P.thermal_rate * np.maximum(0, d - P.talus) for d in diffs]
        moved = sum(mv)                                    # material that slid off each cell
        h -= moved
        self.landslide_map = self.landslide_map * SLIDE_DECAY + moved   # red marks fade
        h[:, :-1] += mv[0][:, 1:]
        h[:, 1:] += mv[1][:, :-1]
        h[:-1, :] += mv[2][1:, :]
        h[1:, :] += mv[3][:-1, :]

        h[self.border] = self.border_h                     # fixed map edge
        np.clip(h, 0, None, out=h)
        visits *= REF_DROPS / max(P.drops, 1)               # same river brightness at any rain level
        self.wet = self.wet * P.wet_decay + visits         # rivers fade without flow
        self.t += 1

    def snapshot(self, sim_rate):
        return dict(
            h=self.h.copy(),
            h0=self.h0,
            wet=self.wet.copy(),
            deposit_map=self.deposit_map.copy(),
            landslide_map=self.landslide_map.copy(),
            t=self.t,
            eroded=self.eroded,
            lost=self.lost,
            sim_rate=sim_rate,
        )


# ------------------------------ timeline (for the years slider) ---------------
class Timeline:
    """Keyframes of the whole simulation indexed by MODEL YEARS, so the slider can scrub
    through the past. Written by the simulation thread, read by the viewer."""

    def __init__(self, max_years, n_frames=120, stop_at_max=True):
        self.max_years = float(max_years)
        self.dy = self.max_years / n_frames
        self.stop_at_max = stop_at_max
        self.lock = threading.Lock()
        self.frames, self.years = [], []
        self.next_y = 0.0

    def record(self, world):
        """Store a keyframe every `dy` model years. Returns True once max_years is reached."""
        yrs = years_elapsed(world.h0, world.h)
        with self.lock:
            if self.years:
                yrs = max(yrs, self.years[-1])
            if not self.frames or yrs >= self.next_y:
                self.frames.append(dict(
                    years=yrs, t=world.t, lost=world.lost, top=highest_point_m(world.h),
                    h=world.h.astype(np.float32), wet=world.wet.astype(np.float32),
                    dep=world.deposit_map.astype(np.float16),
                    slide=world.landslide_map.astype(np.float16)))
                self.years.append(yrs)
                self.next_y = (math.floor(yrs / self.dy) + 1) * self.dy
            return self.stop_at_max and yrs >= self.max_years

    def at(self, years):
        """The keyframe closest to `years` (the latest one if `years` is not computed yet)."""
        with self.lock:
            i = bisect.bisect_right(self.years, years)
            lo, hi = max(i - 1, 0), min(i, len(self.years) - 1)
            return self.frames[lo if years - self.years[lo] <= self.years[hi] - years else hi]

    def latest_years(self):
        with self.lock:
            return self.years[-1] if self.years else 0.0

    def curve(self):
        with self.lock:
            return list(self.years), [f["top"] for f in self.frames]


# ------------------------------ simulation thread -----------------------------
class SimThread(threading.Thread):
    """Steps the world in the background and publishes snapshots.
    lossless=False (live): keeps only the newest snapshot, never waits for the viewer.
    lossless=True  (video): waits for the viewer, so every frame is captured.
    With a timeline it also records keyframes, and stops once max_years is reached."""

    def __init__(self, world, params, steps_per_frame, lossless=False, timeline=None):
        super().__init__(daemon=True)
        self.world, self.params = world, params
        self.steps_per_frame, self.lossless = steps_per_frame, lossless
        self.timeline = timeline
        self.out = queue.Queue(maxsize=2)
        self.stop_event = threading.Event()
        self.paused = threading.Event()
        self.finished = threading.Event()
        self.reset_requested = threading.Event()
        self.pool = ThreadPoolExecutor(max_workers=world.workers)

    def run(self):
        rate = 0.0
        while not self.stop_event.is_set():
            if self.paused.is_set() or self.finished.is_set():
                time.sleep(0.02)
                continue
            if self.reset_requested.is_set():
                self.world.reset()
                self.reset_requested.clear()
            t0 = time.perf_counter()
            P = dataclasses.replace(self.params)          # settings fixed for this frame
            n_steps = self.steps_per_frame                # the viewer raises this to catch up
            for _ in range(n_steps):
                self.world.step(self.pool, P)
            dt = time.perf_counter() - t0
            rate = 0.8 * rate + 0.2 * (n_steps / dt) if rate else n_steps / dt
            if self.timeline is not None and self.timeline.record(self.world):
                self.finished.set()
            snap = self.world.snapshot(rate)
            if self.lossless:
                while not self.stop_event.is_set():
                    try:
                        self.out.put(snap, timeout=0.1)
                        break
                    except queue.Full:
                        pass
            else:
                while True:
                    try:
                        self.out.put_nowait(snap)
                        break
                    except queue.Full:
                        try:
                            self.out.get_nowait()          # drop the oldest
                        except queue.Empty:
                            pass

    def shutdown(self):
        self.stop_event.set()
        self.join(timeout=5)
        self.pool.shutdown(wait=False, cancel_futures=True)


# ------------------------------- drawing ------------------------------------
CMAP = LinearSegmentedColormap.from_list("mountain", [
    (0.00, "#3b2a1a"), (0.30, "#6b4a2b"), (0.55, "#8a6a4a"),
    (0.75, "#9b9690"), (0.90, "#d9d9d9"), (1.00, "#ffffff")])
LS = LightSource(azdeg=315, altdeg=45)
NORM = Normalize(0, HEIGHT)
WATER_RGB = np.array([0.12, 0.40, 0.80])
DEPOSIT_RGB = np.array([0.18, 0.55, 0.22])
LANDSLIDE_RGB = np.array([0.85, 0.15, 0.12])
BG = "#111111"

# Erosion = red, no change = black, deposition = blue
CUSTOM_CHG_CMAP = LinearSegmentedColormap.from_list("RedBlackBlue", [
    (0.00, "#ff3b30"),
    (0.25, "#9a1c18"),
    (0.50, "#000000"),
    (0.75, "#12408f"),
    (1.00, "#3d9bff"),
])


def render_rgb(snap):
    rgb = LS.shade(snap["h"], cmap=CMAP, norm=NORM, vert_exag=2.5, blend_mode="soft")[..., :3]

    # blend deposition green onto the terrain
    dep = snap.get("deposit_map", np.zeros_like(snap["h"]))
    dep_alpha = np.clip(dep * 5.0, 0, 1)[..., None]
    rgb = rgb * (1 - dep_alpha * 0.6) + DEPOSIT_RGB * (dep_alpha * 0.6)

    # blend landslides in red onto the terrain
    sl = snap.get("landslide_map", np.zeros_like(snap["h"]))
    sl_alpha = np.clip(sl * SLIDE_GAIN, 0, 1)[..., None]
    rgb = rgb * (1 - sl_alpha * 0.8) + LANDSLIDE_RGB * (sl_alpha * 0.8)

    # rivers / sea overlay
    a = np.clip((snap["wet"] - 0.25 * WET_SCALE) / WET_SCALE, 0, 1) * 0.85
    a = np.maximum(a, (snap["h"] < SEA_LEVEL) * 0.9)[..., None]
    return rgb * (1 - a) + WATER_RGB * a


class Terrain3DRenderer:
    """Draws the 3D terrain into an image on its own private figure (not shown on
    screen). Because it is an image, the live window only has to paste a picture,
    which is cheap; the slow 3D drawing happens elsewhere."""

    def __init__(self, size, stride, pixels=(823, 405), dpi=90):
        self.fig = Figure(figsize=(pixels[0] / dpi, pixels[1] / dpi), dpi=dpi, facecolor=BG)
        self.canvas = FigureCanvasAgg(self.fig)
        self.ax = self.fig.add_axes([0, 0, 1, 1], projection="3d", facecolor=BG)
        self.ax.set_axis_off()
        self.ax.set_xlim(0, size); self.ax.set_ylim(0, size); self.ax.set_zlim(0, HEIGHT)
        self.ax.set_box_aspect((1, 1, 0.35), zoom=1.3)
        self.stride = stride
        idx = np.arange(0, size, stride)
        self.X, self.Y = np.meshgrid(idx, idx, indexing="ij")
        self.surf = None
        self.shape = (pixels[1], pixels[0], 4)

    def render(self, h, rgb, t):
        s = self.stride
        sub = rgb[::s, ::s]
        rgba = np.dstack([sub, np.ones(sub.shape[:2])])
        if self.surf is not None:
            self.surf.remove()
        self.surf = self.ax.plot_surface(self.X, self.Y, h[::s, ::s], facecolors=rgba,
                                         rstride=1, cstride=1, linewidth=0,
                                         antialiased=False, shade=False)
        self.ax.view_init(elev=38, azim=225 + 0.15 * t)          # slow rotation
        self.canvas.draw()
        return np.asarray(self.canvas.buffer_rgba()).copy()


class Render3DThread(threading.Thread):
    """Third thread: redraws the 3D image whenever the viewer hands it a new snapshot.
    The 2D views never wait for it."""

    def __init__(self, renderer):
        super().__init__(daemon=True)
        self.renderer = renderer
        self.lock = threading.Lock()
        self.pending = None
        self.latest = None
        self.wake = threading.Event()
        self.stop_event = threading.Event()

    def submit(self, h, rgb, t):
        with self.lock:
            self.pending = (h, rgb, t)
        self.wake.set()

    def take(self):
        with self.lock:
            img, self.latest = self.latest, None
        return img

    def run(self):
        while not self.stop_event.is_set():
            if not self.wake.wait(0.1):
                continue
            self.wake.clear()
            with self.lock:
                job, self.pending = self.pending, None
            if job is not None:
                img = self.renderer.render(*job)
                with self.lock:
                    self.latest = img

    def shutdown(self):
        self.stop_event.set()
        self.join(timeout=5)


class Viewer:
    """One window: 3D terrain + 2D map + highest-point curve + erosion/deposition map,
    sliders for rain / evaporation / inertia, and a YEARS slider to scrub through time."""

    def __init__(self, sim, size, video_mode, stride3d=None, label="", info="", timeline=None):
        self.sim, self.video_mode, self.timeline = sim, video_mode, timeline
        self.snap = None
        self.mode = "follow"                 # "follow" = newest state, "scrub" = year chosen on the slider
        self.target_years = 0.0
        self.last_fr = None
        self.sim_rate = 0.0
        self.curve_n = -1
        self.base_spf = sim.steps_per_frame
        n = self.n = size

        stride = stride3d or max(1, size // 128)
        self.r3d = Terrain3DRenderer(size, stride)
        # live: 3D is drawn by its own thread. video: drawn in lockstep, every frame.
        self.thread3d = None if video_mode else Render3DThread(self.r3d)
        if self.thread3d:
            self.thread3d.start()

        self.fig = plt.figure(figsize=(15, 8.4), facecolor=BG)
        self.title = self.fig.text(0.5, 0.985, "starting...", color="w", ha="center", va="top", fontsize=11)

        # TOP ROW:    3D terrain (left)  |  2D river map (right)
        # BOTTOM ROW: curve (left) | erosion/deposition map (middle) | sliders (right)
        # VERY BOTTOM: years slider + LIVE button
        self.ax3 = self.fig.add_axes([0.00, 0.38, 0.60, 0.57]); self.ax3.axis("off")
        self.im3 = self.ax3.imshow(np.zeros(self.r3d.shape), aspect="auto")

        ax_map = self.fig.add_axes([0.61, 0.38, 0.38, 0.50]); ax_map.axis("off")
        head = (f"{label}: rivers (blue), landslides (red)" if label
                else "Map view: rivers (blue), landslides (red)")
        ax_map.set_title(head + (f"\n{info}" if info else ""), color="w", fontsize=9)

        self.ax_curve = self.fig.add_axes([0.06, 0.11, 0.23, 0.21], facecolor="#1a1a1a")
        self.ax_curve.set_title("Highest point (m) over model years", color="w", fontsize=10)
        for sp in self.ax_curve.spines.values():
            sp.set_color("#555")
        self.ax_curve.tick_params(colors="#aaa", labelsize=8)
        (self.line,) = self.ax_curve.plot([], [], color="#4aa3ff", lw=2)
        self.vline = self.ax_curve.axvline(0, color="#f39c12", lw=1)
        if timeline is not None:
            self.ax_curve.set_xlim(0, timeline.max_years)

        self.ax_chg = self.fig.add_axes([0.36, 0.075, 0.30, 0.29], facecolor="black")
        self.ax_chg.axis("off")
        self.ax_chg.set_title("Erosion (red) / Deposition (blue)", color="w", fontsize=9)

        chg = 0.05 * HEIGHT                       # colour range of the change map (was +-4 at HEIGHT 80)
        self.im_map = ax_map.imshow(np.zeros((n, n, 3)), origin="upper")
        self.im_chg = self.ax_chg.imshow(np.zeros((n, n)), cmap=CUSTOM_CHG_CMAP,
                                         vmin=-chg, vmax=chg, origin="upper")
        self.ax_chg.set_xlim(-0.5, n - 0.5)
        self.ax_chg.set_ylim(n - 0.5, -0.5)
        self.cont = None

        # sliders: bottom right
        ax_slider_drops = self.fig.add_axes([0.80, 0.26, 0.16, 0.025], facecolor="#222")
        ax_slider_evap = self.fig.add_axes([0.80, 0.19, 0.16, 0.025], facecolor="#222")
        ax_slider_inert = self.fig.add_axes([0.80, 0.12, 0.16, 0.025], facecolor="#222")

        self.slider_drops = Slider(ax_slider_drops, 'Rain Drops', 100, 15000,
                                   valinit=sim.params.drops, valstep=100, color='#4aa3ff')
        self.slider_evap = Slider(ax_slider_evap, 'Evaporation', 0.001, 0.1,
                                  valinit=sim.params.evaporation, color='#00b894')
        self.slider_inert = Slider(ax_slider_inert, 'Inertia', 0.0, 0.95,
                                   valinit=sim.params.inertia, color='#e17055')
        for s_ in (self.slider_drops, self.slider_evap, self.slider_inert):
            s_.label.set_color('white')
            s_.valtext.set_color('white')
            s_.label.set_fontsize(8)
        self.slider_drops.on_changed(lambda val: setattr(sim.params, 'drops', int(val)))
        self.slider_evap.on_changed(lambda val: setattr(sim.params, 'evaporation', val))
        self.slider_inert.on_changed(lambda val: setattr(sim.params, 'inertia', val))

        # YEARS slider + LIVE button (live window only)
        self.slider_years = self.btn_live = self.status = None
        if timeline is not None and not video_mode:
            ax_y = self.fig.add_axes([0.10, 0.022, 0.70, 0.024], facecolor="#222")
            self.slider_years = Slider(ax_y, "Years", 0, timeline.max_years, valinit=0,
                                       valfmt="%d", color="#f39c12")
            self.slider_years.label.set_color("white"); self.slider_years.label.set_fontsize(10)
            self.slider_years.valtext.set_color("white")
            self.slider_years.on_changed(self._on_years)
            ax_b = self.fig.add_axes([0.86, 0.016, 0.09, 0.036])
            self.btn_live = Button(ax_b, "LIVE", color="#222", hovercolor="#444")
            self.btn_live.label.set_color("white")
            self.btn_live.on_clicked(lambda e: self._go_live())
            self.status = self.fig.text(0.45, 0.058, "", color="#f39c12", ha="center",
                                        va="center", fontsize=9)

        self.fig.canvas.mpl_connect("close_event", lambda e: self.sim.stop_event.set())

    # ---- slider callbacks
    def _on_years(self, val):
        self.mode = "scrub"
        self.target_years = float(val)

    def _go_live(self):
        self.mode = "follow"

    def shutdown(self):
        if self.thread3d:
            self.thread3d.shutdown()

    def _draw_contours(self, h):
        """Thin light contour lines of the terrain height over the change map."""
        if self.cont is not None:
            try:
                self.cont.remove()
            except AttributeError:                    # older matplotlib
                for c in self.cont.collections:
                    c.remove()
        step = HEIGHT / 16                            # 5 units at HEIGHT 80
        self.cont = self.ax_chg.contour(h, levels=np.arange(step, HEIGHT, step),
                                        colors="#cfd8dc", linewidths=0.4, alpha=0.55)
        self.ax_chg.set_xlim(-0.5, self.n - 0.5)
        self.ax_chg.set_ylim(self.n - 0.5, -0.5)

    def newest_snapshot(self):
        if self.video_mode:
            try:
                return self.sim.out.get(timeout=120)
            except queue.Empty:
                return None
        snap = None
        while True:                                   # take the newest, skip older ones
            try:
                snap = self.sim.out.get_nowait()
            except queue.Empty:
                return snap

    def _frame_to_snap(self, fr):
        return dict(h=fr["h"], h0=self.sim.world.h0, wet=fr["wet"],
                    deposit_map=fr["dep"].astype(np.float32),
                    landslide_map=fr["slide"].astype(np.float32), t=fr["t"],
                    lost=fr["lost"], sim_rate=self.sim_rate)

    def _update_curve(self, yrs):
        if self.timeline is None:
            return
        ys, tops = self.timeline.curve()
        if len(ys) != self.curve_n:
            self.curve_n = len(ys)
            self.line.set_data(ys, tops)
            lo, hi = min(tops), max(tops)
            pad = max(2.0, 0.05 * (hi - lo))
            self.ax_curve.set_ylim(lo - pad, hi + pad)
            self.ax_curve.set_xlim(0, max(self.timeline.max_years, ys[-1]))
        self.vline.set_xdata([yrs, yrs])

    def _show(self, snap, yrs, scrubbing, contours):
        rgb = render_rgb(snap)
        self.im_map.set_data(rgb)
        self.im_chg.set_data(snap["h"] - snap["h0"])
        if contours or self.cont is None:
            self._draw_contours(snap["h"])
        if self.video_mode:
            self.im3.set_data(self.r3d.render(snap["h"], rgb, snap["t"]))
        else:
            self.thread3d.submit(snap["h"], rgb, snap["t"])
        self._update_curve(yrs)
        P = self.sim.params
        top_m = highest_point_m(snap["h"])
        rate = DEM_META["rate_mm_yr"]
        if scrubbing:
            self.title.set_text(
                f"YEAR ~{yrs:,.0f}  (scrubbing; press LIVE to follow the simulation)  |  "
                f"highest point {top_m:,.0f} m  |  assumed erosion rate {rate:g} mm/yr")
        else:
            state = "PAUSED" if self.sim.paused.is_set() else ("FINISHED" if self.sim.finished.is_set() else "running")
            self.title.set_text(
                f"Step {snap['t']}  |  {state}  |  rain {P.drops:,} drops/step  |  "
                f"{snap['sim_rate']:.1f} steps/s  |  highest point {top_m:,.0f} m  |  "
                f"model time ~ {yrs:,.0f} years (at {rate:g} mm/yr)")

    def _update_status(self):
        tl = self.timeline
        if self.status is None or tl is None:
            return
        done = tl.latest_years()
        text = f"computed up to year {done:,.0f} of {tl.max_years:,.0f}  ({100 * done / tl.max_years:.0f}%)"
        if self.mode == "scrub" and self.target_years > done + tl.dy and not self.sim.finished.is_set():
            text += f"   |   year {self.target_years:,.0f} not ready yet: computing ahead..."
        elif self.sim.finished.is_set():
            text += "   |   finished"
        self.status.set_text(text)

    def update(self, _frame):
        snap = self.newest_snapshot()
        if snap is not None:
            self.snap = snap
            self.sim_rate = snap["sim_rate"]
        tl = self.timeline
        scrub = self.mode == "scrub" and tl is not None and self.slider_years is not None

        if scrub:
            behind = self.target_years > tl.latest_years() + tl.dy and not self.sim.finished.is_set()
            self.sim.steps_per_frame = self.base_spf * 8 if behind else self.base_spf   # catch up faster
            fr = tl.at(self.target_years)
            if fr is not self.last_fr:                     # only redraw when the keyframe changes
                self.last_fr = fr
                self._show(self._frame_to_snap(fr), fr["years"], True, True)
        else:
            self.sim.steps_per_frame = self.base_spf
            self.last_fr = None
            if snap is not None:
                yrs = years_elapsed(snap["h0"], snap["h"])
                self._show(snap, yrs, False, snap["t"] % 3 == 0)
                if self.slider_years is not None:           # the slider follows the simulation
                    self.slider_years.eventson = False
                    self.slider_years.set_val(min(yrs, tl.max_years))
                    self.slider_years.eventson = True
        self._update_status()
        if self.thread3d:                                   # paste the newest finished 3D picture
            img = self.thread3d.take()
            if img is not None:
                self.im3.set_data(img)


# ------------------------------ start-up mountain menu ----------------------
class PeakMenu:
    """Small start-up window: a drop-down list of mountains and a START button.
    Drawn with Matplotlib only, so it works with whichever backend you already have.
    Click the box to open the list, click a mountain, press START (or Enter)."""

    def __init__(self, default="everest", extent_km=30.0):
        self.keys = sorted(PEAKS, key=lambda k: -PEAK_HEIGHT_M.get(k, 0))
        self.index = self.keys.index(default) if default in self.keys else 0
        self.extent_km = extent_km
        self.hover = None
        self.is_open = False
        self.done = False

        self.fig = plt.figure(figsize=(6.4, 6.8), facecolor=BG)
        try:
            self.fig.canvas.manager.set_window_title("Choose a mountain")
        except Exception:
            pass
        self.fig.text(0.5, 0.945, "Choose a mountain", color="w", ha="center", va="center", fontsize=16)
        self.fig.text(0.5, 0.905, "real elevation data  -  the erosion starts when you press START",
                      color="#888", ha="center", va="center", fontsize=9)

        # the closed drop-down box
        self.ax_head = self.fig.add_axes([0.10, 0.78, 0.80, 0.08], facecolor="#222")
        self.ax_head.set_xticks([]); self.ax_head.set_yticks([])
        for sp in self.ax_head.spines.values():
            sp.set_color("#777")
        tr = self.ax_head.transAxes
        self.head_name = self.ax_head.text(0.04, 0.5, "", color="w", fontsize=13, va="center", transform=tr)
        self.head_arrow = self.ax_head.text(0.96, 0.5, "\u25be", color="#aaa", fontsize=16,
                                            va="center", ha="right", transform=tr)

        # the open list
        n = len(self.keys)
        self.ax_list = self.fig.add_axes([0.10, 0.20, 0.80, 0.57], facecolor="#1b1b1b")
        self.ax_list.set_xticks([]); self.ax_list.set_yticks([])
        self.ax_list.set_xlim(0, 1); self.ax_list.set_ylim(n, 0)          # row i spans y = i .. i+1
        for sp in self.ax_list.spines.values():
            sp.set_color("#777")
        self.row_patches = []
        for i, k in enumerate(self.keys):
            patch = Rectangle((0, i), 1, 1, facecolor="#1b1b1b", edgecolor="none")
            self.ax_list.add_patch(patch)
            self.row_patches.append(patch)
            self.ax_list.text(0.04, i + 0.5, PEAKS[k][0], color="w", fontsize=11, va="center")
            self.ax_list.text(0.96, i + 0.5, f"{PEAK_HEIGHT_M.get(k, 0):,} m", color="#9aa",
                              fontsize=10, va="center", ha="right")
        self.ax_list.set_visible(False)

        self.info = self.fig.text(0.5, 0.168, "", color="#bbb", ha="center", fontsize=10)
        self.status = self.fig.text(0.5, 0.133, "", color="#f39c12", ha="center", fontsize=10)
        ax_btn = self.fig.add_axes([0.10, 0.04, 0.80, 0.08])
        self.btn = Button(ax_btn, "START", color="#2d7d46", hovercolor="#3a9d5a")
        self.btn.label.set_color("white"); self.btn.label.set_fontsize(14)
        self.btn.on_clicked(lambda e: self.start())

        c = self.fig.canvas
        c.mpl_connect("button_press_event", self._on_press)
        c.mpl_connect("motion_notify_event", self._on_move)
        c.mpl_connect("key_press_event", self._on_key)
        self._refresh()

    @property
    def key(self):
        return self.keys[self.index]

    def _refresh(self):
        name, lat, lon = PEAKS[self.key]
        self.head_name.set_text(f"{name}    {PEAK_HEIGHT_M.get(self.key, 0):,} m")
        self.info.set_text(f"{lat:.4f}\u00b0N  {lon:.4f}\u00b0E   -   {self.extent_km:g} km square window")
        for i, p in enumerate(self.row_patches):
            p.set_facecolor("#2a4468" if i == self.index else ("#2c2c2c" if i == self.hover else "#1b1b1b"))
        self.fig.canvas.draw_idle()

    def select(self, i):
        self.index = int(i) % len(self.keys)
        self._refresh()

    def toggle(self, state=None):
        self.is_open = (not self.is_open) if state is None else state
        self.ax_list.set_visible(self.is_open)
        self.head_arrow.set_text("\u25b4" if self.is_open else "\u25be")
        self.fig.canvas.draw_idle()

    def _row_at(self, e):
        if e.inaxes is self.ax_list and e.ydata is not None and 0 <= int(e.ydata) < len(self.keys):
            return int(e.ydata)
        return None

    def _on_press(self, e):
        if e.inaxes is self.ax_head:
            self.toggle()
        elif self.is_open:
            i = self._row_at(e)
            if i is not None:
                self.select(i)
                self.toggle(False)

    def _on_move(self, e):
        if self.is_open:
            h = self._row_at(e)
            if h != self.hover:
                self.hover = h
                self._refresh()

    def _on_key(self, e):
        if e.key == "down":
            self.select(self.index + 1)
        elif e.key == "up":
            self.select(self.index - 1)
        elif e.key == "enter":
            self.start()
        elif e.key == "escape":
            self.toggle(False)

    def start(self):
        if self.done:
            return
        self.toggle(False)
        self.status.set_text(f"Loading {PEAKS[self.key][0]}...  (the first time it downloads the "
                             "elevation tiles, then they are cached)")
        self.fig.canvas.draw_idle()
        try:
            self.fig.canvas.flush_events()
        except Exception:
            pass
        self.done = True

    def close(self):
        plt.close(self.fig)

    @classmethod
    def ask(cls, default="everest", extent_km=30.0):
        """Show the menu and wait. Returns (peak key or None if closed, the menu)."""
        menu = cls(default, extent_km)
        plt.show(block=False)
        while not menu.done and plt.fignum_exists(menu.fig.number):
            plt.pause(0.05)
        return (menu.key if menu.done else None), menu


# ------------------------------ forecast mode --------------------------------
def fast_forward(world, params, targets, max_steps, label):
    """Run as fast as possible (no window) until each target number of model years is reached,
    then draw 'today' next to every checkpoint."""
    targets = sorted(t for t in targets if t > 0)
    snap = lambda yrs: dict(years=yrs, h=world.h.copy(), h0=world.h0, wet=world.wet.copy(),
                            deposit_map=world.deposit_map.copy(),
                            landslide_map=world.landslide_map.copy())
    shots = [snap(0.0)]
    pool = ThreadPoolExecutor(max_workers=world.workers)
    t0, nxt, yrs = time.perf_counter(), 0, 0.0
    print(f"Forecast to {', '.join(f'{t:,.0f}' for t in targets)} years "
          f"(rate {DEM_META['rate_mm_yr']:g} mm/yr). Running without a window...")
    while nxt < len(targets) and world.t < max_steps:
        world.step(pool, params)
        yrs = years_elapsed(world.h0, world.h)
        if world.t % 100 == 0 or world.t == 20:
            sec = (time.perf_counter() - t0) / world.t
            if yrs > 0:
                left = max(targets[-1] - yrs, 0) / (yrs / world.t) * sec
                print(f"  step {world.t:6d}   ~{yrs:10,.0f} years   about {left / 60:.1f} min left")
            else:
                print(f"  step {world.t:6d}   (no material has left the map yet)")
        if yrs >= targets[nxt]:
            shots.append(snap(yrs)); nxt += 1
    pool.shutdown(wait=False, cancel_futures=True)
    if nxt < len(targets):
        print(f"Stopped at --max-steps {max_steps} after ~{yrs:,.0f} model years. "
              "Raise --max-steps, or use a faster --rate-mm-yr, to go further.")
        if yrs > shots[-1]["years"]:
            shots.append(snap(yrs))
    print(f"Done: {world.t} steps in {(time.perf_counter() - t0) / 60:.1f} min.")

    stem = "".join(ch if ch.isalnum() else "_" for ch in label).strip("_").lower()
    m = DEM_META
    for s_ in shots[1:]:                                  # real elevations (m) of each checkpoint
        np.save(f"{stem}_after_{s_['years']:.0f}yr_m.npy", s_["h"] / m["vexag"] * m["cell_m"] + m["elev_min"])

    cols = len(shots)
    r3d = Terrain3DRenderer(world.size, max(1, world.size // 128), pixels=(640, 400), dpi=90)
    fig, ax = plt.subplots(3, cols, figsize=(4.6 * cols, 12.5), facecolor=BG, squeeze=False)
    chg = 0.05 * HEIGHT
    for c, s_ in enumerate(shots):
        rgb = render_rgb(s_)
        ax[0][c].imshow(r3d.render(s_["h"], rgb, 0), aspect="auto")
        ax[1][c].imshow(rgb, origin="upper")
        ax[2][c].imshow(s_["h"] - s_["h0"], cmap=CUSTOM_CHG_CMAP, vmin=-chg, vmax=chg, origin="upper")
        head = "Today" if s_["years"] == 0 else f"after ~{s_['years']:,.0f} years"
        ax[0][c].set_title(f"{head}\nhighest point {highest_point_m(s_['h']):,.0f} m", color="w", fontsize=12)
        ax[1][c].set_title("map (blue = rivers, green = deposits, red = landslides)", color="w", fontsize=9)
        ax[2][c].set_title("change vs today (red = eroded, blue = deposited)", color="w", fontsize=9)
        for r in range(3):
            ax[r][c].axis("off")
    fig.suptitle(f"{label}: estimated terrain after rain erosion, assuming {m['rate_mm_yr']:g} mm/yr average "
                 "denudation (rough estimate; no glaciers, no uplift)", color="w", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = f"{stem}_forecast.png"
    fig.savefig(out, dpi=80, facecolor=BG)
    print("Saved", out, "and the elevation (m) of each checkpoint as", f"{stem}_after_<years>yr_m.npy")
    plt.show()


# -------------------------------- benchmark ----------------------------------
def benchmark(size, seed, drops, dem, steps=12):
    """Find out how many worker threads is fastest on THIS computer."""
    cores = os.cpu_count() or 1
    print(f"CPU cores reported: {cores}   grid {size}x{size}   {drops:,} drops/step")
    P = Params(drops=drops)
    candidates = sorted({1, 2, 4, 8, cores, cores * 2})
    best = None
    for w in candidates:
        world = World(size, seed, w, dem)
        with ThreadPoolExecutor(max_workers=w) as pool:
            world.step(pool, P)                           # warm-up
            t0 = time.perf_counter()
            for _ in range(steps):
                world.step(pool, P)
            rate = steps / (time.perf_counter() - t0)
        print(f"  {w} worker(s): {rate:5.1f} steps/s")
        if best is None or rate > best[1]:
            best = (w, rate)
    print(f"fastest here: --workers {best[0]}")


# ---------------------------------- main -------------------------------------
def build_parser():
    ap = argparse.ArgumentParser(description="Real-time erosion on a real mountain")
    ap.add_argument("--peak", default=None, choices=sorted(PEAKS),
                    help="a Himalayan / Karakoram peak to load. If you give none, a drop-down "
                         "menu opens so you can pick one")
    ap.add_argument("--no-menu", action="store_true",
                    help="skip the start-up menu and use Everest (or whatever --peak says)")
    ap.add_argument("--list-peaks", action="store_true", help="list the built-in peaks and exit")
    ap.add_argument("--lat", type=float, default=None, help="centre latitude (overrides --peak)")
    ap.add_argument("--lon", type=float, default=None, help="centre longitude (overrides --peak)")
    ap.add_argument("--extent-km", type=float, default=30.0,
                    help="width of the square window in km (default 30)")
    ap.add_argument("--vert-exag", type=float, default=1.0,
                    help="vertical exaggeration of the real heights (default 1 = true slopes)")
    ap.add_argument("--talus", type=float, default=None,
                    help="landslide threshold in height units per cell (default 1.3, about 52 deg)")
    ap.add_argument("--dem-file", type=str, default=None,
                    help="use your own .npy elevation array (metres, north up) instead of downloading")
    ap.add_argument("--rate-mm-yr", type=float, default=1.0,
                    help="assumed average denudation rate used to turn erosion into years "
                         "(default 1.0; Everest region ~0.8-1.4, Himalaya overall ~0.15-3)")
    ap.add_argument("--run-years", type=float, nargs="+", default=None, metavar="YEARS",
                    help="forecast mode: run without a window and show the terrain after these "
                         "many model years, e.g. --run-years 1000 10000 50000")
    ap.add_argument("--max-steps", type=int, default=30000, help="safety limit for --run-years")
    ap.add_argument("--size", type=int, default=256, help="grid size (128 fast, 192/256 slower)")
    ap.add_argument("--seed", type=int, default=29)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1,
                    help="max worker threads; only used when there are enough drops "
                         "per thread (see MIN_DROPS_PER_WORKER)")
    ap.add_argument("--drops", type=int, default=None, help="raindrops per step (default 1500)")
    ap.add_argument("--lifetime", type=int, default=None,
                    help="max moves per raindrop = how far water can travel (default 150)")
    ap.add_argument("--evaporation", type=float, default=None,
                    help="fraction of a drop's water lost per move (default 0.008)")
    ap.add_argument("--max-years", type=float, default=10000.0,
                    help="right end of the YEARS slider; the simulation computes up to here (default 10000)")
    ap.add_argument("--keyframes", type=int, default=120,
                    help="how many time points are stored for the YEARS slider (memory ~ 0.6 MB each)")
    ap.add_argument("--steps-per-frame", type=int, default=None)
    ap.add_argument("--video", type=str, default=None, help="render to this .mp4/.gif instead of a live window")
    ap.add_argument("--frames", type=int, default=400, help="frames when rendering a video")
    ap.add_argument("--stride3d", type=int, default=None,
                    help="3D mesh coarseness (1 = full detail but slow; default is automatic)")
    ap.add_argument("--benchmark", action="store_true",
                    help="time the simulation with several worker counts and exit")
    return ap


def main():
    a = build_parser().parse_args()

    if a.list_peaks:
        for k, (name, lat, lon) in sorted(PEAKS.items()):
            print(f"  {k:14s} {name:16s} {lat:8.4f}N {lon:8.4f}E")
        return

    menu = None
    if a.peak is None and a.lat is None and a.lon is None and not a.dem_file:
        windowed = not (a.video or a.benchmark or a.run_years or a.no_menu)
        if windowed:
            a.peak, menu = PeakMenu.ask(extent_km=a.extent_km)      # the drop-down
            if a.peak is None:
                return                                              # window closed without START
        else:
            a.peak = "everest"

    dem, label, info = load_mountain(a)
    if menu is not None:
        menu.close()
    configure_for_dem(dem)
    report_slopes(dem, a.talus or Params.talus)

    if a.benchmark:
        benchmark(a.size, a.seed, a.drops or Params.drops, dem)
        return

    video_mode = a.video is not None
    spf = a.steps_per_frame or (2 if video_mode else 1)
    params = Params()
    for field, value in (("drops", a.drops), ("lifetime", a.lifetime),
                         ("evaporation", a.evaporation), ("talus", a.talus)):
        if value is not None:
            setattr(params, field, value)
    world = World(a.size, a.seed, a.workers, dem)
    if a.run_years:
        fast_forward(world, params, a.run_years, a.max_steps, label)
        return
    timeline = Timeline(a.max_years, a.keyframes, stop_at_max=not video_mode)
    timeline.record(world)                                  # year 0
    sim = SimThread(world, params, spf, lossless=video_mode, timeline=timeline)
    viewer = Viewer(sim, a.size, video_mode, a.stride3d, label, info, timeline)
    print(f"{a.workers} worker thread(s), grid {a.size}x{a.size}, "
          f"{'video' if video_mode else 'live'} mode")
    sim.start()

    try:
        if video_mode:
            ani = FuncAnimation(viewer.fig, viewer.update, frames=a.frames, repeat=False)
            ext = os.path.splitext(a.video)[1].lower()
            if ext == ".gif":
                ani.save(a.video, writer="pillow", fps=20, dpi=80)
            else:
                ani.save(a.video, fps=25, dpi=110, savefig_kwargs={"facecolor": BG})
            print("saved", a.video)
        else:
            ani = FuncAnimation(viewer.fig, viewer.update, interval=30,
                                cache_frame_data=False)
            plt.show()
    finally:
        viewer.shutdown()
        sim.shutdown()


if __name__ == "__main__":
    main()
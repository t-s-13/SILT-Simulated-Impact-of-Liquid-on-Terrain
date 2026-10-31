"""
Real-time rainfall + evaporation erosion simulation (NumPy + Matplotlib only).
FAST VERSION: vectorised raindrop loop + more worker threads.
+ LANDSLIDES marked in red  + YEARS SLIDER (time machine).

PHYSICS (particle / raindrop method)
  Rain      thousands of raindrops land at random places every step.
  Flow      each drop rolls downhill, steered by the slope plus some inertia.
  Erosion   fast, deep drops on steep ground pick up sediment.
  Deposit   slow or uphill drops drop it (fans, plains, filled basins).
  Evaporate each drop loses water as it moves; a dried-up drop leaves its sediment.
  Landslide slopes steeper than the talus angle collapse downhill.
            Recent landslides are marked with semi-transparent RED on the
            river map and on the 3D terrain.
  Sea       the map border is the sea; drops that reach it are removed.

YEARS SLIDER
  The slider under the window is a time machine. While the simulation runs it follows along
  (press LIVE to go back to following it). Drag it to any year up to --max-years: the window
  shows the terrain at that year, and if that year has not been simulated yet it computes
  ahead at high speed until it gets there.

MODEL YEARS (rough!)
  The particle model has no clock of its own, so time is defined through the amount of
  rock removed: model years = (average lowering of the land, in metres) / (average
  denudation rate). Metres come from --cell-m (how wide one grid cell is; 1 cell = 1 height
  unit, so the 80-unit island is 80 * cell-m metres tall). Default rate 1 mm/yr.
  Treat the years as an order-of-magnitude estimate.

SPEED NOTES
  * One np.bincount per drop-iteration (deposit corners + erosion brush merged),
    instead of 13 bincounts and Python loops.
  * Dead drops (sea / flat) are removed from the arrays instead of masked.
  * Flat-index gathers (ndarray.take) instead of 2D fancy indexing.
  * Change-map uses imshow.set_data instead of clear() + contourf() every frame.
  * Worker threads adapt: one thread per 20000 drops, up to --workers (default: CPU cores).

RUN
  python erosion_realtime_fast_years_slides.py                       live window
  python erosion_realtime_fast_years_slides.py --max-years 50000     longer time slider
  python erosion_realtime_fast_years_slides.py --rate-mm-yr 0.5      slower erosion
  python erosion_realtime_fast_years_slides.py --cell-m 100          bigger island (100 m per cell)
  python erosion_realtime_fast_years_slides.py --video out.mp4       render a video (mp4 needs ffmpeg, else .gif)
  python erosion_realtime_fast_years_slides.py --benchmark           find the best --workers for your PC
  python erosion_realtime_fast_years_slides.py --help                all options
"""
import argparse
import bisect
import dataclasses
import math
import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import LightSource, LinearSegmentedColormap, Normalize
from matplotlib.figure import Figure
from matplotlib.widgets import Button, Slider

HEIGHT = 80.0            # max terrain height in model units (1 cell = 1 unit wide)


# ============================ SETTINGS ====================================
@dataclasses.dataclass
class Params:
    """Settings the sim thread reads every step (the sliders change them live)."""
    drops: int = 3000            # raindrops released per step
    lifetime: int = 45           # max moves per drop
    evaporation: float = 0.025   # fraction of a drop's water lost per move
    inertia: float = 0.30        # 0 = follows slope exactly, 1 = ignores slope
    capacity: float = 4.0        # sediment a fast, deep drop can carry
    min_slope: float = 0.02      # capacity floor so drops still carry on flat ground
    erode_rate: float = 0.0002   # fraction of spare capacity picked up per move
    deposit_rate: float = 0.20   # fraction of excess sediment dropped per move
    gravity: float = 4.0         # how strongly slopes accelerate drops
    talus: float = 1.3           # max stable height difference between neighbours
    thermal_rate: float = 0.10   # fraction of excess moved per step
    wet_decay: float = 0.894     # rivers fade by this factor each step


WET_SCALE = 260.0        # flow needed for a river to show fully blue
REF_DROPS = 3000.0       # river display is normalised to this rain amount, so the rain slider
                         # changes the speed of erosion but not how visible the rivers are
SEA_LEVEL = 0.5          # cells below this height are drawn as sea
MIN_DROPS_PER_WORKER = 20000  # don't split rain into chunks smaller than this

# --- landslide marker (red) ---
SLIDE_DECAY = 0.90       # red fades by this factor each step (higher = lingers longer)
SLIDE_MIN = 0.10         # slid amount below this is not shown (raise = less red)
SLIDE_FULL = 0.60        # slid amount that gives the strongest red (lower = redder)
SLIDE_ALPHA = 0.55       # opacity of the strongest red: 1 = solid, lower = more see-through

# --- model years ---
TIME_META = dict(cell_m=50.0, rate_mm_yr=1.0)   # metres per cell, assumed denudation rate
# ==========================================================================


# ------------------------------ model years --------------------------------
def years_elapsed(h0, h):
    """Model years: metres of average lowering divided by the assumed denudation rate."""
    lowering_m = (h0.mean() - h.mean()) * TIME_META["cell_m"]
    return max(lowering_m, 0.0) / (TIME_META["rate_mm_yr"] / 1000.0)


def highest_point_m(h):
    return float(h.max()) * TIME_META["cell_m"]


# ------------------------------ terrain ------------------------------------
def perlin(size, res, rng):
    angles = rng.uniform(0, 2 * np.pi, (res + 1, res + 1))
    grad = np.dstack((np.cos(angles), np.sin(angles)))
    lin = np.linspace(0, res, size, endpoint=False)
    x, y = np.meshgrid(lin, lin, indexing="ij")
    xi, yi = x.astype(int), y.astype(int)
    xf, yf = x - xi, y - yi

    def dot(ix, iy, dx, dy):
        g = grad[ix, iy]
        return g[..., 0] * dx + g[..., 1] * dy

    fade = lambda t: 6 * t**5 - 15 * t**4 + 10 * t**3
    u, v = fade(xf), fade(yf)
    n00 = dot(xi, yi, xf, yf)
    n10 = dot(xi + 1, yi, xf - 1, yf)
    n01 = dot(xi, yi + 1, xf, yf - 1)
    n11 = dot(xi + 1, yi + 1, xf - 1, yf - 1)
    return (n00 * (1 - u) + n10 * u) * (1 - v) + (n01 * (1 - u) + n11 * u) * v


def smooth(a, passes):
    for _ in range(passes):
        p = np.pad(a, 1, mode="edge")
        a = (p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:] + 4 * a) / 8
    return a


def make_terrain(size, seed):
    rng = np.random.default_rng(seed)
    t = sum(w * perlin(size, r, rng)
            for r, w in [(2, 1), (4, 0.75), (8, 0.5), (16, 0.25)])
    t = (t - t.min()) / (t.max() - t.min())
    t = smooth(t, passes=3)
    # island: land slopes down to the sea at the border so water can drain out
    idx = np.arange(size)
    edge = np.minimum(idx, size - 1 - idx)
    e = np.clip(edge / (0.25 * size), 0, 1)          # 1D ramp from the border inward
    w = e * e * (3 - 2 * e)
    t = t * np.outer(w, w)                           # product of two 1D ramps: no diagonal crease
    t = (t - t.min()) / (t.max() - t.min())
    t += rng.normal(0, 0.002, t.shape)               # tiny noise to seed channels
    return np.clip(t, 0, None) * HEIGHT


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

        # drops leaving the safe area reach the sea and are removed;
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
    def __init__(self, size, seed, workers):
        self.size, self.seed, self.workers = size, seed, workers
        self.reset()

    def reset(self):
        self.h = make_terrain(self.size, self.seed)
        self.h0 = self.h.copy()
        self.wet = np.zeros_like(self.h)
        self.deposit_map = np.zeros_like(self.h)      # recent sediment deposition
        self.slide_map = np.zeros_like(self.h)        # recent landslide activity
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
        moved = sum(mv)                                    # material leaving each over-steep cell
        h -= moved
        self.slide_map = self.slide_map * SLIDE_DECAY + moved   # remember recent landslides
        h[:, :-1] += mv[0][:, 1:]
        h[:, 1:] += mv[1][:, :-1]
        h[:-1, :] += mv[2][1:, :]
        h[1:, :] += mv[3][:-1, :]

        h[0, :] = h[-1, :] = h[:, 0] = h[:, -1] = 0        # the sea
        np.clip(h, 0, None, out=h)
        self.wet = self.wet * P.wet_decay + visits         # rivers fade without flow
        self.t += 1

    def snapshot(self, sim_rate):
        return dict(
            h=self.h.copy(),
            h0=self.h0,
            wet=self.wet.copy(),
            deposit_map=self.deposit_map.copy(),
            slide_map=self.slide_map.copy(),
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
                    slide=world.slide_map.astype(np.float16)))
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
SLIDE_RGB = np.array([0.95, 0.10, 0.10])           # landslides: red
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

    # rivers / sea overlay
    a = np.clip((snap["wet"] - 0.25 * WET_SCALE) / WET_SCALE, 0, 1) * 0.85
    a = np.maximum(a, (snap["h"] < SEA_LEVEL) * 0.9)[..., None]
    rgb = rgb * (1 - a) + WATER_RGB * a

    # landslide markers: semi-transparent red on top, so the terrain still shows through.
    # (this one rgb image feeds both the 2D map and the 3D view)
    slide = snap.get("slide_map", np.zeros_like(snap["h"]))
    s = np.clip((slide - SLIDE_MIN) / (SLIDE_FULL - SLIDE_MIN), 0, 1)[..., None] * SLIDE_ALPHA
    return rgb * (1 - s) + SLIDE_RGB * s


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
    """One window: 3D terrain + 2D map + highest-point curve + erosion/deposition map + sliders,
    and a YEARS slider to scrub through time."""

    def __init__(self, sim, size, video_mode, stride3d=None, timeline=None):
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
        self.title = self.fig.text(0.5, 0.97, "starting...", color="w", ha="center", fontsize=12)

        # TOP ROW:    3D terrain (left)  |  2D river map (right)
        # BOTTOM ROW: curve (left) | erosion/deposition map (middle) | sliders (right)
        # VERY BOTTOM: years slider + LIVE button
        self.ax3 = self.fig.add_axes([0.00, 0.38, 0.60, 0.57]); self.ax3.axis("off")
        self.im3 = self.ax3.imshow(np.zeros(self.r3d.shape), aspect="auto")

        ax_map = self.fig.add_axes([0.61, 0.38, 0.38, 0.57]); ax_map.axis("off")
        ax_map.set_title("Map view: rivers (blue), landslides (red)", color="w", fontsize=10)

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

        self.im_map = ax_map.imshow(np.zeros((n, n, 3)), origin="upper")
        self.im_chg = self.ax_chg.imshow(np.zeros((n, n)), cmap=CUSTOM_CHG_CMAP,
                                         vmin=-4, vmax=4, origin="upper")
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

        for s in (self.slider_drops, self.slider_evap, self.slider_inert):
            s.label.set_color('white')
            s.valtext.set_color('white')
            s.label.set_fontsize(8)

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
        self.cont = self.ax_chg.contour(h, levels=np.arange(5, HEIGHT, 5),
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
                    slide_map=fr["slide"].astype(np.float32), t=fr["t"],
                    lost=fr["lost"], eroded=0.0, sim_rate=self.sim_rate)

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
        rate = TIME_META["rate_mm_yr"]
        if scrubbing:
            self.title.set_text(
                f"YEAR ~{yrs:,.0f}  (scrubbing; press LIVE to follow the simulation)  |  "
                f"highest point {top_m:,.0f} m  |  assumed erosion rate {rate:g} mm/yr")
        else:
            state = "PAUSED" if self.sim.paused.is_set() else (
                "FINISHED" if self.sim.finished.is_set() else "running")
            self.title.set_text(
                f"Step {snap['t']}  |  {state}  |  rain {P.drops:,} drops/step  |  "
                f"evaporation {P.evaporation:.3f}  |  {snap['sim_rate']:.1f} steps/s  |  "
                f"highest point {top_m:,.0f} m  |  model time ~ {yrs:,.0f} years (at {rate:g} mm/yr)  |  "
                f"sediment to sea {snap['lost']:.0f}")

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
                self._show(snap, yrs, False, snap["t"] % 3 == 0)   # contours every 3rd step
                if self.slider_years is not None:           # the slider follows the simulation
                    self.slider_years.eventson = False
                    self.slider_years.set_val(min(yrs, tl.max_years))
                    self.slider_years.eventson = True
        self._update_status()
        if self.thread3d:                                   # paste the newest finished 3D picture
            img = self.thread3d.take()
            if img is not None:
                self.im3.set_data(img)


# -------------------------------- benchmark ----------------------------------
def benchmark(size, seed, drops, steps=12):
    """Find out how many worker threads is fastest on THIS computer."""
    cores = os.cpu_count() or 1
    print(f"CPU cores reported: {cores}   grid {size}x{size}   {drops:,} drops/step")
    P = Params(drops=drops)
    candidates = sorted({1, 2, 4, 8, cores, cores * 2})
    best = None
    for w in candidates:
        world = World(size, seed, w)
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
def main():
    ap = argparse.ArgumentParser(description="Real-time rainfall + evaporation erosion")
    ap.add_argument("--size", type=int, default=256, help="grid size (128 fast, 192/256 slower)")
    ap.add_argument("--seed", type=int, default=29)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1,
                    help="max worker threads; only used when there are enough drops "
                         "per thread (see MIN_DROPS_PER_WORKER)")
    ap.add_argument("--drops", type=int, default=3000, help="raindrops per step")
    ap.add_argument("--cell-m", type=float, default=50.0,
                    help="metres per grid cell; sets the real-world scale (island is 80 * this tall)")
    ap.add_argument("--rate-mm-yr", type=float, default=1.0,
                    help="assumed average denudation rate used to turn erosion into years (default 1.0)")
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
    a = ap.parse_args()

    TIME_META.update(cell_m=a.cell_m, rate_mm_yr=a.rate_mm_yr)

    if a.benchmark:
        benchmark(a.size, a.seed, a.drops)
        return

    video_mode = a.video is not None
    spf = a.steps_per_frame or (2 if video_mode else 1)
    params = Params(drops=a.drops)
    world = World(a.size, a.seed, a.workers)
    timeline = Timeline(a.max_years, a.keyframes, stop_at_max=not video_mode)
    timeline.record(world)                                  # year 0
    sim = SimThread(world, params, spf, lossless=video_mode, timeline=timeline)
    viewer = Viewer(sim, a.size, video_mode, a.stride3d, timeline)
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
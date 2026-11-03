"""
Step 1 of the real-noise injections: O4b segment lists for the H1+L1 network.

The GWOSC files hold, next to the strain, two 1 Hz bitmasks (arXiv:2605.27090, Tables 7-8):
    {ifo}:GWOSC-4KHZ_R1_DQMASK  - bit 1 = passes CBC_CAT1 (in O4b CBC_CAT2/3 coincide with it)
    {ifo}:GWOSC-4KHZ_R1_INJMASK - bit 0 = NO CBC hardware injection (1 = no injection)
A second is "good" when both bits are 1.

Input: the GWOSC 4 kHz O4b frames (.gwf), in the subfolders H1/ and L1/ of O4B_DATA_DIR
(set in config/params.py).

Outputs (bilby_inference/segments/):
    h1l1_on.json         - times when H1 and L1 are both good
    events.json          - GPS times of the GWOSC catalog events in O4b and the vetoed windows
    usable.json          - h1l1_on minus the event windows, only stretches long enough
                           for one injection (PSD + analysis + pads)
    injection_times.json - N_INJECTIONS geocentric times, one per disjoint block

Usage (from the repository root):
    python bilby_inference/scripts/make_segments.py
"""

import sys
import glob
import json
import os
import re
import warnings

import numpy as np
from astropy.units import UnitsWarning
from gwpy.timeseries import TimeSeriesDict
from gwpy.segments import Segment, SegmentList

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BILBY_DIR = os.path.dirname(SCRIPT_DIR)      # .../bilby_inference
PROJECT_DIR = os.path.dirname(BILBY_DIR)     # repository root
sys.path.insert(0, PROJECT_DIR)

from config.params import O4B_DATA_DIR as DATA_DIR   # folder containing H1/, L1/

# Edit here
GWF_FORMAT = "gwf.framel"    # force the FrameL backend: gwpy's frameCPP backend may clash
                             # with the installed frameCPP version
IFOS = ["H1", "L1"]
GPS_START, GPS_END = 1396796418, 1422118818  # O4b proper (drops the ER16 data before it)
DQ_BIT_CBC_CAT1 = 1
INJ_BIT_NO_CBC_HW_INJ = 0

DURATION = 32.0          # s - must match inference_utils.DURATION
POST_TRIGGER = 2.0       # s - analysis segment ends 2 s after t_c, as in make_ifo_network
PSD_DURATION = 1024.0    # s - off-source stretch used to estimate the PSD (32 x DURATION)
DATA_PAD = 8.0           # s - margin on each side, cropped away after resampling
EVENT_PAD_BEFORE = 300.0 # s - covers long inspirals entering the band before merger
EVENT_PAD_AFTER = 30.0   # s
N_INJECTIONS = 20
SEED = 1234

OUTDIR = os.path.join(BILBY_DIR, "segments")
REQUIRED_SPAN = PSD_DURATION + DURATION + 2 * DATA_PAD   # contiguous clean data per injection

# The GWOSC mask channels store the bit names in the "unit" field: harmless, but noisy
warnings.filterwarnings("ignore", category=UnitsWarning)

_FNAME_GPS = re.compile(r"-(\d+)-(\d+)\.gwf$")


def mask_to_segments(good, t0, dt):
    """Contiguous runs of True in a boolean array sampled every dt seconds from t0."""
    edges = np.diff(np.concatenate([[0], good.astype(int), [0]]))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return SegmentList([Segment(t0 + s * dt, t0 + e * dt) for s, e in zip(starts, ends)])


def detector_segments(ifo):
    """Good-data segments of one detector, built from the masks of every file."""
    files = []
    for path in glob.glob(os.path.join(DATA_DIR, ifo, "*.gwf")):
        match = _FNAME_GPS.search(os.path.basename(path))
        f_start, f_dur = int(match.group(1)), int(match.group(2))
        if f_start < GPS_END and f_start + f_dur > GPS_START:
            files.append((f_start, path))
    files.sort()

    dq_channel = f"{ifo}:GWOSC-4KHZ_R1_DQMASK"
    inj_channel = f"{ifo}:GWOSC-4KHZ_R1_INJMASK"
    segments = SegmentList()
    bad_files = []
    for k, (_, path) in enumerate(files):
        try:
            masks = TimeSeriesDict.read(path, [dq_channel, inj_channel], format=GWF_FORMAT)
        except Exception as err:   # unreadable/incomplete file: treat its span as no data
            print(f"  WARNING {ifo}: skipping {os.path.basename(path)} ({type(err).__name__}: {err})")
            bad_files.append(path)
            continue
        dq = masks[dq_channel].value.astype(int)
        inj = masks[inj_channel].value.astype(int)
        good = (((dq >> DQ_BIT_CBC_CAT1) & 1) == 1) & (((inj >> INJ_BIT_NO_CBC_HW_INJ) & 1) == 1)
        segments.extend(mask_to_segments(good, masks[dq_channel].t0.value,
                                         masks[dq_channel].dt.value))
        if (k + 1) % 500 == 0:
            print(f"  {ifo}: {k + 1}/{len(files)} files")

    segments = segments.coalesce() & SegmentList([Segment(GPS_START, GPS_END)])
    return segments.coalesce(), len(files), bad_files


def catalog_events():
    """GPS times of all GWOSC event datasets inside [GPS_START, GPS_END)."""
    from gwosc.datasets import find_datasets, event_gps
    events = {}
    for name in find_datasets(type="event", segment=(GPS_START, GPS_END)):
        gps = float(event_gps(name))
        if GPS_START <= gps < GPS_END:
            events[name] = gps
    return events


def draw_injection_times(usable, rng, n):
    """
    Tile every usable segment into disjoint blocks of length REQUIRED_SPAN and place t_c
    inside each block so that PSD stretch, analysis segment and pads all fit:
        block = [PAD | PSD_DURATION | DURATION (t_c at DURATION - POST_TRIGGER) | PAD]
    Then pick n blocks at random: disjoint blocks -> independent noise realizations.
    """
    offset = DATA_PAD + PSD_DURATION + DURATION - POST_TRIGGER
    candidates = []
    for seg in usable:
        n_blocks = int(abs(seg) // REQUIRED_SPAN)
        candidates += [float(seg[0]) + k * REQUIRED_SPAN + offset for k in range(n_blocks)]
    if len(candidates) < n:
        raise RuntimeError(f"Only {len(candidates)} independent blocks, {n} requested")
    return sorted(rng.choice(candidates, size=n, replace=False).tolist()), len(candidates)


def livetime_days(seglist):
    """Total duration of a segment list, in days."""
    return float(sum(abs(seg) for seg in seglist)) / 86400.0


def to_json(seglist):
    """Segment list as a JSON-serializable list of [start, end] pairs."""
    return [[float(seg[0]), float(seg[1])] for seg in seglist]


def main():
    os.makedirs(OUTDIR, exist_ok=True)

    network = SegmentList([Segment(GPS_START, GPS_END)])
    bad_files = {}
    for ifo in IFOS:
        segs, n_files, bad_files[ifo] = detector_segments(ifo)
        print(f"{ifo}: {n_files} files ({len(bad_files[ifo])} unreadable), "
              f"{len(segs)} segments, {livetime_days(segs):.2f} d")
        network &= segs
    network = network.coalesce()
    print(f"{'+'.join(IFOS)} on: {len(network)} segments, {livetime_days(network):.2f} d")

    events = catalog_events()
    veto = SegmentList([Segment(t - EVENT_PAD_BEFORE, t + EVENT_PAD_AFTER)
                        for t in events.values()]).coalesce()
    print(f"Catalog events in O4b: {len(events)}")

    usable = SegmentList([seg for seg in (network - veto).coalesce()
                          if abs(seg) >= REQUIRED_SPAN])
    print(f"Usable: {len(usable)} segments, {livetime_days(usable):.2f} d "
          f"(min length {REQUIRED_SPAN:.0f} s)")

    times, n_blocks = draw_injection_times(usable, np.random.default_rng(SEED), N_INJECTIONS)
    print(f"Drew {len(times)} injection times out of {n_blocks} independent blocks")

    meta = dict(ifos=IFOS, gps_start=GPS_START, gps_end=GPS_END,
                dq_bit_cbc_cat1=DQ_BIT_CBC_CAT1, inj_bit_no_cbc_hw_inj=INJ_BIT_NO_CBC_HW_INJ)
    outputs = {
        "h1l1_on.json": dict(meta, livetime_days=livetime_days(network), bad_files=bad_files,
                             segments=to_json(network)),
        "events.json": dict(meta, pad_before=EVENT_PAD_BEFORE, pad_after=EVENT_PAD_AFTER,
                            events=events, veto_segments=to_json(veto)),
        "usable.json": dict(meta, required_span=REQUIRED_SPAN,
                            livetime_days=livetime_days(usable), segments=to_json(usable)),
        "injection_times.json": dict(meta, seed=SEED, duration=DURATION,
                                     post_trigger=POST_TRIGGER, psd_duration=PSD_DURATION,
                                     data_pad=DATA_PAD, n_blocks=n_blocks,
                                     geocent_times=times),
    }
    for fname, content in outputs.items():
        path = os.path.join(OUTDIR, fname)
        with open(path, "w") as f:
            json.dump(content, f, indent=2)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
train_matcha.py — retrain the Matcha Match scorer on the 14 perceptual features.

The browser's headline score is now a *transparent* colorimetric scorer (see
`perceptualScore()` in index.html). This script lets you train a learned SVR on
top of the same 14 features so the final 1–5 can be calibrated to human taste
ratings, and emits the artifacts to paste back into index.html:

  * a base64-encoded ONNX pipeline (StandardScaler + RBF SVR) for MODEL_B64
  * SCALER_MEAN / SCALER_SCALE arrays for the community KNN

The feature extraction here MIRRORS index.html's `extractPerceptualFeatures`
exactly (same segmentation, same sRGB→Lab/HSV math, same circular hue, same
white-balance estimator). Keep the two in sync if you touch either.

Usage
-----
  # 1) prepare a CSV: image_path,rating  (rating is a float 1.0–5.0)
  python train_matcha.py --csv labels.csv --images_dir ./photos --out onnx

  # 2) verify feature-extraction parity with the in-browser sanity harness
  #    (no data needed — scores synthetic patches and prints them):
  python train_matcha.py --selftest

Requirements
------------
  pip install numpy pillow scikit-learn skl2onnx onnxruntime

Notes
-----
  * Images are resized to 400×400 exactly as the browser canvas does.
  * Cross-validation (5-fold) R² / MAE are reported.
  * If you have fewer than ~20 rated images, prefer the transparent scorer; an
    SVR on 14 features needs at least ~5–10 samples per feature to be stable.
"""

import argparse
import base64
import csv
import os
import sys
import math
import struct
from pathlib import Path

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Color math — mirror of the JS helpers in index.html (rgbToHsv / rgbToLab)
# ---------------------------------------------------------------------------
def rgb_to_hsv(r, g, b):
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    mx, mn = max(r, g, b), min(r, g, b)
    d = mx - mn
    h = 0.0
    if d > 0:
        if mx == r:
            h = (60 * ((g - b) / d) + 360) % 360
        elif mx == g:
            h = (60 * ((b - r) / d) + 120) % 360
        else:
            h = (60 * ((r - g) / d) + 240) % 360
    s = (d / mx) if mx > 0 else 0.0
    return h, s, mx


def _f_lab(t):
    return t ** (1 / 3) if t > 0.008856 else 7.787 * t + 16 / 116


def rgb_to_lab(r, g, b):
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    r = r ** 2.4 / 12.92 if r <= 0.04045 else ((r + 0.055) / 1.055) ** 2.4
    g = g ** 2.4 / 12.92 if g <= 0.04045 else ((g + 0.055) / 1.055) ** 2.4
    b = b ** 2.4 / 12.92 if b <= 0.04045 else ((b + 0.055) / 1.055) ** 2.4
    x = (r * 0.4124564 + g * 0.3575761 + b * 0.1804375) / 0.95047
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = (r * 0.0193339 + g * 0.1191920 + b * 0.9503041) / 1.08883
    return 116 * _f_lab(y) - 16, 500 * (_f_lab(x) - _f_lab(y)), 200 * (_f_lab(y) - _f_lab(z))


def percentile(arr, p):
    arr = np.sort(np.asarray(arr, dtype=float))
    if len(arr) == 0:
        return 0.0
    idx = (p / 100.0) * (len(arr) - 1)
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return float(arr[lo])
    return float(arr[lo] + (arr[hi] - arr[lo]) * (idx - lo))


def circular_hue_mean(hues):
    if len(hues) == 0:
        return 0.0
    rad = np.deg2rad(hues)
    d = math.degrees(math.atan2(np.sin(rad).sum(), np.cos(rad).sum()))
    return d + 360 if d < 0 else d


def circular_hue_distance(h1, h2):
    d = abs(h1 - h2) % 360
    return 360 - d if d > 180 else d


# ---------------------------------------------------------------------------
# White balance — mirror of estimateWhiteBalance() in index.html
# ---------------------------------------------------------------------------
def estimate_white_balance(pixels):
    # pixels: (H*W, 3) uint8
    rgb = pixels.astype(float)
    bright = rgb.sum(axis=1)
    p20, p80 = percentile(bright, 20), percentile(bright, 80)
    # HSV saturation per pixel
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    s = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-9), 0.0)
    mask = (s < 0.25) & (bright >= p20) & (bright <= p80)
    if mask.sum() >= 50:
        mr, mg, mb = r[mask].mean(), g[mask].mean(), b[mask].mean()
        if min(mr, mg, mb) >= 10:
            avg = (mr + mg + mb) / 3
            return avg / mr, avg / mg, avg / mb
    return 1.0, 1.0, 1.0


# ---------------------------------------------------------------------------
# Feature extraction — mirror of extractPerceptualFeatures() in index.html
# ---------------------------------------------------------------------------
IDEAL_RGB = (100, 165, 60)
IDEAL_LAB = rgb_to_lab(*IDEAL_RGB)
IDEAL_HUE = rgb_to_hsv(*IDEAL_RGB)[0]


def extract_features(img):
    """img: PIL.Image (any size) -> 14-feature np.float32 vector. Mirrors the JS exactly."""
    img = img.convert("RGB").resize((400, 400), Image.BILINEAR)
    arr = np.asarray(img)  # (400,400,3)
    H, W = 400, 400
    mh = int(H * 0.12)
    mw = int(W * 0.12)

    flat = arr.reshape(-1, 3)
    wbR, wbG, wbB = estimate_white_balance(flat)

    # Center region
    cy0, cy1, cx0, cx1 = mh, H - mh, mw, W - mw
    center = arr[cy0:cy1, cx0:cx1].reshape(-1, 3).astype(float)
    r, g, b = center[:, 0], center[:, 1], center[:, 2]
    green_ish = (g > r * 0.85) & (g > b * 0.95)
    not_dark = (r + g + b) > 120
    not_bright = (r + g + b) < 720
    has_color = (np.abs(r - g) + np.abs(g - b) + np.abs(r - b)) > 20
    mask = green_ish & not_dark & not_bright & has_color

    mr, mg, mb = r[mask], g[mask], b[mask]
    if len(mr) < 500:
        # fallback: centered circle on the center region
        ch, cw = cy1 - cy0, cx1 - cx0
        cx, cy = cw / 2, ch / 2
        rad = min(ch, cw) / 3
        yy, xx = np.mgrid[0:ch, 0:cw]
        circ = ((xx - cx) ** 2 + (yy - cy) ** 2) <= rad * rad
        sub = arr[cy0:cy1, cx0:cx1][circ].astype(float)
        mr, mg, mb = sub[:, 0], sub[:, 1], sub[:, 2]

    total = center.shape[0]
    green_fraction = len(mr) / total if total else 0.0

    # Per-pixel Lab + HSV
    lab = np.array([rgb_to_lab(mr[i], mg[i], mb[i]) for i in range(len(mr))])
    labL, labA, labB = lab[:, 0], lab[:, 1], lab[:, 2]
    chroma = np.sqrt(labA ** 2 + labB ** 2)
    hues, sats = [], []
    for i in range(len(mr)):
        h, s, _ = rgb_to_hsv(mr[i], mg[i], mb[i])
        hues.append(h); sats.append(s)
    hues = np.array(hues); sats = np.array(sats)

    Lmean, amean, bmean = rgb_to_lab(mr.mean(), mg.mean(), mb.mean()) if len(mr) else IDEAL_LAB
    chroma_mean = math.sqrt(amean ** 2 + bmean ** 2)
    dE_ab = math.sqrt((amean - IDEAL_LAB[1]) ** 2 + (bmean - IDEAL_LAB[2]) ** 2)
    dE_Lab = math.sqrt((Lmean - IDEAL_LAB[0]) ** 2 + (amean - IDEAL_LAB[1]) ** 2 + (bmean - IDEAL_LAB[2]) ** 2)
    hue_mean = circular_hue_mean(hues) if len(hues) else IDEAL_HUE
    hue_dist = circular_hue_distance(hue_mean, IDEAL_HUE)
    gr = mg - mr
    green_dom_mean = float(gr.mean()) if len(gr) else 0.0
    green_dom_std = float(gr.std()) if len(gr) else 0.0

    features = np.array([
        Lmean, amean, bmean, float(labA.std()) if len(labA) else 0.0,
        chroma_mean, float(chroma.std()) if len(chroma) else 0.0,
        dE_ab, dE_Lab,
        hue_dist, float(sats.mean()) if len(sats) else 0.0, float(sats.std()) if len(sats) else 0.0,
        green_fraction, green_dom_mean, green_dom_std,
    ], dtype=np.float32)
    return features


# ---------------------------------------------------------------------------
# Self-test — synthetic patches, mirrors the in-browser sanity harness
# ---------------------------------------------------------------------------
SANITY_CASES = [
    ("Ideal 5/5",        (100, 165, 60),  5),
    ("Typical 1/5",      (170, 175, 110), 1),
    ("Ceremonial dark",  (60, 140, 50),   5),
    ("Vivid ceremonial", (80, 175, 50),   5),
    ("Mid cafe green",   (120, 160, 90),  3),
    ("Pale watery",      (200, 210, 150), 1),
    ("Brown/oxidized",   (130, 110, 70),  1),
    ("White foam",       (235, 232, 225), 1),
    ("Cup beige",        (190, 180, 150), 1),
]


def perceptual_score(features):
    """Mirror of perceptualScore() in index.html."""
    Lmean, amean, bmean, a_std, chroma_mean, chroma_std, dE_ab, dE_Lab, \
        hue_dist, sat_mean, sat_std, green_frac, green_dom, green_dom_std = features
    def clamp(v, lo=1.0, hi=5.0):
        return max(lo, min(hi, v))
    sDom = clamp(1 + (green_dom / 70) * 4)
    sA = clamp(1 + ((-amean) - 3) / (35 - 3) * 4)
    sChroma = clamp(1 + ((chroma_mean - 15) / (55 - 15)) * 4)
    sHue = clamp(5 - ((hue_dist - 10) / (45 - 10)) * 4)
    sSat = clamp(1 + ((sat_mean - 0.30) / (0.62 - 0.30)) * 4)
    sDE = clamp(5 - max(0, dE_ab - 20) / 30 * 4)
    raw = sDom * 0.30 + sA * 0.18 + sChroma * 0.14 + sHue * 0.14 + sSat * 0.14 + sDE * 0.10
    if green_frac < 0.25:
        w = (0.25 - green_frac) / 0.25
        raw = raw * (1 - w * 0.3) + 3 * (w * 0.3)
    shaped = 1 + (max(0, raw - 1) / 4) ** 1.5 * 4
    return clamp(shaped), dict(sDom=sDom, sA=sA, sChroma=sChroma, sHue=sHue, sSat=sSat, sDE=sDE)


def selftest():
    print("Self-test: synthetic patches (should match the in-browser sanity table).")
    print("-" * 70)
    print(f"{'case':22} {'rgb':16} {'score':>6}  {'expect':>6}  {'G-R':>5}")
    print("-" * 70)
    for name, rgb, expect in SANITY_CASES:
        img = Image.new("RGB", (400, 400), rgb)
        feats = extract_features(img)
        score, _ = perceptual_score(feats)
        gr = feats[12]
        print(f"{name:22} {str(rgb):16} {score:6.2f}  {expect:6d}  {gr:5.0f}")
    print("-" * 70)
    print("Ideal Lab =", [round(v, 1) for v in IDEAL_LAB], "| hue =", round(IDEAL_HUE, 1))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def load_dataset(csv_path, images_dir):
    X, y, names = [], [], []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            path = row["image_path"]
            if not os.path.isabs(path):
                path = os.path.join(images_dir or "", path)
            try:
                img = Image.open(path)
            except Exception as e:
                print(f"  ! skip {path}: {e}", file=sys.stderr)
                continue
            X.append(extract_features(img))
            y.append(float(row["rating"]))
            names.append(path)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), names


def train(X, y):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVR
    from sklearn.model_selection import cross_val_predict

    pipe = make_pipeline(StandardScaler(), SVR(kernel="rbf", C=4.0, gamma="scale", epsilon=0.2))
    yp = cross_val_predict(pipe, X, y, cv=min(5, len(y)))
    mae = float(np.mean(np.abs(yp - y)))
    within1 = float(np.mean(np.abs(yp - y) <= 1.0))
    exact = float(np.mean(np.abs(yp - y) < 0.5))
    print(f"Cross-validation: MAE={mae:.3f}  exact(±0.5)={exact*100:.0f}%  within1={within1*100:.0f}%")
    pipe.fit(X, y)
    return pipe


def export_onnx(pipe, out_dir):
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    os.makedirs(out_dir, exist_ok=True)
    initial_type = [("float_input", FloatTensorType([1, 14]))]
    onx = convert_sklearn(pipe, initial_types=initial_type, target_opset=15,
                         options={id(pipe): {"zipmap": False}})
    onnx_path = os.path.join(out_dir, "matcha_svr.onnx")
    with open(onnx_path, "wb") as f:
        f.write(onx.SerializeToString())
    b64 = base64.b64encode(onx.SerializeToString()).decode("ascii")
    scaler = pipe.named_steps["standardscaler"]
    mean = list(map(float, scaler.mean_))
    scale = list(map(float, scaler.scale_))
    return onnx_path, b64, mean, scale


def emit_artifacts(onnx_path, b64, mean, scale):
    print("\n" + "=" * 70)
    print("ARTIFACTS — paste these into index.html")
    print("=" * 70)
    print(f"// ONNX written to {onnx_path} ({len(b64)} chars base64)")
    print(f"const MODEL_B64 = `{b64}`;")
    print(f"\nconst SCALER_MEAN = [{','.join(repr(m) for m in mean)}];")
    print(f"const SCALER_SCALE = [{','.join(repr(s) for s in scale)}];")
    print("=" * 70)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", help="CSV with columns: image_path,rating")
    ap.add_argument("--images_dir", default="", help="base dir for relative image_path")
    ap.add_argument("--out", default="onnx", help="directory to write matcha_svr.onnx")
    ap.add_argument("--selftest", action="store_true", help="score synthetic patches and exit")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    if not args.csv:
        ap.error("--csv is required (or use --selftest)")

    print("Loading dataset...")
    X, y, names = load_dataset(args.csv, args.images_dir)
    if len(X) < 5:
        print(f"Only {len(X)} images loaded — too few for a stable SVR. "
              "Falling back to the transparent scorer is recommended.", file=sys.stderr)
    print(f"Loaded {len(X)} images.")

    print("Training (with 5-fold CV)...")
    pipe = train(X, y)

    print("Exporting ONNX + scaler constants...")
    onnx_path, b64, mean, scale = export_onnx(pipe, args.out)
    emit_artifacts(onnx_path, b64, mean, scale)


if __name__ == "__main__":
    main()

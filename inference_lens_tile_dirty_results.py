import argparse
import csv
import glob
import math
import os
from collections import defaultdict
from datetime import datetime
from time import time
from urllib.parse import unquote, urlparse

import cv2
import numpy as np
from tqdm import tqdm

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
SCORE_COLUMNS = ['branch_score', 'dust_score', 'hair_score']
RESULT_COLUMNS = SCORE_COLUMNS + [
    'branch_metric_mean', 'branch_metric_p95', 'temporal_std_mean', 'temporal_std_p10',
    'highfreq_mean', 'entropy_mean', 'contrast_mean',
    'dust_shape_response', 'hair_shape_response', 'dust_contour_response', 'hair_contour_response',
    'hair_coherence_response', 'unit_width', 'unit_height', 'score_tile_count', 'valid_frames'
]


def normalize_input_path(path):
    path = path.strip()
    if path.startswith('file://'):
        parsed = urlparse(path)
        if parsed.netloc not in ('', 'localhost'):
            raise ValueError(f'Unsupported non-local file URI: {path}')
        path = unquote(parsed.path)
    return path


def parse_scene_marker(line):
    return line[3:-1].strip() if line.startswith('#-[') and line.endswith(']') else None


def read_paths_from_txt(txt_path):
    txt_dir = os.path.dirname(os.path.abspath(txt_path))
    paths, scenes, scene = [], [], None
    with open(txt_path, 'r', encoding='utf-8') as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            marker = parse_scene_marker(line)
            if marker is not None:
                scene = marker
                continue
            if line.startswith('#'):
                continue
            line = normalize_input_path(line)
            if os.path.isabs(line):
                p = line
            elif os.path.exists(line):
                p = line
            else:
                p = os.path.join(txt_dir, line)
            paths.append(p); scenes.append(scene)
    return paths, scenes


def is_image_file(path):
    return os.path.splitext(path)[1].lower() in IMAGE_EXTENSIONS


def natural_key(path):
    import re
    name = os.path.basename(path)
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r'(\d+)', name)]


def timestamp_key(path):
    """Sort by camera timestamp encoded in file name.

    Expected data names look like ``00_9803820953.jpg``: the numeric part
    before ``_`` is minute, and the numeric part after ``_`` is the timestamp
    inside that minute. Therefore the correct temporal key is
    ``(minute, intra_minute_timestamp)``. For non-standard names, fall back to
    all numeric groups and then natural_key.
    """
    import re
    stem = os.path.splitext(os.path.basename(path))[0]
    m = re.match(r'^(\d+)_(\d+)$', stem)
    if m:
        return (0, int(m.group(1)), int(m.group(2)), natural_key(path))
    nums = re.findall(r'\d+', stem)
    if nums:
        return (1, tuple(int(x) for x in nums), natural_key(path))
    return (2, natural_key(path))


def get_input_paths(target, target_txt=None):
    if target_txt:
        return read_paths_from_txt(target_txt)
    if os.path.isfile(target):
        return [target], [None]
    paths = sorted(glob.glob(os.path.join(target, '*')), key=natural_key)
    paths = [p for p in paths if is_image_file(p)]
    return paths, [None] * len(paths)


def list_images_in_same_dir(img_path):
    d = os.path.dirname(os.path.abspath(img_path))
    paths = [os.path.join(d, n) for n in os.listdir(d)
             if is_image_file(n) and os.path.isfile(os.path.join(d, n))]
    return sorted(paths, key=timestamp_key)


def sample_even_span_refs(imgs, seed_real, sample_count):
    seed_abs = os.path.abspath(seed_real)
    if sample_count <= 1:
        return []
    if sample_count >= len(imgs):
        return [x for x in imgs if os.path.abspath(x) != seed_abs]
    idxs = np.rint(np.linspace(0, len(imgs) - 1, sample_count)).astype(np.int64)
    picked = sorted(set(int(max(0, min(len(imgs) - 1, i))) for i in idxs))
    seed_idx = imgs.index(seed_real)
    if seed_idx not in picked:
        rp = min(range(len(picked)), key=lambda j: abs(picked[j] - seed_idx))
        picked[rp] = seed_idx
    return [imgs[i] for i in sorted(set(picked)) if os.path.abspath(imgs[i]) != seed_abs]


def build_seed_sequences(seed_paths, window, sequence_mode, sample_count):
    win = max(0, int(window))
    effective_seed_paths, seqs, all_paths, adjustments, invalids, weak_seeds = [], [], set(), [], [], []
    for seed_path in seed_paths:
        seed_abs = os.path.abspath(seed_path)
        imgs = list_images_in_same_dir(seed_abs)
        amap = {os.path.abspath(p): p for p in imgs}
        if seed_abs not in amap:
            raise FileNotFoundError(f'Seed image does not exist in its directory list: {seed_path}')
        original_seed = amap[seed_abs]
        seed_real = original_seed
        idx = imgs.index(seed_real)
        if sequence_mode == 'even_span':
            refs = sample_even_span_refs(imgs, seed_real, sample_count)
        elif sequence_mode in ('previous', 'previous_next'):
            if idx < win and len(imgs) > 1:
                new_idx = min(win, len(imgs) - 1)
                seed_real = imgs[new_idx]
                adjustments.append((original_seed, seed_real, idx, new_idx))
                idx = new_idx
            refs = imgs[max(0, idx - win):idx]
        else:
            raise ValueError(f'Unsupported sequence_mode: {sequence_mode}')
        if len(refs) < 1:
            invalids.append((original_seed, seed_real, 'no_reference_frame'))
        elif sequence_mode in ('previous', 'previous_next') and win > 0 and len(refs) < win:
            weak_seeds.append((original_seed, seed_real, len(refs), win))
        seq = [seed_real] + refs
        effective_seed_paths.append(seed_real); seqs.append(seq); all_paths.update(seq)
    return effective_seed_paths, seqs, sorted(all_paths, key=natural_key), adjustments, invalids, weak_seeds


def imread_image(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f'Failed to read image: {path}')
    return img.astype(np.float32)


def gray_from_bgr(img):
    return cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)


def ensure_odd(v, min_value=3):
    v = max(min_value, int(v))
    return v + 1 if v % 2 == 0 else v


def normalize_01(x, low=None, high=None):
    x = x.astype(np.float32)
    if low is None: low = float(np.nanmin(x))
    if high is None: high = float(np.nanmax(x))
    if high - low < 1e-6:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - low) / (high - low), 0, 1).astype(np.float32)


def aggregate_map_to_units(value_map, unit_width=16, unit_height=12, reduce='mean'):
    h, w = value_map.shape[:2]
    uw, uh = max(1, int(unit_width)), max(1, int(unit_height))
    rows, cols = int(math.ceil(h / float(uh))), int(math.ceil(w / float(uw)))
    unit = np.zeros((rows, cols), dtype=np.float32)
    for r in range(rows):
        for c in range(cols):
            patch = value_map[r*uh:min(h, (r+1)*uh), c*uw:min(w, (c+1)*uw)]
            if reduce == 'max': unit[r, c] = float(patch.max())
            elif reduce == 'median': unit[r, c] = float(np.median(patch))
            else: unit[r, c] = float(patch.mean())
    expd = cv2.resize(unit, (w, h), interpolation=cv2.INTER_NEAREST).astype(np.float32)
    return unit, expd


def top_percent_mean(unit_map, percent, largest=True):
    flat = np.sort(unit_map.reshape(-1).astype(np.float32))
    if flat.size == 0:
        return float('nan'), 0
    k = max(1, int(math.ceil(flat.size * max(0, min(100, percent)) / 100.0)))
    vals = flat[-k:] if largest else flat[:k]
    return float(vals.mean()), k


def adaptive_unit_size(height, width):
    """Choose tile/unit size from image resolution without resizing image.

    Target about 40 x 40 tiles for ordinary images. This keeps regions larger
    than pixels but still small enough to see dust/hair local responses.
    """
    unit_w = max(1, int(round(width / 40.0)))
    unit_h = max(1, int(round(height / 40.0)))
    return unit_w, unit_h


def adaptive_top_mean(unit_map, largest=True, percent=2.0, min_tiles=8):
    """Internal score pooling: average the most suspicious tiles.

    The old --top_unit_percent CLI parameter is intentionally removed. Scoring
    still needs a pooling rule, but visualization always shows full maps.
    """
    flat = np.sort(unit_map.reshape(-1).astype(np.float32))
    if flat.size == 0:
        return float('nan'), 0
    k = max(1, min(flat.size, max(int(min_tiles), int(math.ceil(flat.size * percent / 100.0)))))
    vals = flat[-k:] if largest else flat[:k]
    return float(vals.mean()), k


def local_entropy(gray_u8, ksize):
    # Tile/patch entropy implementation: compute entropy per unit later for speed.
    raise NotImplementedError


def tile_entropy_and_contrast(gray, unit_width, unit_height, hist_bins=32):
    h, w = gray.shape[:2]
    uw, uh = max(1, int(unit_width)), max(1, int(unit_height))
    rows, cols = int(math.ceil(h / float(uh))), int(math.ceil(w / float(uw)))
    ent = np.zeros((rows, cols), dtype=np.float32)
    con = np.zeros((rows, cols), dtype=np.float32)
    for r in range(rows):
        for c in range(cols):
            patch = gray[r*uh:min(h, (r+1)*uh), c*uw:min(w, (c+1)*uw)].astype(np.float32)
            hist = cv2.calcHist([patch.astype(np.uint8)], [0], None, [hist_bins], [0, 256]).reshape(-1)
            prob = hist / max(1.0, float(hist.sum()))
            prob = prob[prob > 0]
            ent[r, c] = float(-(prob * np.log2(prob)).sum() / math.log2(hist_bins))
            con[r, c] = float(patch.std() / 64.0)
    ent = np.clip(ent, 0, 1); con = np.clip(con, 0, 1)
    ent_map = cv2.resize(ent, (w, h), interpolation=cv2.INTER_NEAREST)
    con_map = cv2.resize(con, (w, h), interpolation=cv2.INTER_NEAREST)
    return ent, con, ent_map.astype(np.float32), con_map.astype(np.float32)


def dust_morphology(gray, args):
    k = ensure_odd(args.dust_spot_ksize)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    blackhat = cv2.morphologyEx(gray.astype(np.uint8), cv2.MORPH_BLACKHAT, kernel).astype(np.float32)
    tophat = cv2.morphologyEx(gray.astype(np.uint8), cv2.MORPH_TOPHAT, kernel).astype(np.float32)
    spot = np.maximum(blackhat, args.dust_bright_weight * tophat)
    dog = np.abs(cv2.GaussianBlur(gray, (ensure_odd(args.dust_log_ksize), ensure_odd(args.dust_log_ksize)), 0) -
                 cv2.GaussianBlur(gray, (ensure_odd(args.dust_bg_ksize, 7), ensure_odd(args.dust_bg_ksize, 7)), 0))
    return np.maximum(normalize_01(spot, 0, max(1, np.percentile(spot, 98))),
                      normalize_01(dog, 0, max(1, np.percentile(dog, 98))))


def make_line_kernels(length=25, width=3):
    length = ensure_odd(length, 5); width = max(1, int(width))
    ks = []
    kh = np.ones((width, length), dtype=np.uint8); ks.append(kh)
    kv = np.ones((length, width), dtype=np.uint8); ks.append(kv)
    kd1 = np.zeros((length, length), dtype=np.uint8); cv2.line(kd1, (0, 0), (length-1, length-1), 1, width); ks.append(kd1)
    kd2 = np.zeros((length, length), dtype=np.uint8); cv2.line(kd2, (0, length-1), (length-1, 0), 1, width); ks.append(kd2)
    return ks


def hair_morphology(gray, args):
    rs = []
    for k in make_line_kernels(args.hair_line_length, args.hair_line_width):
        rs.append(cv2.morphologyEx(gray.astype(np.uint8), cv2.MORPH_BLACKHAT, k).astype(np.float32))
    line = np.max(np.stack(rs, axis=0), axis=0)
    line = normalize_01(line, 0, max(1, np.percentile(line, 98)))
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    jxx = cv2.GaussianBlur(gx * gx, (15, 15), 0); jyy = cv2.GaussianBlur(gy * gy, (15, 15), 0); jxy = cv2.GaussianBlur(gx * gy, (15, 15), 0)
    coherence = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / (jxx + jyy + 1e-6)
    return np.sqrt(np.clip(line * (0.4 + 0.6 * np.clip(coherence, 0, 1)), 0, 1)).astype(np.float32), np.clip(coherence, 0, 1).astype(np.float32)



def make_oriented_line_kernels(length=31, width=3, angle_step=15):
    """Create thin line kernels for many orientations."""
    length = ensure_odd(length, 7)
    width = max(1, int(width))
    center = length // 2
    radius = center
    kernels = []
    for angle in range(0, 180, max(1, int(angle_step))):
        rad = math.radians(angle)
        dx = int(round(math.cos(rad) * radius))
        dy = int(round(math.sin(rad) * radius))
        k = np.zeros((length, length), dtype=np.uint8)
        cv2.line(k, (center - dx, center - dy), (center + dx, center + dy), 1, width, cv2.LINE_AA)
        kernels.append(k)
    return kernels


def threshold_response_map(resp, min_percentile=94.0):
    """Threshold a 0..1 response map while keeping weak but local structures visible."""
    r = np.clip(resp, 0, 1).astype(np.float32)
    u8 = np.clip(r * 255, 0, 255).astype(np.uint8)
    nz = r[r > 1e-6]
    if nz.size == 0:
        return np.zeros_like(u8)
    perc = float(np.percentile(nz, min_percentile))
    _, otsu = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    perc_mask = (r >= max(perc, 0.05)).astype(np.uint8) * 255
    return cv2.bitwise_or(otsu, perc_mask)


def dust_contour_prior(dust_response, temporal_static, unit_width, unit_height):
    """Small, soft, near-round stable blobs: dust/mud particle prior."""
    h, w = dust_response.shape[:2]
    cand = np.clip(dust_response * (0.35 + 0.65 * temporal_static), 0, 1)
    mask = threshold_response_map(cand, 94.0)
    open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_k)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    prior = np.zeros((h, w), dtype=np.float32)
    min_area = max(3.0, 0.08 * unit_width * unit_height)
    max_area = max(min_area + 1.0, 10.0 * unit_width * unit_height)
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < min_area or area > max_area:
            continue
        per = float(cv2.arcLength(cnt, True))
        if per <= 1e-6:
            continue
        rect = cv2.minAreaRect(cnt)
        rw, rh = rect[1]
        if rw <= 1e-3 or rh <= 1e-3:
            continue
        aspect = max(rw, rh) / (min(rw, rh) + 1e-6)
        if aspect > 4.0:
            continue
        circularity = np.clip(4.0 * math.pi * area / (per * per + 1e-6), 0, 1)
        area_prior = np.exp(-area / max(max_area, 1.0))
        ar_prior = np.exp(-(aspect - 1.0) / 1.6)
        tmp = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(tmp, [cnt], -1, 255, -1)
        local_resp = float(cand[tmp > 0].mean()) if np.any(tmp > 0) else 0.0
        score = float(np.clip(local_resp * (0.35 + 0.45 * circularity + 0.20 * ar_prior) * area_prior, 0, 1))
        prior[tmp > 0] = np.maximum(prior[tmp > 0], score)
    return prior, (prior > 0).astype(np.float32)


def hair_contour_prior(hair_response, coherence, temporal_static, unit_width, unit_height):
    """Thin elongated stable contours: hair/fiber prior."""
    h, w = hair_response.shape[:2]
    cand = np.clip(hair_response * (0.35 + 0.65 * coherence) * (0.35 + 0.65 * temporal_static), 0, 1)
    mask = threshold_response_map(cand, 93.0)
    close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_k)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    prior = np.zeros((h, w), dtype=np.float32)
    min_area = max(4.0, 0.12 * unit_width * unit_height)
    max_area = max(min_area + 1.0, 30.0 * unit_width * unit_height)
    min_len = max(8.0, 1.2 * max(unit_width, unit_height))
    max_width = max(3.0, 2.5 * min(unit_width, unit_height))
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < min_area or area > max_area:
            continue
        rect = cv2.minAreaRect(cnt)
        rw, rh = rect[1]
        if rw <= 1e-3 or rh <= 1e-3:
            continue
        length = max(rw, rh)
        width = min(rw, rh)
        aspect = length / (width + 1e-6)
        if aspect < 4.0 or length < min_len or width > max_width:
            continue
        tmp = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(tmp, [cnt], -1, 255, -1)
        local_resp = float(cand[tmp > 0].mean()) if np.any(tmp > 0) else 0.0
        aspect_prior = np.clip((aspect - 3.0) / 9.0, 0, 1)
        width_prior = np.exp(-width / max(max_width, 1.0))
        score = float(np.clip(local_resp * (0.45 + 0.35 * aspect_prior + 0.20 * width_prior), 0, 1))
        prior[tmp > 0] = np.maximum(prior[tmp > 0], score)
    return prior, (prior > 0).astype(np.float32)


def advanced_dirty_morphology(gray, temporal_static, args, unit_width, unit_height):
    """Complex-background dust/hair morphology gated by temporal invariance.

    Dust: multi-scale soft blob + small near-round contour + anti-line prior.
    Hair: multi-orientation dark/bright line + structure-tensor coherence + elongated contour.
    """
    gray_u8 = np.clip(gray, 0, 255).astype(np.uint8)

    # Dust: multi-scale dark/bright soft spots and DoG spots.
    dust_responses = []
    for k in [7, 11, 17, ensure_odd(args.dust_spot_ksize)]:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ensure_odd(k), ensure_odd(k)))
        blackhat = cv2.morphologyEx(gray_u8, cv2.MORPH_BLACKHAT, kernel).astype(np.float32)
        tophat = cv2.morphologyEx(gray_u8, cv2.MORPH_TOPHAT, kernel).astype(np.float32)
        spot = np.maximum(blackhat, args.dust_bright_weight * tophat)
        dust_responses.append(normalize_01(spot, 0, max(1, np.percentile(spot, 98))))
    for s1, s2 in [(1.2, 2.5), (2.0, 4.5), (3.0, 7.0)]:
        dog = np.abs(cv2.GaussianBlur(gray, (0, 0), s1) - cv2.GaussianBlur(gray, (0, 0), s2))
        dust_responses.append(normalize_01(dog, 0, max(1, np.percentile(dog, 98))))
    dust_blob = np.max(np.stack(dust_responses, axis=0), axis=0).astype(np.float32)

    # Hair: multi-orientation line response, both dark and bright lines.
    line_rs = []
    for k in make_oriented_line_kernels(args.hair_line_length, args.hair_line_width, 15):
        dark = cv2.morphologyEx(gray_u8, cv2.MORPH_BLACKHAT, k).astype(np.float32)
        bright = cv2.morphologyEx(gray_u8, cv2.MORPH_TOPHAT, k).astype(np.float32)
        line_rs.append(np.maximum(dark, 0.65 * bright))
    line_raw = np.max(np.stack(line_rs, axis=0), axis=0)
    hair_line = normalize_01(line_raw, 0, max(1, np.percentile(line_raw, 98)))

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    jxx = cv2.GaussianBlur(gx * gx, (15, 15), 0)
    jyy = cv2.GaussianBlur(gy * gy, (15, 15), 0)
    jxy = cv2.GaussianBlur(gx * gy, (15, 15), 0)
    coherence = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / (jxx + jyy + 1e-6)
    coherence = np.clip(coherence, 0, 1).astype(np.float32)

    dust_contour, dust_contour_mask = dust_contour_prior(dust_blob, temporal_static, unit_width, unit_height)
    hair_contour, hair_contour_mask = hair_contour_prior(hair_line, coherence, temporal_static, unit_width, unit_height)

    anti_line = 1.0 - normalize_01(hair_line * coherence, 0, max(0.05, np.percentile(hair_line * coherence, 98)))
    dust_shape = np.clip((0.60 * dust_blob + 0.40 * dust_contour) * (0.55 + 0.45 * anti_line), 0, 1)
    hair_shape = np.clip(0.55 * hair_line + 0.25 * coherence + 0.20 * hair_contour, 0, 1)
    dust_prior = np.clip(temporal_static * dust_shape, 0, 1)
    hair_prior = np.clip(temporal_static * hair_shape, 0, 1)

    return {
        'dust_shape': dust_shape.astype(np.float32),
        'hair_shape': hair_shape.astype(np.float32),
        'dust_prior': dust_prior.astype(np.float32),
        'hair_prior': hair_prior.astype(np.float32),
        'dust_blob': dust_blob.astype(np.float32),
        'dust_contour': dust_contour.astype(np.float32),
        'dust_contour_mask': dust_contour_mask.astype(np.float32),
        'hair_line': hair_line.astype(np.float32),
        'hair_coherence': coherence.astype(np.float32),
        'hair_contour': hair_contour.astype(np.float32),
        'hair_contour_mask': hair_contour_mask.astype(np.float32),
    }


def particle_size_analysis_prior(gray, temporal_static, args, unit_width, unit_height):
    """SandDet-style particle segmentation branch.

    This is an intentionally simple implementation of the soil-particle paper
    pipeline for lens-dirt experiments:

      Gaussian/CLAHE pre-process -> adaptive threshold -> contour extraction
      -> minAreaRect/shape filtering -> temporal-invariance gating.

    It will definitely fire on texture-rich background, so the output should be
    treated as an experiment/visual diagnostic rather than a final detector.
    """
    h, w = gray.shape[:2]
    gray_u8 = np.clip(gray, 0, 255).astype(np.uint8)
    blur_k = ensure_odd(args.particle_blur_ksize)
    block = ensure_odd(args.particle_block_size, 3)

    # Mild illumination normalization improves adaptiveThreshold stability.
    clahe = cv2.createCLAHE(clipLimit=args.particle_clahe_clip, tileGridSize=(8, 8))
    norm = clahe.apply(gray_u8)
    blur = cv2.GaussianBlur(norm, (blur_k, blur_k), 0)

    # Dark and bright local particles are both possible on lens dirt. Keep both
    # masks but score them later by contour geometry.
    dark_mask = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
        block, args.particle_adaptive_c)
    bright_mask = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
        block, -args.particle_adaptive_c)
    mask = cv2.bitwise_or(dark_mask, bright_mask)

    # Remove isolated threshold noise; close tiny gaps in a possible droplet or
    # hair/fiber contour. Temporal gate is deliberately soft for visualization.
    open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_k)
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_k)
    static_mask = (temporal_static >= args.particle_static_threshold).astype(np.uint8) * 255
    gated_mask = cv2.bitwise_and(mask, static_mask)

    contours, _ = cv2.findContours(gated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dust_prior = np.zeros((h, w), dtype=np.float32)
    hair_prior = np.zeros((h, w), dtype=np.float32)
    contour_prior = np.zeros((h, w), dtype=np.float32)

    unit_area = max(1.0, float(unit_width * unit_height))
    min_area = max(2.0, args.particle_min_area_units * unit_area)
    max_area = max(min_area + 1.0, args.particle_max_area_units * unit_area)

    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < min_area or area > max_area:
            continue
        per = float(cv2.arcLength(cnt, True))
        if per <= 1e-6:
            continue
        rect = cv2.minAreaRect(cnt)
        rw, rh = rect[1]
        if rw <= 1e-3 or rh <= 1e-3:
            continue
        aspect = max(rw, rh) / (min(rw, rh) + 1e-6)
        circularity = float(np.clip(4.0 * math.pi * area / (per * per + 1e-6), 0, 1))
        extent = area / max(1.0, rw * rh)

        tmp = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(tmp, [cnt], -1, 255, -1)
        local_static = float(temporal_static[tmp > 0].mean()) if np.any(tmp > 0) else 0.0

        # Dust/mud spot: small/medium near-round particle.
        dust_shape = circularity * math.exp(-(aspect - 1.0) / 1.8) * np.clip(extent / 0.55, 0, 1)
        dust_size = math.exp(-area / max(max_area, 1.0))
        dust_score = float(np.clip(local_static * dust_shape * (0.45 + 0.55 * dust_size), 0, 1))

        # Hair/fiber: elongated contour found by the same particle segmentation.
        hair_shape = np.clip((aspect - 3.0) / 9.0, 0, 1) * np.clip(1.0 - circularity, 0, 1)
        hair_score = float(np.clip(local_static * hair_shape, 0, 1))

        contour_score = max(dust_score, hair_score)
        contour_prior[tmp > 0] = np.maximum(contour_prior[tmp > 0], contour_score)
        dust_prior[tmp > 0] = np.maximum(dust_prior[tmp > 0], dust_score)
        hair_prior[tmp > 0] = np.maximum(hair_prior[tmp > 0], hair_score)

    return {
        'raw_mask': (mask.astype(np.float32) / 255.0),
        'gated_mask': (gated_mask.astype(np.float32) / 255.0),
        'contour_prior': contour_prior.astype(np.float32),
        'dust_prior': dust_prior.astype(np.float32),
        'hair_prior': hair_prior.astype(np.float32),
    }

def calculate_metrics(frames, args):
    """Calculate one selected branch only.

    --branch temporal  : suspicious = low temporal std / static in image coordinates.
    --branch frequency : suspicious = local high-frequency energy loss.
    --branch entropy   : suspicious = local low entropy + low contrast.
    --branch morphology: temporal invariance gated dust/hair morphology.
    --branch particle  : SandDet-style adaptive-threshold contour particles.

    Dust/hair scores are only branch_score multiplied by their corresponding
    morphology priors. This keeps branch effectiveness easy to judge.
    """
    start = time()
    seed = frames[0]
    gray = gray_from_bgr(seed)
    all_grays = [gray_from_bgr(f) for f in frames]
    h, w = gray.shape[:2]
    unit_width, unit_height = adaptive_unit_size(h, w)

    # 1) Temporal branch: same-position tile is almost unchanged across frames.
    if len(all_grays) >= 2:
        stack = np.stack(all_grays, axis=0).astype(np.float32)
        temporal_std = stack.std(axis=0)
        temporal_static = np.exp(-temporal_std / max(args.temporal_std_scale, 1e-6)).astype(np.float32)
    else:
        temporal_std = np.zeros_like(gray, dtype=np.float32)
        temporal_static = np.zeros_like(gray, dtype=np.float32)
    _, temporal_static_heat = aggregate_map_to_units(temporal_static, unit_width, unit_height)
    temporal_std_heat = normalize_01(temporal_std, 0, max(1, np.percentile(temporal_std, 95)))

    # 2) Frequency branch: local high-frequency energy is abnormally low.
    hp = np.abs(gray - cv2.GaussianBlur(gray, (ensure_odd(args.highpass_ksize), ensure_odd(args.highpass_ksize)), 0))
    uhf, _ = aggregate_map_to_units(hp, unit_width, unit_height)
    hf_ref = max(args.highfreq_ref, float(np.percentile(uhf, args.highfreq_ref_percentile)))
    highfreq_norm_unit = np.clip(uhf / max(hf_ref, 1e-6), 0, 1)
    highfreq_norm_map = cv2.resize(highfreq_norm_unit, (w, h), interpolation=cv2.INTER_NEAREST).astype(np.float32)
    low_highfreq = 1.0 - highfreq_norm_map

    # 3) Entropy/contrast branch: local information is monotonous.
    ent_unit, con_unit, entropy_map, contrast_map = tile_entropy_and_contrast(gray, unit_width, unit_height, args.entropy_bins)
    low_entropy = 1.0 - entropy_map
    low_contrast = 1.0 - contrast_map
    low_info = np.clip(args.entropy_weight * low_entropy + (1.0 - args.entropy_weight) * low_contrast, 0, 1).astype(np.float32)

    morph = advanced_dirty_morphology(gray, temporal_static, args, unit_width, unit_height)

    if args.branch == 'temporal':
        branch_raw = temporal_std_heat
        branch_raw_name = 'temporal_std'
        branch_map = temporal_static
        branch_heat_name = 'temporal_static'
        dust_shape = morph['dust_shape']
        hair_shape = morph['hair_shape']
        dust_map = np.clip(branch_map * (0.5 + 0.5 * dust_shape), 0, 1)
        hair_map = np.clip(branch_map * (0.5 + 0.5 * hair_shape), 0, 1)
    elif args.branch == 'frequency':
        branch_raw = highfreq_norm_map
        branch_raw_name = 'highfreq_energy'
        branch_map = low_highfreq
        branch_heat_name = 'low_highfreq'
        dust_shape = morph['dust_shape']
        hair_shape = morph['hair_shape']
        dust_map = np.clip(branch_map * (0.5 + 0.5 * dust_shape), 0, 1)
        hair_map = np.clip(branch_map * (0.5 + 0.5 * hair_shape), 0, 1)
    elif args.branch == 'entropy':
        branch_raw = entropy_map
        branch_raw_name = 'local_entropy'
        branch_map = low_info
        branch_heat_name = 'low_entropy_contrast'
        dust_shape = morph['dust_shape']
        hair_shape = morph['hair_shape']
        dust_map = np.clip(branch_map * (0.5 + 0.5 * dust_shape), 0, 1)
        hair_map = np.clip(branch_map * (0.5 + 0.5 * hair_shape), 0, 1)
    elif args.branch == 'morphology':
        branch_raw = temporal_static_heat
        branch_raw_name = 'temporal_static'
        branch_map = np.maximum(morph['dust_prior'], morph['hair_prior'])
        branch_heat_name = 'temporal_gated_morphology'
        dust_shape = morph['dust_shape']
        hair_shape = morph['hair_shape']
        dust_map = morph['dust_prior']
        hair_map = morph['hair_prior']
    elif args.branch == 'particle':
        particle = particle_size_analysis_prior(gray, temporal_static, args, unit_width, unit_height)
        branch_raw = particle['raw_mask']
        branch_raw_name = 'adaptive_threshold_mask'
        branch_map = particle['contour_prior']
        branch_heat_name = 'sanddet_temporal_gated_contours'
        dust_shape = particle['dust_prior']
        hair_shape = particle['hair_prior']
        dust_map = particle['dust_prior']
        hair_map = particle['hair_prior']
        morph = dict(morph)
        morph['dust_blob'] = particle['raw_mask']
        morph['dust_contour'] = particle['dust_prior']
        morph['hair_line'] = particle['gated_mask']
        morph['hair_contour'] = particle['hair_prior']
    else:
        raise ValueError(f'Unsupported branch: {args.branch}')

    branch_unit, branch_heat = aggregate_map_to_units(branch_map, unit_width, unit_height)
    dust_unit, dust_heat = aggregate_map_to_units(dust_map, unit_width, unit_height)
    hair_unit, hair_heat = aggregate_map_to_units(hair_map, unit_width, unit_height)

    branch_score, score_tile_count = adaptive_top_mean(branch_unit, True, percent=2.0, min_tiles=8)
    dust_score, _ = adaptive_top_mean(dust_unit, True, percent=2.0, min_tiles=8)
    hair_score, _ = adaptive_top_mean(hair_unit, True, percent=3.0, min_tiles=12)

    return {
        'branch_score': 100.0 * branch_score,
        'dust_score': 100.0 * dust_score,
        'hair_score': 100.0 * hair_score,
        'branch_metric_mean': float(branch_map.mean()),
        'branch_metric_p95': float(np.percentile(branch_map, 95)),
        'temporal_std_mean': float(temporal_std.mean()),
        'temporal_std_p10': float(np.percentile(temporal_std, 10)),
        'highfreq_mean': float(highfreq_norm_map.mean()),
        'entropy_mean': float(entropy_map.mean()),
        'contrast_mean': float(contrast_map.mean()),
        'dust_shape_response': float(dust_shape.mean()),
        'hair_shape_response': float(hair_shape.mean()),
        'dust_contour_response': float(morph['dust_contour'].mean()),
        'hair_contour_response': float(morph['hair_contour'].mean()),
        'hair_coherence_response': float(morph['hair_coherence'].mean()),
        'unit_width': float(unit_width),
        'unit_height': float(unit_height),
        'score_tile_count': float(score_tile_count),
        'valid_frames': len(frames),
        'elapsed_time': time() - start,
        'branch_raw_heat': branch_raw.astype(np.float32),
        'branch_raw_name': branch_raw_name,
        'branch_heat': branch_heat.astype(np.float32),
        'branch_heat_name': branch_heat_name,
        'temporal_std_heat': temporal_std_heat,
        'temporal_static_heat': temporal_static_heat,
        'highfreq_heat': highfreq_norm_map,
        'low_highfreq_heat': low_highfreq,
        'entropy_heat': entropy_map,
        'contrast_heat': contrast_map,
        'low_info_heat': low_info,
        'dust_shape_heat': dust_shape,
        'hair_shape_heat': hair_shape,
        'dust_blob_heat': morph['dust_blob'],
        'dust_contour_heat': morph['dust_contour'],
        'hair_line_heat': morph['hair_line'],
        'hair_coherence_heat': morph['hair_coherence'],
        'hair_contour_heat': morph['hair_contour'],
        'dust_heat': dust_heat,
        'hair_heat': hair_heat,
    }


def format_float(v):
    try: v = float(v)
    except Exception: return str(v)
    return 'nan' if math.isnan(v) else f'{v:.4f}'


def make_safe_name(text):
    import re
    return re.sub(r'[^0-9A-Za-z._-]+', '_', str(text or '').strip()).strip('_') or 'none'


def build_auto_txt_save_path(save_txt_dir, metric_name):
    os.makedirs(save_txt_dir, exist_ok=True)
    return os.path.join(save_txt_dir, f'{datetime.now().strftime("%Y%m%d_%H%M%S")}.{metric_name.replace(os.sep, "_")}.txt')


def build_vis_save_dir(save_txt_path):
    d = os.path.join(os.path.dirname(os.path.abspath(save_txt_path)), os.path.splitext(os.path.basename(save_txt_path))[0])
    os.makedirs(d, exist_ok=True); return d


def colorize_01(x):
    return cv2.applyColorMap(np.clip(x * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)


def overlay(base, heat, alpha):
    b = np.clip(base, 0, 255).astype(np.uint8)
    h = np.clip(heat, 0, 1).astype(np.float32)
    c = colorize_01(h); blend = cv2.addWeighted(b, 1-alpha, c, alpha, 0)
    out = b.copy(); out[h > 0] = blend[h > 0]; return out


def put_label(img, text, y=26):
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255,255,255), 1, cv2.LINE_AA)


def save_visualization(img_path, seed_img, result, out_dir, index, scene, alpha=0.55):
    base = np.clip(seed_img, 0, 255).astype(np.uint8)
    info = np.zeros_like(base)
    put_label(info, f'branch={result.get("branch", "")}: inspect selected metric only', 26)
    put_label(info, f'raw={result["branch_raw_name"]}; suspicious={result["branch_heat_name"]}', 54)
    put_label(info, f'frames={result["valid_frames"]} unit={int(result["unit_width"])}x{int(result["unit_height"])} tiles={int(result["score_tile_count"])}', 82)
    put_label(info, f'branch={format_float(result["branch_score"])} dust={format_float(result["dust_score"])} hair={format_float(result["hair_score"])}', 110)

    # Show the selected branch and morphology internals. The middle panels are
    # intentionally not hidden, so we can judge whether temporal invariance,
    # dust blob/contour, or hair line/coherence is really contributing.
    panels = [
        base.copy(),
        colorize_01(result['branch_raw_heat']),
        overlay(base, result['branch_heat'], alpha),
        colorize_01(result['dust_blob_heat']),
        colorize_01(result['dust_contour_heat']),
        colorize_01(result['dust_shape_heat']),
        overlay(base, result['dust_heat'], alpha),
        colorize_01(result['hair_line_heat']),
        colorize_01(result['hair_coherence_heat']),
        colorize_01(result['hair_contour_heat']),
        overlay(base, result['hair_heat'], alpha),
        info,
    ]
    labels = [
        'seed',
        result['branch_raw_name'],
        f'{result["branch_heat_name"]} score={format_float(result["branch_score"])}',
        'dust_blob_multi_scale',
        'dust_contour_round_blob',
        'dust_shape_prior',
        f'final_dust={format_float(result["dust_score"])}',
        'hair_line_multi_orient',
        'hair_structure_coherence',
        'hair_contour_elongated',
        f'final_hair={format_float(result["hair_score"])}',
        'info'
    ]
    labs = []
    for p, l in zip(panels, labels):
        p = p.copy(); put_label(p, l); labs.append(p)
    canvas = np.concatenate([
        np.concatenate(labs[:4], axis=1),
        np.concatenate(labs[4:8], axis=1),
        np.concatenate(labs[8:12], axis=1),
    ], axis=0)
    suffix = make_safe_name(result.get('branch', 'branch'))
    out = os.path.join(out_dir, f'{index:06d}_{make_safe_name(scene) if scene else "no_scene"}_{make_safe_name(os.path.splitext(os.path.basename(img_path))[0])}_{suffix}.jpg')
    cv2.imwrite(out, canvas); return out


def split_aliases(text):
    return [x.strip().lower() for x in str(text).split(',') if x.strip()]


def scene_matches(scene, aliases):
    s = (scene or '').lower()
    return any(a in s for a in aliases)


def scene_to_label(scene, args):
    if not scene: return None
    if scene.lower() == args.normal_scene.lower(): return 'normal'
    if scene_matches(scene, split_aliases(args.dust_positive_scenes)): return 'dust'
    if scene_matches(scene, split_aliases(args.hair_positive_scenes)): return 'hair'
    return scene.lower()


def best_threshold(samples, label, metric):
    vals = []
    for lab, _, res in samples:
        try: score = float(res[metric])
        except Exception: continue
        if math.isnan(score): continue
        vals.append((lab == label, score))
    if not vals or not any(x[0] for x in vals) or not any(not x[0] for x in vals):
        return float('inf')
    scores = sorted(set(x[1] for x in vals))
    cands = [scores[0]-1e-6] + [(scores[i]+scores[i+1])/2 for i in range(len(scores)-1)] + [scores[-1]+1e-6]
    best = None
    for th in cands:
        tp=tn=fp=fn=0
        for pos, score in vals:
            pred = score > th
            if pos and pred: tp += 1
            elif pos and not pred: fn += 1
            elif (not pos) and pred: fp += 1
            else: tn += 1
        pos_rec = tp/float(tp+fn) if tp+fn else 0
        neg_rec = tn/float(tn+fp) if tn+fp else 0
        key = (0.5*(pos_rec+neg_rec), (tp+tn)/float(tp+tn+fp+fn), th)
        if best is None or key > best[0]: best = (key, th)
    return best[1]


def multiclass_confusion_msgs(samples, args):
    labeled = [(scene_to_label(scene, args), path, res) for scene, path, res in samples if scene_to_label(scene, args)]
    msgs = ['Unified multi-class confusion matrix based on selected tile branch dust/hair scores.']
    if not labeled:
        msgs.append('  No labeled samples, skip.'); return msgs
    thresholds = {'dust': best_threshold(labeled, 'dust', 'dust_score'), 'hair': best_threshold(labeled, 'hair', 'hair_score')}
    msgs.append(f'  threshold[dust_score]={format_float(thresholds["dust"])}; threshold[hair_score]={format_float(thresholds["hair"])}')
    labels = ['normal', 'dust', 'hair']
    for lab, _, _ in labeled:
        if lab not in labels: labels.append(lab)
    mat = {r: {c: 0 for c in labels} for r in labels}
    total=correct=0
    for true, path, res in labeled:
        candidates = []
        for lab, metric in [('dust','dust_score'), ('hair','hair_score')]:
            th = thresholds[lab]; score = float(res[metric])
            if not math.isinf(th) and score > th:
                candidates.append(((score-th)/max(abs(th),1.0), lab))
        pred = max(candidates)[1] if candidates else 'normal'
        mat[true][pred] += 1; total += 1; correct += int(true == pred)
    w = max(12, max(len(x) for x in labels)+6)
    msgs.append(' ' * w + ''.join(f'{("pred_"+c):>{w}s}' for c in labels))
    for r in labels:
        msgs.append(f'{("true_"+r):>{w}s}' + ''.join(f'{mat[r].get(c,0):>{w}d}' for c in labels))
    recalls = [mat[r].get(r,0)/float(sum(mat[r].values())) for r in labels if sum(mat[r].values())]
    msgs.append(f'  total={total}, accuracy={format_float(correct/float(total) if total else 0)}, macro_recall={format_float(sum(recalls)/len(recalls) if recalls else 0)}')
    return msgs


def update_scene_stat(stats, scene, score, path):
    if not scene or math.isnan(float(score)): return
    st = stats[scene]; st['sum'] += score; st['count'] += 1
    if st['max_score'] is None or score > st['max_score']: st['max_score'], st['max_path'] = score, path
    if st['min_score'] is None or score < st['min_score']: st['min_score'], st['min_path'] = score, path


def main():
    parser = argparse.ArgumentParser(description='Tile-based dirty lens metrics: temporal stability + frequency + entropy/contrast.')
    parser.add_argument('-t', '--target', default=None)
    parser.add_argument('--target_txt', default=None)
    parser.add_argument('-m', '--metric_name', default='lens_tile_dirty')
    parser.add_argument('--metric_mode', default='NR')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--save_txt_dir', default=None)
    parser.add_argument('--save_file', default=None)
    parser.add_argument('--no_vis', action='store_true')
    parser.add_argument('--sequence_mode', default='previous', choices=['previous','previous_next','even_span'])
    parser.add_argument('--window', type=int, default=10)
    parser.add_argument('--sample_count', type=int, default=7)
    parser.add_argument('--branch', default='temporal', choices=['temporal','frequency','entropy','morphology','particle'], help='Run only one metric branch each time.')
    parser.add_argument('--temporal_std_scale', type=float, default=8.0)
    parser.add_argument('--highpass_ksize', type=int, default=9)
    parser.add_argument('--highfreq_ref', type=float, default=3.0)
    parser.add_argument('--highfreq_ref_percentile', type=float, default=75.0)
    parser.add_argument('--entropy_bins', type=int, default=32)
    parser.add_argument('--entropy_weight', type=float, default=0.55)
    parser.add_argument('--dust_spot_ksize', type=int, default=7)
    parser.add_argument('--dust_log_ksize', type=int, default=5)
    parser.add_argument('--dust_bg_ksize', type=int, default=31)
    parser.add_argument('--dust_bright_weight', type=float, default=0.6)
    parser.add_argument('--hair_line_length', type=int, default=25)
    parser.add_argument('--hair_line_width', type=int, default=3)
    parser.add_argument('--particle_blur_ksize', type=int, default=5)
    parser.add_argument('--particle_block_size', type=int, default=31)
    parser.add_argument('--particle_adaptive_c', type=float, default=3.0)
    parser.add_argument('--particle_clahe_clip', type=float, default=2.0)
    parser.add_argument('--particle_static_threshold', type=float, default=0.55)
    parser.add_argument('--particle_min_area_units', type=float, default=0.08)
    parser.add_argument('--particle_max_area_units', type=float, default=35.0)
    parser.add_argument('--score_metric', default='branch_score', choices=SCORE_COLUMNS)
    parser.add_argument('--normal_scene', default='normal')
    parser.add_argument('--dust_positive_scenes', default='dust,dirty,灰尘,尘,脏污,污渍')
    parser.add_argument('--hair_positive_scenes', default='hair,毛发,头发')
    parser.add_argument('--heatmap_alpha', type=float, default=0.55)
    args = parser.parse_args()
    if args.metric_mode != 'NR': raise ValueError('This script only supports NR mode.')
    if args.target is None and args.target_txt is None: raise ValueError('Please specify --target or --target_txt.')

    paths, scenes = get_input_paths(args.target, args.target_txt)
    paths, seqs, load_paths, seed_adjustments, invalid_seeds, weak_seeds = build_seed_sequences(paths, args.window, args.sequence_mode, args.sample_count)
    print('Loading seed images and tile-dirty sequence frames...')
    print(f'Seed images: {len(paths)}; images to load: {len(load_paths)}; sequence_mode={args.sequence_mode}; branch={args.branch}')
    if seed_adjustments:
        print(f'Auto-shifted early seeds: {len(seed_adjustments)}')
        for old_p, new_p, old_i, new_i in seed_adjustments[:10]:
            print(f'  seed_shift idx {old_i}->{new_i}: {old_p} -> {new_p}')
    if weak_seeds: print(f'WARNING: {len(weak_seeds)} weak seeds have fewer refs than requested.')
    if invalid_seeds: print(f'WARNING: {len(invalid_seeds)} invalid seeds have no refs.')
    cache = {os.path.abspath(p): imread_image(p) for p in tqdm(load_paths, unit='image')}

    save_txt_path=vis_dir=txt_f=None
    if args.save_txt_dir:
        save_txt_path = build_auto_txt_save_path(args.save_txt_dir, args.metric_name)
        if not args.no_vis: vis_dir = build_vis_save_dir(save_txt_path)
        txt_f = open(save_txt_path, 'w', encoding='utf-8')
        txt_f.write(f'metric_name: {args.metric_name}\nmetric_mode: NR\nscore_direction: larger_means_more_abnormal\n')
        txt_f.write('method: selected tile metric branch only: temporal low-variance OR frequency high-frequency loss OR entropy/contrast monotony OR temporal-gated morphology.\n')
        for k in vars(args): txt_f.write(f'{k}: {getattr(args,k)}\n')
        txt_f.write(f'seed_count: {len(paths)}\nloaded_image_count: {len(load_paths)}\nauto_shifted_seed_count: {len(seed_adjustments)}\nweak_seed_count: {len(weak_seeds)}\ninvalid_seed_count: {len(invalid_seeds)}\nvis_dir: {vis_dir}\ntime: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
        if seed_adjustments:
            txt_f.write('auto_shifted_seeds:\n')
            for old_p, new_p, old_i, new_i in seed_adjustments:
                txt_f.write(f'  idx {old_i}->{new_i}: {old_p} -> {new_p}\n')
        txt_f.write('\nscene\timage\t' + '\t'.join(RESULT_COLUMNS) + '\ttime\tvisualization\n')
    sf=writer=None
    if args.save_file:
        sf = open(args.save_file, 'w', newline=''); writer = csv.writer(sf); writer.writerow(['scene','image']+RESULT_COLUMNS+['time','visualization'])

    stats = {m: defaultdict(lambda: {'sum':0.0,'count':0,'max_score':None,'max_path':None,'min_score':None,'min_path':None}) for m in SCORE_COLUMNS}
    samples=[]; avg=0.0; cnt=0
    pbar = tqdm(total=len(paths), unit='image')
    for i, p in enumerate(paths):
        scene = scenes[i]
        frames = [cache[os.path.abspath(x)] for x in seqs[i]]
        res = calculate_metrics(frames, args)
        res['branch'] = args.branch
        vis = save_visualization(p, cache[os.path.abspath(p)], res, vis_dir, i, scene, args.heatmap_alpha) if vis_dir else ''
        score = float(res[args.score_metric]); avg += score; cnt += 1
        for m in SCORE_COLUMNS: update_scene_stat(stats[m], scene, float(res[m]), p)
        samples.append((scene, p, res))
        vals = [format_float(res[c]) for c in RESULT_COLUMNS]; elapsed = format_float(res['elapsed_time'])
        prefix = f'[{scene}] ' if scene else ''
        pbar.update(1); pbar.set_description(f'{prefix}{args.metric_name}/{args.score_metric}: {format_float(score)}')
        pbar.write(f'{prefix}{os.path.basename(p)} branch={args.branch} score={format_float(res["branch_score"])} dust={format_float(res["dust_score"])} hair={format_float(res["hair_score"])} frames={res["valid_frames"]} Time={elapsed}s')
        row = [scene or '', p] + vals + [elapsed, vis]
        if writer: writer.writerow(row)
        if txt_f: txt_f.write('\t'.join(map(str,row)) + '\n')
    pbar.close()
    msg = f'Average {args.metric_name}/{args.branch}/{args.score_metric} score of {args.target or args.target_txt} with {cnt}/{len(paths)} valid images is: {format_float(avg/cnt if cnt else float("nan"))}'
    print(msg)
    scene_msgs=[]
    print('Scene statistics by metric:')
    for m in SCORE_COLUMNS:
        scene_msgs.append(f'[{m}]'); print(f'[{m}]')
        for scene, st in stats[m].items():
            if st['count'] <= 0: continue
            line = f'  [{scene}] count={st["count"]}: avg={format_float(st["sum"]/st["count"])}, max={format_float(st["max_score"])} ({st["max_path"]}), min={format_float(st["min_score"])} ({st["min_path"]})'
            scene_msgs.append(line); print(line)
    cm_msgs = multiclass_confusion_msgs(samples, args)
    for x in cm_msgs: print(x)
    if txt_f:
        txt_f.write('\n' + msg + '\nScene statistics by metric:\n')
        for x in scene_msgs: txt_f.write(x + '\n')
        txt_f.write('\n')
        for x in cm_msgs: txt_f.write(x + '\n')
        txt_f.close()
    if sf: sf.close()
    if args.save_file: print(f'Done! CSV results are in {args.save_file}.')
    if save_txt_path: print(f'Done! TXT results are in {save_txt_path}.')
    if vis_dir: print(f'Done! Visualizations are in {vis_dir}.')


if __name__ == '__main__':
    main()

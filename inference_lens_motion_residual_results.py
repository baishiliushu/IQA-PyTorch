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
SCORE_COLUMNS = ['motion_residual_score', 'dust_score', 'hair_score', 'black_occlusion_score']
RESULT_COLUMNS = SCORE_COLUMNS + [
    'global_motion_mean', 'residual_mean', 'residual_p95', 'persistent_mean',
    'persistent_top_mean', 'dust_spot_response', 'hair_line_response',
    'hair_orientation_response', 'homography_ok_frames', 'valid_frames'
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


def sample_even_span_refs_around_seed(imgs, seed_real, sample_count):
    if not imgs:
        return []
    seed_abs = os.path.abspath(seed_real)
    if sample_count <= 1:
        return []
    if sample_count >= len(imgs):
        return [x for x in imgs if os.path.abspath(x) != seed_abs]
    idxs = np.rint(np.linspace(0, len(imgs) - 1, sample_count)).astype(np.int64)
    picked, seen = [], set()
    for i in idxs:
        i = int(max(0, min(len(imgs) - 1, i)))
        if i not in seen:
            picked.append(i); seen.add(i)
    seed_idx = imgs.index(seed_real)
    if seed_idx not in seen:
        rp = min(range(len(picked)), key=lambda j: abs(picked[j] - seed_idx))
        seen.discard(picked[rp]); picked[rp] = seed_idx; seen.add(seed_idx)
    while len(picked) < sample_count:
        best, dist = None, -1
        for i in range(len(imgs)):
            if i in seen:
                continue
            d = min(abs(i - p) for p in picked)
            if d > dist:
                best, dist = i, d
        picked.append(best); seen.add(best)
    return [imgs[i] for i in sorted(picked) if os.path.abspath(imgs[i]) != seed_abs]


def build_seed_sequences(seed_paths, window, sequence_mode, sample_count):
    """Build sequence as [effective_seed, reference_frames...].

    previous mode uses only N existing frames before seed. If seed is too early,
    the effective seed is automatically shifted to the earliest later frame that
    has a full previous window whenever possible.
    """
    win = max(0, int(window))
    effective_seed_paths, seqs, all_paths, missing, adjustments, invalids, weak_seeds = [], [], set(), [], [], [], []
    for seed_path in seed_paths:
        seed_abs = os.path.abspath(seed_path)
        imgs = list_images_in_same_dir(seed_abs)
        amap = {os.path.abspath(p): p for p in imgs}
        if seed_abs not in amap:
            missing.append(seed_path); effective_seed_paths.append(seed_path); seqs.append([]); continue
        original_seed = amap[seed_abs]
        seed_real = original_seed
        idx = imgs.index(seed_real)
        if sequence_mode == 'even_span':
            refs = sample_even_span_refs_around_seed(imgs, seed_real, sample_count)
        elif sequence_mode in ('previous', 'previous_next'):
            if idx < win and len(imgs) > 1:
                new_idx = min(win, len(imgs) - 1)
                seed_real = imgs[new_idx]
                adjustments.append((original_seed, seed_real, idx, new_idx))
                idx = new_idx
            refs = imgs[max(0, idx - win): idx]
        else:
            raise ValueError(f'Unsupported sequence_mode: {sequence_mode}')
        if len(refs) < 1:
            invalids.append((original_seed, seed_real, 'no_reference_frame'))
        elif sequence_mode in ('previous', 'previous_next') and win > 0 and len(refs) < win:
            weak_seeds.append((original_seed, seed_real, len(refs), win))
        seq = [seed_real] + refs
        effective_seed_paths.append(seed_real); seqs.append(seq); all_paths.update(seq)
    if missing:
        raise FileNotFoundError('Some seed images do not exist: ' + ', '.join(missing[:5]))
    return effective_seed_paths, seqs, sorted(all_paths, key=natural_key), adjustments, invalids, weak_seeds


def imread_image_resize(path, resize_width):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f'Failed to read image: {path}')
    if resize_width and resize_width > 0 and img.shape[1] != resize_width:
        scale = resize_width / float(img.shape[1])
        img = cv2.resize(img, (resize_width, max(1, int(round(img.shape[0] * scale)))), interpolation=cv2.INTER_AREA)
    return img.astype(np.float32)


def gray_from_bgr(img):
    return cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)


def ensure_odd(v, min_value=3):
    v = max(min_value, int(v))
    return v + 1 if v % 2 == 0 else v


def normalize_01(x, low=None, high=None):
    x = x.astype(np.float32)
    if low is None:
        low = float(np.nanmin(x))
    if high is None:
        high = float(np.nanmax(x))
    if high - low < 1e-6:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - low) / (high - low), 0, 1).astype(np.float32)


def aggregate_map_to_units(value_map, unit_width=4, unit_height=3, reduce='mean'):
    h, w = value_map.shape[:2]
    uw, uh = max(1, int(unit_width)), max(1, int(unit_height))
    rows, cols = int(math.ceil(h / float(uh))), int(math.ceil(w / float(uw)))
    unit = np.zeros((rows, cols), dtype=np.float32)
    for r in range(rows):
        for c in range(cols):
            patch = value_map[r*uh:min(h, (r+1)*uh), c*uw:min(w, (c+1)*uw)]
            unit[r, c] = float(patch.max() if reduce == 'max' else np.median(patch) if reduce == 'median' else patch.mean())
    expd = cv2.resize(unit, (w, h), interpolation=cv2.INTER_NEAREST).astype(np.float32)
    return unit, expd


def top_percent_mean(unit_map, percent, largest=True):
    flat = np.sort(unit_map.reshape(-1).astype(np.float32))
    if flat.size == 0:
        return float('nan'), 0
    k = max(1, int(math.ceil(flat.size * max(0, min(100, percent)) / 100.0)))
    vals = flat[-k:] if largest else flat[:k]
    return float(vals.mean()), k


def estimate_ref_to_seed_motion(seed_gray, ref_gray, args):
    """Estimate global background transform mapping ref -> seed.

    LK tracks seed points into ref, then estimates seed->ref. We invert it to
    warp the reference frame back to seed coordinates. Sparse/RANSAC estimation
    intentionally represents dominant background motion; lens-attached dirt does
    not control the transform.
    """
    seed_u8 = seed_gray.astype(np.uint8); ref_u8 = ref_gray.astype(np.uint8)
    pts0 = cv2.goodFeaturesToTrack(seed_u8, maxCorners=args.max_corners, qualityLevel=args.feature_quality,
                                   minDistance=args.feature_min_distance, blockSize=7)
    if pts0 is None or len(pts0) < args.min_matches:
        return None, 0, 0.0
    pts1, st, _ = cv2.calcOpticalFlowPyrLK(seed_u8, ref_u8, pts0, None,
                                           winSize=(21, 21), maxLevel=3,
                                           criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    if pts1 is None:
        return None, 0, 0.0
    good0 = pts0[st.reshape(-1) == 1].reshape(-1, 2)
    good1 = pts1[st.reshape(-1) == 1].reshape(-1, 2)
    if len(good0) < args.min_matches:
        return None, len(good0), 0.0
    if args.motion_model == 'homography':
        H, mask = cv2.findHomography(good1, good0, cv2.RANSAC, args.ransac_thresh)
        if H is None:
            return None, len(good0), 0.0
        inlier_ratio = float(mask.mean()) if mask is not None else 0.0
        return H.astype(np.float32), len(good0), inlier_ratio
    A, inliers = cv2.estimateAffinePartial2D(good0, good1, method=cv2.RANSAC,
                                             ransacReprojThreshold=args.ransac_thresh,
                                             maxIters=2000, confidence=0.99)
    if A is None:
        return None, len(good0), 0.0
    Ainv = cv2.invertAffineTransform(A).astype(np.float32)
    inlier_ratio = float(inliers.mean()) if inliers is not None else 0.0
    return Ainv, len(good0), inlier_ratio


def warp_ref_to_seed(ref, transform, motion_model, size):
    h, w = size
    if motion_model == 'homography':
        warped = cv2.warpPerspective(ref, transform, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        valid = cv2.warpPerspective(np.ones((h, w), np.uint8) * 255, transform, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    else:
        warped = cv2.warpAffine(ref, transform, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        valid = cv2.warpAffine(np.ones((h, w), np.uint8) * 255, transform, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return warped, (valid > 0).astype(np.float32)


def transform_motion_magnitude(transform, motion_model, size):
    h, w = size
    pts = np.array([[[0, 0]], [[w - 1, 0]], [[0, h - 1]], [[w - 1, h - 1]], [[w * 0.5, h * 0.5]]], dtype=np.float32)
    if motion_model == 'homography':
        dst = cv2.perspectiveTransform(pts, transform)
    else:
        dst = cv2.transform(pts, transform)
    d = dst.reshape(-1, 2) - pts.reshape(-1, 2)
    return float(np.mean(np.sqrt((d ** 2).sum(axis=1))))


def calc_flow(seed_gray, target_gray, method='dis'):
    s = seed_gray.astype(np.uint8); t = target_gray.astype(np.uint8)
    if method == 'farneback':
        return cv2.calcOpticalFlowFarneback(s, t, None, 0.5, 3, 21, 3, 5, 1.2, 0)
    if hasattr(cv2, 'DISOpticalFlow_create'):
        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        return dis.calc(s, t, None)
    return cv2.calcOpticalFlowFarneback(s, t, None, 0.5, 3, 21, 3, 5, 1.2, 0)


def make_line_kernels(length=25, width=3):
    length = ensure_odd(length, 5); width = max(1, int(width))
    ks = []
    kh = np.ones((width, length), dtype=np.uint8); ks.append(kh)
    kv = np.ones((length, width), dtype=np.uint8); ks.append(kv)
    kd1 = np.zeros((length, length), dtype=np.uint8); cv2.line(kd1, (0, 0), (length-1, length-1), 1, width); ks.append(kd1)
    kd2 = np.zeros((length, length), dtype=np.uint8); cv2.line(kd2, (0, length-1), (length-1, 0), 1, width); ks.append(kd2)
    return ks


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


def hair_morphology(gray, args):
    rs = []
    for k in make_line_kernels(args.hair_line_length, args.hair_line_width):
        rs.append(cv2.morphologyEx(gray.astype(np.uint8), cv2.MORPH_BLACKHAT, k).astype(np.float32))
    line = np.max(np.stack(rs, axis=0), axis=0)
    line = normalize_01(line, 0, max(1, np.percentile(line, 98)))
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    jxx = cv2.GaussianBlur(gx * gx, (15, 15), 0); jyy = cv2.GaussianBlur(gy * gy, (15, 15), 0); jxy = cv2.GaussianBlur(gx * gy, (15, 15), 0)
    coherence = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / (jxx + jyy + 1e-6)
    return line, np.clip(coherence, 0, 1).astype(np.float32)


def calculate_motion_residual_metrics(frames, args):
    start = time()
    seed = frames[0]
    seed_gray = gray_from_bgr(seed)
    h, w = seed_gray.shape[:2]
    residual_maps, hit_maps, valid_masks, motion_mags = [], [], [], []
    flow_u = flow_v = None; flow_key = -1.0
    homography_ok = 0

    for ref in frames[1:]:
        ref_gray = gray_from_bgr(ref)
        transform, nmatch, inlier_ratio = estimate_ref_to_seed_motion(seed_gray, ref_gray, args)
        if transform is None or inlier_ratio < args.min_inlier_ratio:
            continue
        warped_gray, valid = warp_ref_to_seed(ref_gray, transform, args.motion_model, (h, w))
        residual = np.abs(seed_gray - warped_gray.astype(np.float32)) * valid
        residual_norm = np.clip((residual - args.residual_low) / max(args.residual_scale, 1e-6), 0, 1)
        hit = ((residual >= args.residual_hit_threshold).astype(np.float32) * valid)
        residual_maps.append(residual_norm.astype(np.float32))
        hit_maps.append(hit.astype(np.float32))
        valid_masks.append(valid.astype(np.float32))
        motion_mags.append(transform_motion_magnitude(transform, args.motion_model, (h, w)))
        homography_ok += 1

        flow = calc_flow(seed_gray, ref_gray, args.flow_method)
        u, v = flow[..., 0].astype(np.float32), flow[..., 1].astype(np.float32)
        mag = np.sqrt(u * u + v * v)
        key = float(np.percentile(mag, 95))
        if key > flow_key:
            flow_key = key; flow_u = u; flow_v = v

    z = np.zeros_like(seed_gray, dtype=np.float32)
    if homography_ok == 0:
        return {
            'motion_residual_score': float('nan'), 'dust_score': float('nan'), 'hair_score': float('nan'), 'black_occlusion_score': float('nan'),
            'global_motion_mean': float('nan'), 'residual_mean': float('nan'), 'residual_p95': float('nan'), 'persistent_mean': float('nan'),
            'persistent_top_mean': float('nan'), 'dust_spot_response': float('nan'), 'hair_line_response': float('nan'), 'hair_orientation_response': float('nan'),
            'homography_ok_frames': 0, 'valid_frames': len(frames), 'elapsed_time': time() - start,
            'residual_heat': z,
            'raw_residual_heat': z, 'hit_ratio_heat': z, 'residual_prior_heat': z,
            'persistent_heat': z,
            'dust_shape_heat': z, 'hair_shape_heat': z, 'black_prior_heat': z,
            'dust_heat': z, 'hair_heat': z, 'black_heat': z,
            'flow_mag_map': z, 'flow_u_map': z, 'flow_v_map': z,
        }

    residual_stack = np.stack(residual_maps, axis=0)
    hit_stack = np.stack(hit_maps, axis=0)
    valid_stack = np.stack(valid_masks, axis=0)
    valid_count = np.maximum(valid_stack.sum(axis=0), 1.0)
    residual_mean = (residual_stack * valid_stack).sum(axis=0) / valid_count
    hit_ratio = hit_stack.sum(axis=0) / valid_count
    persistent = np.clip(residual_mean * (0.5 + 0.5 * hit_ratio), 0, 1).astype(np.float32)

    # Slightly suppress pure strong seed edges; background registration errors often live there.
    gx = cv2.Sobel(seed_gray, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(seed_gray, cv2.CV_32F, 0, 1, ksize=3)
    edge = normalize_01(np.sqrt(gx * gx + gy * gy), 0, max(1, np.percentile(np.sqrt(gx * gx + gy * gy), 98)))
    residual_prior = np.clip(persistent * (1.0 - args.edge_suppress_weight * edge), 0, 1)

    dust_shape = dust_morphology(seed_gray, args)
    hair_line, hair_orient = hair_morphology(seed_gray, args)
    black_degree = np.exp(-seed_gray / max(args.black_luma_scale, 1e-6)).astype(np.float32)

    hair_shape = np.sqrt(np.clip(hair_line * (0.4 + 0.6 * hair_orient), 0, 1)).astype(np.float32)
    dust_map = np.clip(residual_prior * dust_shape, 0, 1)
    hair_map = np.clip(residual_prior * hair_shape, 0, 1)
    black_map = np.clip(residual_prior * black_degree, 0, 1)

    ur, residual_heat = aggregate_map_to_units(residual_prior, args.unit_width, args.unit_height)
    ud, dust_heat = aggregate_map_to_units(dust_map, args.unit_width, args.unit_height)
    uh, hair_heat = aggregate_map_to_units(hair_map, args.unit_width, args.unit_height)
    ub, black_heat = aggregate_map_to_units(black_map, args.unit_width, args.unit_height)
    residual_score, _ = top_percent_mean(ur, args.top_unit_percent, True)
    dust_score, _ = top_percent_mean(ud, args.top_unit_percent, True)
    hair_score, _ = top_percent_mean(uh, max(args.top_unit_percent, 3.0), True)
    black_score, _ = top_percent_mean(ub, args.top_unit_percent, True)
    persistent_top, _ = top_percent_mean(ur, args.top_unit_percent, True)

    if flow_u is None:
        flow_u = z; flow_v = z; flow_mag = z
    else:
        flow_mag = np.sqrt(flow_u * flow_u + flow_v * flow_v).astype(np.float32)

    return {
        'motion_residual_score': 100.0 * residual_score,
        'dust_score': 100.0 * dust_score,
        'hair_score': 100.0 * hair_score,
        'black_occlusion_score': 100.0 * black_score,
        'global_motion_mean': float(np.mean(motion_mags)),
        'residual_mean': float((residual_mean * (valid_stack.mean(axis=0) > 0)).mean()),
        'residual_p95': float(np.percentile(residual_mean, 95)),
        'persistent_mean': float(persistent.mean()),
        'persistent_top_mean': float(persistent_top),
        'dust_spot_response': float(dust_shape.mean()),
        'hair_line_response': float(hair_line.mean()),
        'hair_orientation_response': float(hair_orient.mean()),
        'homography_ok_frames': homography_ok,
        'valid_frames': len(frames),
        'elapsed_time': time() - start,
        'residual_heat': residual_heat,
        'raw_residual_heat': residual_mean.astype(np.float32),
        'hit_ratio_heat': hit_ratio.astype(np.float32),
        'residual_prior_heat': residual_prior.astype(np.float32),
        'persistent_heat': persistent,
        'dust_shape_heat': dust_shape.astype(np.float32),
        'hair_shape_heat': hair_shape.astype(np.float32),
        'black_prior_heat': black_degree.astype(np.float32),
        'dust_heat': dust_heat, 'hair_heat': hair_heat, 'black_heat': black_heat,
        'flow_mag_map': flow_mag, 'flow_u_map': flow_u, 'flow_v_map': flow_v,
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


def draw_flow_arrows(img, flow_u, flow_v, step=32, min_mag=None):
    out = img.copy(); h, w = flow_u.shape[:2]
    mag = np.sqrt(flow_u.astype(np.float32) ** 2 + flow_v.astype(np.float32) ** 2)
    if min_mag is None:
        min_mag = max(0.5, float(np.percentile(mag, 60)))
    for y in range(step // 2, h, step):
        for x in range(step // 2, w, step):
            u = float(flow_u[y, x]); v = float(flow_v[y, x]); m = float(mag[y, x])
            if m < min_mag:
                continue
            scale = min(3.0, (step * 0.8) / max(m, 1e-6))
            x2 = int(round(x + u * scale)); y2 = int(round(y + v * scale))
            cv2.arrowedLine(out, (x, y), (x2, y2), (0, 0, 0), 3, cv2.LINE_AA, tipLength=0.35)
            cv2.arrowedLine(out, (x, y), (x2, y2), (255, 255, 255), 1, cv2.LINE_AA, tipLength=0.35)
    return out


def flow_to_bgr(flow_u, flow_v, mag_p95=None, with_arrows=True):
    mag, ang = cv2.cartToPolar(flow_u.astype(np.float32), flow_v.astype(np.float32), angleInDegrees=False)
    if mag_p95 is None or math.isnan(float(mag_p95)):
        mag_p95 = float(np.percentile(mag, 95)) if mag.size else 1.0
    hsv = np.zeros((flow_u.shape[0], flow_u.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = np.clip(ang * 180.0 / np.pi / 2.0, 0, 179).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(normalize_01(mag, 0, max(1.0, float(mag_p95))) * 255.0, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return draw_flow_arrows(bgr, flow_u, flow_v) if with_arrows else bgr


def save_visualization(img_path, seed_img, result, out_dir, index, scene, alpha=0.55):
    base = np.clip(seed_img, 0, 255).astype(np.uint8)
    flow_bgr = flow_to_bgr(result['flow_u_map'], result['flow_v_map'])
    flow_mag = result['flow_mag_map']
    flow_mag_norm = normalize_01(flow_mag, 0, max(1, np.percentile(flow_mag, 95)))
    info = np.zeros_like(base)
    put_label(info, 'motion residual: warp refs to seed by global motion', 26)
    put_label(info, 'diagnostic maps are separated from final class maps', 54)
    put_label(info, f'frames={result["valid_frames"]} ok={result["homography_ok_frames"]}', 82)
    put_label(info, f'motion={format_float(result["global_motion_mean"])} residual_p95={format_float(result["residual_p95"])}', 110)
    panels = [
        base.copy(),
        flow_bgr,
        colorize_01(flow_mag_norm),
        overlay(base, result['raw_residual_heat'], alpha),
        overlay(base, result['hit_ratio_heat'], alpha),
        overlay(base, result['residual_prior_heat'], alpha),
        colorize_01(result['dust_shape_heat']),
        colorize_01(result['hair_shape_heat']),
        colorize_01(result['black_prior_heat']),
        overlay(base, result['dust_heat'], alpha),
        overlay(base, result['hair_heat'], alpha),
        overlay(base, result['black_heat'], alpha),
        info,
    ]
    labels = [
        'seed', 'flow_color+arrows', 'flow_mag',
        'raw_residual_mean', 'hit_ratio',
        f'residual_prior score={format_float(result["motion_residual_score"])}',
        'dust_shape_prior', 'hair_shape_prior', 'black_luma_prior',
        f'final_dust={format_float(result["dust_score"])}',
        f'final_hair={format_float(result["hair_score"])}',
        f'final_black={format_float(result["black_occlusion_score"])}',
        'info'
    ]
    labs = []
    for panel, label in zip(panels, labels):
        p = panel.copy(); put_label(p, label); labs.append(p)
    blank = np.zeros_like(base); put_label(blank, '')
    while len(labs) < 16:
        labs.append(blank.copy())
    canvas = np.concatenate([
        np.concatenate(labs[:4], axis=1),
        np.concatenate(labs[4:8], axis=1),
        np.concatenate(labs[8:12], axis=1),
        np.concatenate(labs[12:16], axis=1),
    ], axis=0)
    out = os.path.join(out_dir, f'{index:06d}_{make_safe_name(scene) if scene else "no_scene"}_{make_safe_name(os.path.splitext(os.path.basename(img_path))[0])}_motion_residual.jpg')
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
    if scene_matches(scene, split_aliases(args.black_positive_scenes)): return 'black_occlusion'
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
    labeled = [(lab, path, res) for lab, path, res in labeled if not math.isnan(float(res.get('motion_residual_score', float('nan'))))]
    msgs = ['Unified multi-class confusion matrix based on motion-residual scores.']
    if not labeled:
        msgs.append('  No labeled valid samples, skip.'); return msgs
    thresholds = {
        'dust': best_threshold(labeled, 'dust', 'dust_score'),
        'hair': best_threshold(labeled, 'hair', 'hair_score'),
        'black_occlusion': best_threshold(labeled, 'black_occlusion', 'black_occlusion_score'),
    }
    msgs.append('  ' + '; '.join(f'threshold[{k}]={format_float(v)}' for k, v in thresholds.items()))
    labels = ['normal', 'dust', 'hair', 'black_occlusion']
    for lab, _, _ in labeled:
        if lab not in labels: labels.append(lab)
    mat = {r: {c: 0 for c in labels} for r in labels}
    total=correct=0
    for true, path, res in labeled:
        candidates = []
        for lab, metric in [('dust','dust_score'), ('hair','hair_score'), ('black_occlusion','black_occlusion_score')]:
            th = thresholds[lab]; score = float(res[metric])
            if not math.isinf(th) and score > th:
                candidates.append(((score-th)/max(abs(th),1.0), lab))
        # If no specific defect wins but residual itself is high, mark as generic abnormal scene label if true is known abnormal.
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
    parser = argparse.ArgumentParser(description='Global-motion compensation residual validation for lens-attached dust/hair/occlusion.')
    parser.add_argument('-t', '--target', default=None)
    parser.add_argument('--target_txt', default=None)
    parser.add_argument('-m', '--metric_name', default='lens_motion_residual')
    parser.add_argument('--metric_mode', default='NR')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--save_txt_dir', default=None)
    parser.add_argument('--save_file', default=None)
    parser.add_argument('--no_vis', action='store_true')
    parser.add_argument('--sequence_mode', default='previous', choices=['previous','previous_next','even_span'])
    parser.add_argument('--window', type=int, default=10)
    parser.add_argument('--sample_count', type=int, default=7)
    parser.add_argument('--resize_width', type=int, default=640)
    parser.add_argument('--motion_model', default='affine', choices=['affine','homography'])
    parser.add_argument('--flow_method', default='dis', choices=['dis','farneback'])
    parser.add_argument('--max_corners', type=int, default=1200)
    parser.add_argument('--feature_quality', type=float, default=0.01)
    parser.add_argument('--feature_min_distance', type=int, default=8)
    parser.add_argument('--min_matches', type=int, default=30)
    parser.add_argument('--min_inlier_ratio', type=float, default=0.35)
    parser.add_argument('--ransac_thresh', type=float, default=3.0)
    parser.add_argument('--residual_low', type=float, default=5.0)
    parser.add_argument('--residual_scale', type=float, default=35.0)
    parser.add_argument('--residual_hit_threshold', type=float, default=18.0)
    parser.add_argument('--edge_suppress_weight', type=float, default=0.35)
    parser.add_argument('--unit_width', type=int, default=4)
    parser.add_argument('--unit_height', type=int, default=3)
    parser.add_argument('--top_unit_percent', type=float, default=1.0)
    parser.add_argument('--dust_spot_ksize', type=int, default=7)
    parser.add_argument('--dust_log_ksize', type=int, default=5)
    parser.add_argument('--dust_bg_ksize', type=int, default=31)
    parser.add_argument('--dust_bright_weight', type=float, default=0.6)
    parser.add_argument('--hair_line_length', type=int, default=25)
    parser.add_argument('--hair_line_width', type=int, default=3)
    parser.add_argument('--black_luma_scale', type=float, default=45.0)
    parser.add_argument('--score_metric', default='motion_residual_score', choices=SCORE_COLUMNS)
    parser.add_argument('--normal_scene', default='normal')
    parser.add_argument('--dust_positive_scenes', default='dust,dirty,灰尘,尘,脏污,污渍')
    parser.add_argument('--hair_positive_scenes', default='hair,毛发,头发')
    parser.add_argument('--black_positive_scenes', default='occlusion,black,遮挡,安装遮挡')
    parser.add_argument('--heatmap_alpha', type=float, default=0.55)
    args = parser.parse_args()
    if args.metric_mode != 'NR': raise ValueError('This script only supports NR mode.')
    if args.target is None and args.target_txt is None: raise ValueError('Please specify --target or --target_txt.')

    paths, scenes = get_input_paths(args.target, args.target_txt)
    orig_paths = list(paths)
    paths, seqs, load_paths, seed_adjustments, invalid_seeds, weak_seeds = build_seed_sequences(paths, args.window, args.sequence_mode, args.sample_count)
    print('Loading seed images and motion-residual sequence frames...')
    print(f'Seed images: {len(paths)}; images to load: {len(load_paths)}; sequence_mode={args.sequence_mode}')
    if seed_adjustments:
        print(f'Auto-shifted early seeds: {len(seed_adjustments)}')
        for old_p, new_p, old_i, new_i in seed_adjustments[:10]:
            print(f'  seed_shift idx {old_i}->{new_i}: {old_p} -> {new_p}')
        if len(seed_adjustments) > 10: print(f'  ... {len(seed_adjustments)-10} more seed shifts')
    if weak_seeds:
        print(f'WARNING: {len(weak_seeds)} seeds have fewer references than requested window/sample_count.')
    if invalid_seeds:
        print(f'WARNING: {len(invalid_seeds)} seeds still have no reference frame.')
    cache = {os.path.abspath(p): imread_image_resize(p, args.resize_width if args.resize_width > 0 else None) for p in tqdm(load_paths, unit='image')}

    save_txt_path=vis_dir=txt_f=None
    if args.save_txt_dir:
        save_txt_path = build_auto_txt_save_path(args.save_txt_dir, args.metric_name)
        if not args.no_vis: vis_dir = build_vis_save_dir(save_txt_path)
        txt_f = open(save_txt_path, 'w', encoding='utf-8')
        txt_f.write(f'metric_name: {args.metric_name}\nmetric_mode: NR\nscore_direction: larger_means_more_abnormal\n')
        txt_f.write('method: estimate dominant global background motion, warp reference frames to seed, then detect objects that do not follow background motion via persistent residuals.\n')
        for k in vars(args): txt_f.write(f'{k}: {getattr(args,k)}\n')
        txt_f.write(f'seed_count: {len(paths)}\nloaded_image_count: {len(load_paths)}\nauto_shifted_seed_count: {len(seed_adjustments)}\nweak_seed_count: {len(weak_seeds)}\ninvalid_seed_count: {len(invalid_seeds)}\nvis_dir: {vis_dir}\ntime: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
        if seed_adjustments:
            txt_f.write('auto_shifted_seeds:\n')
            for old_p, new_p, old_i, new_i in seed_adjustments:
                txt_f.write(f'  idx {old_i}->{new_i}: {old_p} -> {new_p}\n')
        if weak_seeds:
            txt_f.write('weak_seeds:\n')
            for old_p, new_p, got, need in weak_seeds:
                txt_f.write(f'  refs {got}/{need}: {old_p} -> {new_p}\n')
        if invalid_seeds:
            txt_f.write('invalid_seeds:\n')
            for old_p, new_p, reason in invalid_seeds:
                txt_f.write(f'  {reason}: {old_p} -> {new_p}\n')
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
        res = calculate_motion_residual_metrics(frames, args)
        vis = save_visualization(p, cache[os.path.abspath(p)], res, vis_dir, i, scene, args.heatmap_alpha) if vis_dir else ''
        score = float(res[args.score_metric])
        if not math.isnan(score): avg += score; cnt += 1
        for m in SCORE_COLUMNS: update_scene_stat(stats[m], scene, float(res[m]), p)
        samples.append((scene, p, res))
        vals = [format_float(res[c]) for c in RESULT_COLUMNS]; elapsed = format_float(res['elapsed_time'])
        prefix = f'[{scene}] ' if scene else ''
        pbar.update(1); pbar.set_description(f'{prefix}{args.metric_name}/{args.score_metric}: {format_float(score)}')
        pbar.write(f'{prefix}{os.path.basename(p)} residual={format_float(res["motion_residual_score"])} dust={format_float(res["dust_score"])} hair={format_float(res["hair_score"])} black={format_float(res["black_occlusion_score"])} ok={res["homography_ok_frames"]}/{max(0,res["valid_frames"]-1)} Time={elapsed}s')
        row = [scene or '', p] + vals + [elapsed, vis]
        if writer: writer.writerow(row)
        if txt_f: txt_f.write('\t'.join(map(str,row)) + '\n')
    pbar.close()
    msg = f'Average {args.metric_name}/{args.score_metric} score of {args.target or args.target_txt} with {cnt}/{len(paths)} valid images is: {format_float(avg/cnt if cnt else float("nan"))}'
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

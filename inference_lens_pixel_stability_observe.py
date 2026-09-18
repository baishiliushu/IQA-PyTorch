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
RESULT_COLUMNS = [
    'pixel_stability_score', 'similarity_mean', 'similarity_p95', 'similarity_max',
    'pair_hit_ratio_mean', 'temporal_inlier_ratio_mean',
    'temporal_gradient_mean', 'temporal_range_mean',
    'mean_adjacent_brightness_change', 'mean_adaptive_threshold',
    'overexp_excluded_ratio', 'valid_pixel_ratio',
    'sequence_length', 'valid_pair_count'
]


def normalize_input_path(path):
    path = path.strip().strip('"').strip("'")
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
            paths.append(p)
            scenes.append(scene)
    return paths, scenes


def is_image_file(path):
    return os.path.splitext(path)[1].lower() in IMAGE_EXTENSIONS


def natural_key(path):
    import re
    name = os.path.basename(path)
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r'(\d+)', name)]


def timestamp_key(path):
    """Sort names like 00_9803820953.jpg as (minute, timestamp-in-minute)."""
    import re
    stem = os.path.splitext(os.path.basename(path))[0]
    m = re.match(r'^(\d+)_(\d+)$', stem)
    if m:
        return (0, int(m.group(1)), int(m.group(2)), natural_key(path))
    nums = re.findall(r'\d+', stem)
    if nums:
        return (1, tuple(int(x) for x in nums), natural_key(path))
    return (2, natural_key(path))


def get_input_paths(target=None, target_txt=None):
    if target_txt:
        return read_paths_from_txt(target_txt)
    if not target:
        raise ValueError('Please specify --target or --target_txt.')
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


def _sample_by_mode(idx, total, win, step, sequence_mode):
    picked = {idx}
    if sequence_mode == 'previous':
        for k in range(1, win + 1):
            j = idx - k * step
            if j >= 0:
                picked.add(j)
    elif sequence_mode == 'previous_next':
        for k in range(1, win + 1):
            j = idx - k * step
            if j >= 0:
                picked.add(j)
            j = idx + k * step
            if j < total:
                picked.add(j)
    elif sequence_mode == 'forward':
        for k in range(1, win + 1):
            j = idx + k * step
            if j < total:
                picked.add(j)
    else:
        raise ValueError(f'Unsupported sequence_mode: {sequence_mode}')
    return picked


def _fill_nearest_indices(picked, idx, total, target_count):
    """Fill from nearest temporal neighbors when interval/edge makes samples short."""
    if len(picked) >= target_count or len(picked) >= total:
        return picked
    radius = 1
    while len(picked) < target_count and len(picked) < total:
        left, right = idx - radius, idx + radius
        if left >= 0:
            picked.add(left)
        if len(picked) >= target_count or len(picked) >= total:
            break
        if right < total:
            picked.add(right)
        radius += 1
        if left < 0 and right >= total:
            break
    return picked


def build_sequence_for_seed(seed_path, window, sample_interval, sequence_mode):
    """Build temporal sequence Ns from seed directory and sort by timestamp.

    window means the number of sampled frames on one side for previous_next,
    or the number of sampled previous/forward frames for previous/forward.
    sample_interval is measured in file-order steps. If the requested interval
    makes the sequence too short, reduce interval to 1 first; if the directory
    or seed position still cannot satisfy the requested window length, use every
    available neighboring frame, and when requested length exceeds the directory
    size, use all images in the directory.
    """
    seed_abs = os.path.abspath(seed_path)
    imgs = list_images_in_same_dir(seed_abs)
    amap = {os.path.abspath(p): p for p in imgs}
    if seed_abs not in amap:
        raise FileNotFoundError(f'Seed image does not exist in its directory list: {seed_path}')
    seed_real = amap[seed_abs]
    idx = imgs.index(seed_real)
    win = max(0, int(window))
    step = max(1, int(sample_interval))
    total = len(imgs)

    if sequence_mode == 'previous_next':
        target_count = 2 * win + 1
    else:
        target_count = win + 1
    target_count = max(1, int(target_count))

    # If the requested temporal length is larger than the directory itself, do
    # not fail or skip: use all available images.
    if target_count >= total:
        return list(imgs)

    picked = _sample_by_mode(idx, total, win, step, sequence_mode)

    # Sacrifice sampling interval first to satisfy the requested window length.
    if len(picked) < target_count and step > 1:
        picked = _sample_by_mode(idx, total, win, 1, sequence_mode)

    # If edge position still cannot satisfy the count under the strict mode, use
    # nearest frames from the other side as fallback rather than not computing.
    if len(picked) < target_count:
        picked = _fill_nearest_indices(picked, idx, total, target_count)

    return [imgs[i] for i in sorted(picked)]


def build_all_sequences(seed_paths, window, sample_interval, sequence_mode):
    seqs, all_paths, weak = [], set(), []
    for p in seed_paths:
        seq = build_sequence_for_seed(p, window, sample_interval, sequence_mode)
        if len(seq) < 2:
            weak.append((p, len(seq)))
        seqs.append(seq)
        all_paths.update(seq)
    return seqs, sorted(all_paths, key=timestamp_key), weak


def imread_image(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f'Failed to read image: {path}')
    return img


def preprocess_each_frame(img, args):
    """Gray + brightness equalization + highlight suppression.

    1) convert to gray;
    2) percentile clipping suppresses over-exposed outliers;
    3) CLAHE reduces global exposure differences before pixel comparison.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lo = float(np.percentile(gray, args.pre_clip_low))
    hi = float(np.percentile(gray, args.pre_clip_high))
    if hi - lo > 1e-6:
        gray = np.clip((gray - lo) * 255.0 / (hi - lo), 0, 255)
    else:
        gray = np.clip(gray, 0, 255)
    gray_u8 = gray.astype(np.uint8)
    if args.clahe_clip > 0:
        clahe = cv2.createCLAHE(clipLimit=args.clahe_clip, tileGridSize=(args.clahe_grid, args.clahe_grid))
        gray_u8 = clahe.apply(gray_u8)
    if args.overexp_suppress_threshold < 255:
        mask = gray_u8 >= args.overexp_suppress_threshold
        if np.any(mask):
            # Compress saturated pixels toward local median so they do not dominate
            # adjacent-frame comparisons.
            med = cv2.medianBlur(gray_u8, 5)
            gray_u8 = gray_u8.copy()
            gray_u8[mask] = med[mask]
    return gray_u8.astype(np.float32)


def apply_pixel_tol_bounds(thr, args):
    if args.min_pixel_tol > 0:
        thr = max(float(args.min_pixel_tol), float(thr))
    if args.max_pixel_tol > 0:
        thr = min(float(args.max_pixel_tol), float(thr))
    return float(thr)


def detect_long_overexposure_mask(frames, args):
    """Detect long-time bright and spatially concentrated over-exposed pixels.

    These pixels are usually saturated highlight regions. They can remain very
    stable over time but are not lens contaminants, so they are excluded from
    all pixel-stability statistics and shown as black in the similarity map.
    """
    if not frames or args.overexp_long_threshold >= 255:
        h, w = frames[0].shape[:2]
        return np.zeros((h, w), dtype=bool)

    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) for f in frames]
    stack = np.stack(grays, axis=0)
    bright = (stack >= float(args.overexp_long_threshold)).astype(np.float32)
    persistence = bright.mean(axis=0)

    k = max(3, int(args.overexp_neighborhood))
    if k % 2 == 0:
        k += 1
    # Spatial concentration in a large neighborhood. A single hot pixel will be
    # ignored; a persistent saturated patch will be excluded.
    concentration = cv2.boxFilter((persistence >= args.overexp_persistence_ratio).astype(np.float32),
                                  ddepth=-1, ksize=(k, k), normalize=True)
    mask = ((persistence >= args.overexp_persistence_ratio) &
            (concentration >= args.overexp_neighborhood_ratio))

    # Slightly dilate the exclusion mask so highlight boundaries do not become
    # false stable contours.
    dk = max(1, int(args.overexp_exclude_dilate))
    if dk > 1:
        if dk % 2 == 0:
            dk += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dk, dk))
        mask = cv2.dilate(mask.astype(np.uint8), kernel) > 0
    return mask.astype(bool)


def valid_mean(x, valid_mask):
    vals = x[valid_mask]
    return float(vals.mean()) if vals.size else 0.0


def valid_percentile(x, valid_mask, q):
    vals = x[valid_mask]
    return float(np.percentile(vals, q)) if vals.size else 0.0


def calculate_pixel_stability(frames, args):
    start = time()
    pre = [preprocess_each_frame(f, args) for f in frames]
    n = len(pre)
    h, w = pre[0].shape[:2]
    overexp_mask = detect_long_overexposure_mask(frames, args)
    valid_mask = ~overexp_mask

    if n <= 1:
        sim = np.zeros((h, w), dtype=np.float32)
        pair_hit_ratio = np.zeros_like(sim)
        temporal_inlier_ratio = np.zeros_like(sim)
        temporal_gradient = np.zeros_like(sim)
        temporal_range = np.zeros_like(sim)
        brightness_changes, thresholds = [], []
    else:
        stack = np.stack(pre, axis=0).astype(np.float32)
        diffs = np.abs(stack[1:] - stack[:-1]).astype(np.float32)

        brightness_changes, thresholds = [], []
        pair_hits = []
        for i in range(1, n):
            prev, cur = pre[i - 1], pre[i]
            global_delta = abs(float(cur.mean()) - float(prev.mean()))
            thr = apply_pixel_tol_bounds(args.brightness_change_ratio * global_delta, args)
            pair_hits.append((np.abs(cur - prev) <= thr).astype(np.float32))
            brightness_changes.append(global_delta)
            thresholds.append(thr)

        # 同位置像素必须在整个时间轴上稳定才得高分，而非只依赖相邻两帧。
        pair_hit_ratio = np.mean(np.stack(pair_hits, axis=0), axis=0).astype(np.float32)
        temporal_gradient = diffs.mean(axis=0).astype(np.float32)
        temporal_range = (np.percentile(stack, 95, axis=0) - np.percentile(stack, 5, axis=0)).astype(np.float32)

        base_thr = float(np.mean(thresholds)) if thresholds else 0.0
        series_thr = apply_pixel_tol_bounds(base_thr * args.series_threshold_scale, args)
        series_thr = max(series_thr, 1e-6)
        median = np.median(stack, axis=0).astype(np.float32)
        temporal_inlier_ratio = (np.abs(stack - median[None, :, :]) <= series_thr).mean(axis=0).astype(np.float32)

        gradient_score = np.exp(-temporal_gradient / series_thr).astype(np.float32)
        range_score = np.exp(-temporal_range / max(series_thr * args.range_threshold_scale, 1e-6)).astype(np.float32)
        sim = np.clip(pair_hit_ratio * temporal_inlier_ratio * gradient_score * range_score, 0, 1).astype(np.float32)

    # 长时间特别亮且在较大邻域内集中分布的区域认为是过曝，完全不参与统计；
    # 可视化中直接置黑，避免被误认为稳定污染物。
    sim_vis = sim.copy()
    sim_vis[overexp_mask] = 0.0

    return {
        'pixel_stability_score': valid_percentile(sim, valid_mask, args.score_percentile) * 100.0,
        'similarity_mean': valid_mean(sim, valid_mask),
        'similarity_p95': valid_percentile(sim, valid_mask, 95),
        'similarity_max': valid_percentile(sim, valid_mask, 100),
        'pair_hit_ratio_mean': valid_mean(pair_hit_ratio, valid_mask),
        'temporal_inlier_ratio_mean': valid_mean(temporal_inlier_ratio, valid_mask),
        'temporal_gradient_mean': valid_mean(temporal_gradient, valid_mask),
        'temporal_range_mean': valid_mean(temporal_range, valid_mask),
        'mean_adjacent_brightness_change': float(np.mean(brightness_changes)) if brightness_changes else 0.0,
        'mean_adaptive_threshold': float(np.mean(thresholds)) if thresholds else 0.0,
        'overexp_excluded_ratio': float(overexp_mask.mean()),
        'valid_pixel_ratio': float(valid_mask.mean()),
        'sequence_length': float(n),
        'valid_pair_count': float(max(0, n - 1)),
        'elapsed_time': time() - start,
        'similarity_map': sim_vis,
        'similarity_u8': np.clip(sim_vis * 255.0, 0, 255).astype(np.uint8),
        'overexp_mask_u8': (overexp_mask.astype(np.uint8) * 255),
        'pre_seed': pre[0].astype(np.uint8),
    }


def format_float(v):
    try:
        v = float(v)
    except Exception:
        return str(v)
    return 'nan' if math.isnan(v) else f'{v:.4f}'


def make_safe_name(text):
    import re
    return re.sub(r'[^0-9A-Za-z._-]+', '_', str(text or '').strip()).strip('_') or 'none'


def build_auto_txt_save_path(save_txt_dir, metric_name):
    os.makedirs(save_txt_dir, exist_ok=True)
    return os.path.join(save_txt_dir, f'{datetime.now().strftime("%Y%m%d_%H%M%S")}.{metric_name.replace(os.sep, "_")}.txt')


def build_vis_save_dir(save_txt_path):
    d = os.path.join(os.path.dirname(os.path.abspath(save_txt_path)), os.path.splitext(os.path.basename(save_txt_path))[0])
    os.makedirs(d, exist_ok=True)
    return d


def put_label(img, text, y=26):
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)


def colorize_gray_u8(gray):
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def save_visualization(img_path, seed_img, result, out_dir, index, scene):
    base = seed_img.copy()
    pre = colorize_gray_u8(result['pre_seed'])
    sim_gray = colorize_gray_u8(result['similarity_u8'])
    info = np.zeros_like(base)
    put_label(info, f'pixel_stability_score={format_float(result["pixel_stability_score"])}', 26)
    put_label(info, f'Ns={int(result["sequence_length"])} pairs={int(result["valid_pair_count"])}', 54)
    put_label(info, f'mean_brightness_delta={format_float(result["mean_adjacent_brightness_change"])}', 82)
    put_label(info, f'mean_pixel_threshold={format_float(result["mean_adaptive_threshold"])}', 110)

    panels = [base, pre, sim_gray, info]
    labels = ['seed', 'preprocessed gray', 'similarity_score * 255', 'info']
    h, w = base.shape[:2]
    max_w = min(640, max(320, w))
    max_h = int(round(max_w * h / float(w)))
    resized = []
    for p, lab in zip(panels, labels):
        r = cv2.resize(p, (max_w, max_h), interpolation=cv2.INTER_AREA)
        put_label(r, lab)
        resized.append(r)
    canvas = np.concatenate([
        np.concatenate(resized[:2], axis=1),
        np.concatenate(resized[2:], axis=1),
    ], axis=0)
    stem = f'{index:06d}_{make_safe_name(scene) if scene else "no_scene"}_{make_safe_name(os.path.splitext(os.path.basename(img_path))[0])}'
    vis_path = os.path.join(out_dir, stem + '.jpg')
    cv2.imwrite(vis_path, canvas)
    return vis_path


def update_scene_stat(stats, scene, score, path):
    if not scene:
        return
    st = stats[scene]
    st['sum'] += score
    st['count'] += 1
    if st['max_score'] is None or score > st['max_score']:
        st['max_score'], st['max_path'] = score, path
    if st['min_score'] is None or score < st['min_score']:
        st['min_score'], st['min_path'] = score, path


def main():
    parser = argparse.ArgumentParser(description='Observe per-pixel temporal similarity after gray/equalization preprocessing.')
    parser.add_argument('-t', '--target', default=None)
    parser.add_argument('--target_txt', default=None)
    parser.add_argument('-m', '--metric_name', default='lens_pixel_stability_observe')
    parser.add_argument('--metric_mode', default='NR')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--save_txt_dir', default=None)
    parser.add_argument('--save_file', default=None)
    parser.add_argument('--no_vis', action='store_true')
    parser.add_argument('--sequence_mode', default='previous_next', choices=['previous', 'previous_next', 'forward'])
    parser.add_argument('--window', type=int, default=5, help='sampled frame count per side for previous_next, or previous/forward count.')
    parser.add_argument('--sample_interval', type=int, default=1, help='file-order sampling interval inside the seed directory.')
    parser.add_argument('--brightness_change_ratio', type=float, default=0.30, help='pixel drift threshold = this ratio * adjacent whole-image mean brightness change.')
    parser.add_argument('--min_pixel_tol', type=float, default=0.0, help='optional lower bound for pixel drift threshold; default follows the requested formula exactly.')
    parser.add_argument('--max_pixel_tol', type=float, default=0.0, help='optional upper bound for pixel drift threshold; <=0 disables.')
    parser.add_argument('--score_percentile', type=float, default=95.0, help='summary score percentile of similarity map, visualization still saves full score*255 map.')
    parser.add_argument('--series_threshold_scale', type=float, default=1.0, help='threshold scale for comparing each same-position pixel to its temporal median.')
    parser.add_argument('--range_threshold_scale', type=float, default=2.0, help='larger value makes the full-sequence temporal range penalty softer.')
    parser.add_argument('--pre_clip_low', type=float, default=1.0)
    parser.add_argument('--pre_clip_high', type=float, default=99.0)
    parser.add_argument('--clahe_clip', type=float, default=2.0)
    parser.add_argument('--clahe_grid', type=int, default=8)
    parser.add_argument('--overexp_suppress_threshold', type=int, default=245)
    parser.add_argument('--overexp_long_threshold', type=int, default=245, help='raw gray threshold for long-time bright overexposure exclusion; >=255 disables.')
    parser.add_argument('--overexp_persistence_ratio', type=float, default=0.70, help='pixel must be bright in at least this ratio of Ns frames.')
    parser.add_argument('--overexp_neighborhood', type=int, default=31, help='large neighborhood size for concentrated overexposure detection.')
    parser.add_argument('--overexp_neighborhood_ratio', type=float, default=0.35, help='minimum bright-persistent ratio in the neighborhood.')
    parser.add_argument('--overexp_exclude_dilate', type=int, default=5, help='dilate excluded overexposure mask to cover highlight boundary.')
    args = parser.parse_args()

    if args.metric_mode != 'NR':
        raise ValueError('This script only supports NR mode.')
    if args.target is None and args.target_txt is None:
        raise ValueError('Please specify --target or --target_txt.')

    input_paths, input_scenes = get_input_paths(args.target, args.target_txt)
    if not input_paths:
        raise ValueError('No input images found.')
    seqs, load_paths, weak = build_all_sequences(input_paths, args.window, args.sample_interval, args.sequence_mode)
    print('Loading seed images and temporal sequence frames...')
    print(f'Seed images: {len(input_paths)}; images to load: {len(load_paths)}; sequence_mode={args.sequence_mode}; window={args.window}; sample_interval={args.sample_interval}')
    if weak:
        print(f'WARNING: {len(weak)} seeds have fewer than 2 frames and cannot form adjacent pairs.')
    cache = {os.path.abspath(p): imread_image(p) for p in tqdm(load_paths, unit='image')}

    save_txt_path = vis_dir = txt_f = None
    if args.save_txt_dir:
        save_txt_path = build_auto_txt_save_path(args.save_txt_dir, args.metric_name)
        if not args.no_vis:
            vis_dir = build_vis_save_dir(save_txt_path)
        txt_f = open(save_txt_path, 'w', encoding='utf-8')
        txt_f.write(f'metric_name: {args.metric_name}\nmetric_mode: NR\n')
        txt_f.write('score_direction: larger_means_more_same-position_pixels_are_temporally_similar\n')
        txt_f.write('method: gray + exposure-normalized preprocessing, exclude long-time bright concentrated overexposure pixels, then per-pixel temporal-gradient stability. It combines adjacent-pair hits, closeness to temporal median, mean temporal gradient, and full-sequence temporal range. Base tolerance still comes from 0.30 * adjacent whole-image brightness change.\n')
        for k in vars(args):
            txt_f.write(f'{k}: {getattr(args, k)}\n')
        txt_f.write(f'seed_count: {len(input_paths)}\nloaded_image_count: {len(load_paths)}\nweak_seed_count: {len(weak)}\nvis_dir: {vis_dir}\ntime: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')
        txt_f.write('scene\timage\t' + '\t'.join(RESULT_COLUMNS) + '\ttime\tvisualization\tsequence\n')

    sf = writer = None
    if args.save_file:
        sf = open(args.save_file, 'w', newline='', encoding='utf-8')
        writer = csv.writer(sf)
        writer.writerow(['scene', 'image'] + RESULT_COLUMNS + ['time', 'visualization', 'sequence'])

    stats = defaultdict(lambda: {'sum': 0.0, 'count': 0, 'max_score': None, 'max_path': None, 'min_score': None, 'min_path': None})
    avg = 0.0
    count = 0
    pbar = tqdm(total=len(input_paths), unit='image')
    for i, p in enumerate(input_paths):
        scene = input_scenes[i]
        frames = [cache[os.path.abspath(x)] for x in seqs[i]]
        res = calculate_pixel_stability(frames, args)
        vis = ''
        if vis_dir:
            vis = save_visualization(p, cache[os.path.abspath(p)], res, vis_dir, i, scene)
        score = float(res['pixel_stability_score'])
        avg += score
        count += 1
        update_scene_stat(stats, scene, score, p)
        elapsed = format_float(res['elapsed_time'])
        prefix = f'[{scene}] ' if scene else ''
        pbar.update(1)
        pbar.set_description(f'{prefix}{args.metric_name}: {format_float(score)}')
        pbar.write(f'{prefix}{os.path.basename(p)} score={format_float(score)} Ns={int(res["sequence_length"])} mean_thr={format_float(res["mean_adaptive_threshold"])} mean_brightness_delta={format_float(res["mean_adjacent_brightness_change"])} Time={elapsed}s')
        seq_str = ','.join(seqs[i])
        row = [scene or '', p] + [format_float(res[c]) for c in RESULT_COLUMNS] + [elapsed, vis, seq_str]
        if writer:
            writer.writerow(row)
        if txt_f:
            txt_f.write('\t'.join(map(str, row)) + '\n')
    pbar.close()

    msg = f'Average {args.metric_name}/pixel_stability_score of {args.target or args.target_txt} with {count}/{len(input_paths)} valid images is: {format_float(avg / count if count else float("nan"))}'
    print(msg)
    scene_msgs = []
    if stats:
        print('Scene statistics:')
        scene_msgs.append('Scene statistics:')
        for scene, st in stats.items():
            if st['count'] <= 0:
                continue
            line = f'  [{scene}] count={st["count"]}: avg={format_float(st["sum"] / st["count"])} max={format_float(st["max_score"])} ({st["max_path"]}) min={format_float(st["min_score"])} ({st["min_path"]})'
            print(line)
            scene_msgs.append(line)
    if txt_f:
        txt_f.write('\n' + msg + '\n')
        for x in scene_msgs:
            txt_f.write(x + '\n')
        txt_f.close()
    if sf:
        sf.close()
    if args.save_file:
        print(f'Done! CSV results are in {args.save_file}.')
    if save_txt_path:
        print(f'Done! TXT results are in {save_txt_path}.')
    if vis_dir:
        print(f'Done! Visualizations are in {vis_dir}.')


if __name__ == '__main__':
    main()

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
SCORE_COLUMNS = ['optical_static_score', 'dust_score', 'hair_score']
RESULT_COLUMNS = SCORE_COLUMNS + [
    'flow_mag_mean', 'flow_mag_p95', 'static_prior_mean', 'static_prior_top_mean',
    'flow_inconsistency_mean', 'dust_spot_response', 'hair_line_response',
    'hair_orientation_response', 'valid_frames'
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
    import re
    stem = os.path.splitext(os.path.basename(path))[0]
    nums = re.findall(r'\d+', stem)
    return (int(nums[-1]), natural_key(path)) if nums else (0, natural_key(path))


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


def sample_even_span_paths(paths, sample_count):
    if not paths:
        return []
    if sample_count <= 0 or sample_count >= len(paths):
        return list(paths)
    idxs = np.rint(np.linspace(0, len(paths) - 1, sample_count)).astype(np.int64)
    picked, seen = [], set()
    for i in idxs:
        i = int(max(0, min(len(paths) - 1, i)))
        if i not in seen:
            picked.append(i); seen.add(i)
    while len(picked) < sample_count:
        best, dist = None, -1
        for i in range(len(paths)):
            if i in seen:
                continue
            d = min(abs(i - p) for p in picked)
            if d > dist:
                best, dist = i, d
        picked.append(best); seen.add(best)
    return [paths[i] for i in sorted(picked)]


def build_seed_sequences(seed_paths, window, sequence_mode, sample_count):
    seqs, all_paths, missing = [], set(), []
    for seed_path in seed_paths:
        seed_abs = os.path.abspath(seed_path)
        imgs = list_images_in_same_dir(seed_abs)
        amap = {os.path.abspath(p): p for p in imgs}
        if seed_abs not in amap:
            missing.append(seed_path); seqs.append([]); continue
        seed_real = amap[seed_abs]
        if sequence_mode == 'even_span':
            seq = sample_even_span_paths(imgs, sample_count)
        elif sequence_mode == 'previous_next':
            idx = imgs.index(seed_real)
            seq = imgs[max(0, idx - window): min(len(imgs), idx + window + 1)]
        else:
            raise ValueError(f'Unsupported sequence_mode: {sequence_mode}')
        if seed_real not in seq:
            if sample_count > 0 and len(seq) >= sample_count:
                seed_idx = imgs.index(seed_real)
                rp = min(range(len(seq)), key=lambda i: abs(imgs.index(seq[i]) - seed_idx))
                seq[rp] = seed_real
                seq = sorted(set(seq), key=timestamp_key)
            else:
                seq = sorted(seq + [seed_real], key=timestamp_key)
        seqs.append(seq); all_paths.update(seq)
    if missing:
        raise FileNotFoundError('Some seed images do not exist: ' + ', '.join(missing[:5]))
    return seqs, sorted(all_paths, key=natural_key)


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


def calc_flow(seed_gray, target_gray, method='dis'):
    s = seed_gray.astype(np.uint8); t = target_gray.astype(np.uint8)
    if method == 'farneback':
        return cv2.calcOpticalFlowFarneback(s, t, None, 0.5, 3, 21, 3, 5, 1.2, 0)
    if hasattr(cv2, 'DISOpticalFlow_create'):
        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        return dis.calc(s, t, None)
    return cv2.calcOpticalFlowFarneback(s, t, None, 0.5, 3, 21, 3, 5, 1.2, 0)


def local_ring_mean(x, inner_ksize=7, outer_ksize=41):
    inner = cv2.GaussianBlur(x, (ensure_odd(inner_ksize), ensure_odd(inner_ksize)), 0)
    outer = cv2.GaussianBlur(x, (ensure_odd(outer_ksize), ensure_odd(outer_ksize)), 0)
    return inner, outer


def optical_static_prior(frames, args):
    seed_gray = gray_from_bgr(frames[0])
    static_maps, incons_maps, mag_maps = [], [], []
    for f in frames[1:]:
        g = gray_from_bgr(f)
        flow = calc_flow(seed_gray, g, args.flow_method)
        u, v = flow[..., 0], flow[..., 1]
        mag = np.sqrt(u * u + v * v).astype(np.float32)
        local_mag, ring_mag = local_ring_mean(mag, args.flow_local_ksize, args.flow_ring_ksize)
        low_local = np.exp(-local_mag / max(args.static_flow_scale, 1e-6))
        surrounding_motion = 1.0 - np.exp(-ring_mag / max(args.motion_flow_scale, 1e-6))
        contrast = np.clip((ring_mag - local_mag) / (ring_mag + 1e-6), 0, 1)
        static = np.clip(low_local * surrounding_motion * (0.4 + 0.6 * contrast), 0, 1)

        mu = cv2.GaussianBlur(u, (ensure_odd(args.flow_ring_ksize), ensure_odd(args.flow_ring_ksize)), 0)
        mv = cv2.GaussianBlur(v, (ensure_odd(args.flow_ring_ksize), ensure_odd(args.flow_ring_ksize)), 0)
        inc = np.sqrt((u - mu) ** 2 + (v - mv) ** 2) / (ring_mag + 1.0)
        inc = np.clip(inc, 0, 1).astype(np.float32)
        static_maps.append(static.astype(np.float32)); incons_maps.append(inc); mag_maps.append(mag)
    if not static_maps:
        z = np.zeros_like(seed_gray, dtype=np.float32)
        return z, z, z, z
    static_prior = np.mean(np.stack(static_maps, axis=0), axis=0).astype(np.float32)
    incons = np.mean(np.stack(incons_maps, axis=0), axis=0).astype(np.float32)
    mag = np.mean(np.stack(mag_maps, axis=0), axis=0).astype(np.float32)
    unit_static, static_prior = aggregate_map_to_units(static_prior, args.unit_width, args.unit_height)
    return static_prior, incons, mag, unit_static


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
    return line, np.clip(coherence, 0, 1).astype(np.float32)


def calculate_metrics(frames, args):
    start = time()
    gray = gray_from_bgr(frames[0])
    static_prior, incons, mag, unit_static = optical_static_prior(frames, args)
    dust_shape = dust_morphology(gray, args)
    hair_line, hair_orient = hair_morphology(gray, args)

    optical_map = np.clip(static_prior * (0.7 + 0.3 * incons), 0, 1).astype(np.float32)
    dust_map = np.clip(optical_map * dust_shape, 0, 1).astype(np.float32)
    hair_map = np.clip(optical_map * np.sqrt(np.clip(hair_line * (0.4 + 0.6 * hair_orient), 0, 1)), 0, 1).astype(np.float32)

    uo, optical_heat = aggregate_map_to_units(optical_map, args.unit_width, args.unit_height)
    ud, dust_heat = aggregate_map_to_units(dust_map, args.unit_width, args.unit_height)
    uh, hair_heat = aggregate_map_to_units(hair_map, args.unit_width, args.unit_height)
    optical_score, _ = top_percent_mean(uo, args.top_unit_percent, True)
    dust_score, _ = top_percent_mean(ud, args.top_unit_percent, True)
    hair_score, _ = top_percent_mean(uh, max(args.top_unit_percent, 3.0), True)
    static_top, _ = top_percent_mean(unit_static, args.top_unit_percent, True)
    return {
        'optical_static_score': 100.0 * optical_score,
        'dust_score': 100.0 * dust_score,
        'hair_score': 100.0 * hair_score,
        'flow_mag_mean': float(mag.mean()),
        'flow_mag_p95': float(np.percentile(mag, 95)),
        'static_prior_mean': float(static_prior.mean()),
        'static_prior_top_mean': float(static_top),
        'flow_inconsistency_mean': float(incons.mean()),
        'dust_spot_response': float(dust_shape.mean()),
        'hair_line_response': float(hair_line.mean()),
        'hair_orientation_response': float(hair_orient.mean()),
        'valid_frames': len(frames),
        'elapsed_time': time() - start,
        'optical_heat': optical_heat, 'dust_heat': dust_heat, 'hair_heat': hair_heat,
        'flow_mag_map': mag, 'inconsistency_map': incons,
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
    mag_norm = normalize_01(result['flow_mag_map'], 0, max(1, result['flow_mag_p95']))
    panels = [base.copy(), colorize_01(mag_norm), overlay(base, result['optical_heat'], alpha), colorize_01(result['inconsistency_map']), overlay(base, result['dust_heat'], alpha), overlay(base, result['hair_heat'], alpha)]
    labels = ['seed', f'flow_mag p95={format_float(result["flow_mag_p95"])}', f'optical_static={format_float(result["optical_static_score"])}', 'flow_inconsistency', f'dust={format_float(result["dust_score"])}', f'hair={format_float(result["hair_score"])}']
    labs = []
    for p, l in zip(panels, labels):
        p = p.copy(); put_label(p, l); labs.append(p)
    canvas = np.concatenate([np.concatenate(labs[:3], axis=1), np.concatenate(labs[3:], axis=1)], axis=0)
    out = os.path.join(out_dir, f'{index:06d}_{make_safe_name(scene) if scene else "no_scene"}_{make_safe_name(os.path.splitext(os.path.basename(img_path))[0])}_optical_static.jpg')
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
    vals = [(lab == label, float(res[metric])) for lab, _, res in samples if not math.isnan(float(res[metric]))]
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
        bal = 0.5*(pos_rec+neg_rec); acc = (tp+tn)/float(tp+tn+fp+fn)
        key = (bal, acc, th)
        if best is None or key > best[0]: best = (key, th)
    return best[1]


def multiclass_confusion_msgs(samples, args):
    labeled = [(scene_to_label(scene, args), path, res) for scene, path, res in samples if scene_to_label(scene, args)]
    msgs = ['Unified multi-class confusion matrix based on optical-static dust/hair scores.']
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
    parser = argparse.ArgumentParser(description='Optical-flow static-in-image validation for lens-attached dust/hair pollution.')
    parser.add_argument('-t', '--target', default=None)
    parser.add_argument('--target_txt', default=None)
    parser.add_argument('-m', '--metric_name', default='lens_optical_static')
    parser.add_argument('--metric_mode', default='NR')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--save_txt_dir', default=None)
    parser.add_argument('--save_file', default=None)
    parser.add_argument('--no_vis', action='store_true')
    parser.add_argument('--sequence_mode', default='previous_next', choices=['previous_next','even_span'])
    parser.add_argument('--window', type=int, default=3)
    parser.add_argument('--sample_count', type=int, default=7)
    parser.add_argument('--resize_width', type=int, default=640)
    parser.add_argument('--flow_method', default='dis', choices=['dis','farneback'])
    parser.add_argument('--flow_local_ksize', type=int, default=7)
    parser.add_argument('--flow_ring_ksize', type=int, default=41)
    parser.add_argument('--static_flow_scale', type=float, default=1.0)
    parser.add_argument('--motion_flow_scale', type=float, default=2.0)
    parser.add_argument('--unit_width', type=int, default=4)
    parser.add_argument('--unit_height', type=int, default=3)
    parser.add_argument('--top_unit_percent', type=float, default=1.0)
    parser.add_argument('--dust_spot_ksize', type=int, default=7)
    parser.add_argument('--dust_log_ksize', type=int, default=5)
    parser.add_argument('--dust_bg_ksize', type=int, default=31)
    parser.add_argument('--dust_bright_weight', type=float, default=0.6)
    parser.add_argument('--hair_line_length', type=int, default=25)
    parser.add_argument('--hair_line_width', type=int, default=3)
    parser.add_argument('--score_metric', default='optical_static_score', choices=SCORE_COLUMNS)
    parser.add_argument('--normal_scene', default='normal')
    parser.add_argument('--dust_positive_scenes', default='dust,dirty,灰尘,尘,脏污,污渍')
    parser.add_argument('--hair_positive_scenes', default='hair,毛发,头发')
    parser.add_argument('--heatmap_alpha', type=float, default=0.55)
    args = parser.parse_args()
    if args.metric_mode != 'NR': raise ValueError('This script only supports NR mode.')
    if args.target is None and args.target_txt is None: raise ValueError('Please specify --target or --target_txt.')

    paths, scenes = get_input_paths(args.target, args.target_txt)
    seqs, load_paths = build_seed_sequences(paths, args.window, args.sequence_mode, args.sample_count)
    print('Loading seed images and optical-flow sequence frames...')
    print(f'Seed images: {len(paths)}; images to load: {len(load_paths)}; sequence_mode={args.sequence_mode}')
    cache = {os.path.abspath(p): imread_image_resize(p, args.resize_width if args.resize_width > 0 else None) for p in tqdm(load_paths, unit='image')}

    save_txt_path=vis_dir=txt_f=None
    if args.save_txt_dir:
        save_txt_path = build_auto_txt_save_path(args.save_txt_dir, args.metric_name)
        if not args.no_vis: vis_dir = build_vis_save_dir(save_txt_path)
        txt_f = open(save_txt_path, 'w', encoding='utf-8')
        txt_f.write(f'metric_name: {args.metric_name}\nmetric_mode: NR\nscore_direction: larger_means_more_abnormal\n')
        txt_f.write('method: dense optical flow finds local static pixels whose surrounding area moves; dust/hair scores multiply this prior by morphology priors.\n')
        for k in vars(args): txt_f.write(f'{k}: {getattr(args,k)}\n')
        txt_f.write(f'seed_count: {len(paths)}\nloaded_image_count: {len(load_paths)}\nvis_dir: {vis_dir}\ntime: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')
        txt_f.write('scene\timage\t' + '\t'.join(RESULT_COLUMNS) + '\ttime\tvisualization\n')
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
        vis = save_visualization(p, cache[os.path.abspath(p)], res, vis_dir, i, scene, args.heatmap_alpha) if vis_dir else ''
        score = float(res[args.score_metric]); avg += score; cnt += 1
        for m in SCORE_COLUMNS: update_scene_stat(stats[m], scene, float(res[m]), p)
        samples.append((scene, p, res))
        vals = [format_float(res[c]) for c in RESULT_COLUMNS]; elapsed = format_float(res['elapsed_time'])
        prefix = f'[{scene}] ' if scene else ''
        pbar.update(1); pbar.set_description(f'{prefix}{args.metric_name}/{args.score_metric}: {format_float(score)}')
        pbar.write(f'{prefix}{os.path.basename(p)} optical={format_float(res["optical_static_score"])} dust={format_float(res["dust_score"])} hair={format_float(res["hair_score"])} frames={res["valid_frames"]} Time={elapsed}s')
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

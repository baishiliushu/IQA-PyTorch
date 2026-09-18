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
SCORE_COLUMNS = [
    'noise_score',
    'black_occlusion_score',
    'dust_score',
    'hair_score',
]
RESULT_COLUMNS = SCORE_COLUMNS + [
    'noise_response',
    'black_response',
    'dust_temporal_stable',
    'dust_spot_response',
    'dust_freq_response',
    'hair_temporal_stable',
    'hair_line_response',
    'hair_freq_response',
    'hair_edge_persistence',
    'valid_noise_frames',
    'valid_black_frames',
    'valid_dust_frames',
    'valid_hair_frames',
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
    if line.startswith('#-[') and line.endswith(']'):
        return line[3:-1].strip()
    return None


def read_paths_from_txt(txt_path):
    txt_dir = os.path.dirname(os.path.abspath(txt_path))
    paths, scenes = [], []
    current_scene = None
    with open(txt_path, 'r', encoding='utf-8') as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            scene = parse_scene_marker(line)
            if scene is not None:
                current_scene = scene
                continue
            if line.startswith('#'):
                continue
            line = normalize_input_path(line)
            if os.path.isabs(line):
                path = line
            elif os.path.exists(line):
                path = line
            else:
                path = os.path.join(txt_dir, line)
            paths.append(path)
            scenes.append(current_scene)
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
def get_input_paths(input_path, input_txt=None):
    if input_txt is not None:
        return read_paths_from_txt(input_txt)
    if os.path.isfile(input_path):
        return [input_path], [None]
    paths = sorted(glob.glob(os.path.join(input_path, '*')), key=natural_key)
    paths = [p for p in paths if is_image_file(p)]
    return paths, [None] * len(paths)


def list_images_in_same_dir(img_path, sort_by_timestamp=True):
    img_dir = os.path.dirname(os.path.abspath(img_path))
    if not os.path.isdir(img_dir):
        return []
    paths = [
        os.path.join(img_dir, name)
        for name in os.listdir(img_dir)
        if is_image_file(name) and os.path.isfile(os.path.join(img_dir, name))
    ]
    return sorted(paths, key=timestamp_key if sort_by_timestamp else natural_key)


def sample_even_span_paths(paths, sample_count):
    """Sample M images with maximum temporal span and near-even intervals."""
    if not paths:
        return []
    sample_count = int(sample_count)
    if sample_count <= 0 or sample_count >= len(paths):
        return list(paths)
    indices = np.linspace(0, len(paths) - 1, sample_count)
    indices = np.rint(indices).astype(np.int64)
    picked, seen = [], set()
    for idx in indices:
        idx = int(max(0, min(len(paths) - 1, idx)))
        if idx not in seen:
            picked.append(idx)
            seen.add(idx)
    while len(picked) < sample_count:
        best_idx, best_dist = None, -1
        for idx in range(len(paths)):
            if idx in seen:
                continue
            dist = min(abs(idx - p) for p in picked)
            if dist > best_dist:
                best_idx, best_dist = idx, dist
        picked.append(best_idx)
        seen.add(best_idx)
    return [paths[i] for i in sorted(picked)]


def build_one_seed_sequence(seed_path, window, sequence_mode='even_span', sample_count=7, include_seed=True):
    seed_abs = os.path.abspath(seed_path)
    dir_images = list_images_in_same_dir(seed_abs, sort_by_timestamp=True)
    abs_to_path = {os.path.abspath(p): p for p in dir_images}
    if seed_abs not in abs_to_path:
        raise FileNotFoundError(f'Seed image does not exist: {seed_path}')
    seed_real = abs_to_path[seed_abs]
    mode = (sequence_mode or 'even_span').lower()
    if mode == 'even_span':
        ctx = sample_even_span_paths(dir_images, sample_count)
    elif mode == 'previous_next':
        idx = dir_images.index(seed_real)
        start = max(0, idx - window)
        end = min(len(dir_images), idx + window + 1)
        ctx = dir_images[start:end]
    else:
        raise ValueError(f'Unsupported sequence_mode: {sequence_mode}')
    if include_seed and seed_real not in ctx:
        if sample_count > 0 and len(ctx) >= sample_count:
            seed_idx = dir_images.index(seed_real)
            replace_pos = min(range(len(ctx)), key=lambda i: abs(dir_images.index(ctx[i]) - seed_idx))
            ctx[replace_pos] = seed_real
            ctx = sorted(set(ctx), key=timestamp_key)
        else:
            ctx = sorted(ctx + [seed_real], key=timestamp_key)
    if not include_seed:
        ctx = [p for p in ctx if os.path.abspath(p) != seed_abs]
    return ctx


def resolve_metric_mode(metric_mode, global_mode):
    return global_mode if metric_mode == 'inherit' else metric_mode


def build_metric_sequences(seed_paths, args):
    modes = {
        'noise': resolve_metric_mode(args.noise_sequence_mode, args.sequence_mode),
        'black': resolve_metric_mode(args.black_sequence_mode, args.sequence_mode),
        'dust': resolve_metric_mode(args.dust_sequence_mode, args.sequence_mode),
        'hair': resolve_metric_mode(args.hair_sequence_mode, args.sequence_mode),
    }
    counts = {
        'noise': args.noise_sample_count if args.noise_sample_count > 0 else args.sample_count,
        'black': args.black_sample_count if args.black_sample_count > 0 else args.sample_count,
        'dust': args.dust_sample_count if args.dust_sample_count > 0 else args.sample_count,
        'hair': args.hair_sample_count if args.hair_sample_count > 0 else args.sample_count,
    }
    seqs = {k: [] for k in modes}
    all_paths = set()
    for seed_path in seed_paths:
        for name in modes:
            seq = build_one_seed_sequence(seed_path, args.window, modes[name], counts[name], include_seed=True)
            seqs[name].append(seq)
            all_paths.update(seq)
    return seqs, sorted(all_paths, key=natural_key), modes, counts


def imread_image_resize(img_path, resize_width):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f'Failed to read image: {img_path}')
    if resize_width is not None and resize_width > 0 and img.shape[1] != resize_width:
        scale = resize_width / float(img.shape[1])
        resize_height = max(1, int(round(img.shape[0] * scale)))
        img = cv2.resize(img, (resize_width, resize_height), interpolation=cv2.INTER_AREA)
    return img.astype(np.float32)


def gray_from_bgr(img):
    return cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)


def ensure_odd_kernel(v, min_value=1):
    v = int(v)
    if v < min_value:
        v = min_value
    if v % 2 == 0:
        v += 1
    return v


def normalize_01(x, low=None, high=None):
    x = x.astype(np.float32)
    if low is None:
        low = float(np.nanmin(x))
    if high is None:
        high = float(np.nanmax(x))
    if high - low < 1e-6:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - low) / (high - low), 0.0, 1.0).astype(np.float32)


def format_float(v):
    try:
        v = float(v)
    except Exception:
        return str(v)
    if math.isnan(v):
        return 'nan'
    return f'{v:.4f}'


def make_safe_name(text):
    import re
    text = '' if text is None else str(text)
    return re.sub(r'[^0-9A-Za-z._-]+', '_', text.strip()).strip('_') or 'none'


def build_auto_txt_save_path(save_txt_dir, metric_name):
    os.makedirs(save_txt_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    return os.path.join(save_txt_dir, f'{ts}.{metric_name.replace(os.sep, "_")}.txt')


def build_vis_save_dir(save_txt_path):
    out_dir = os.path.join(os.path.dirname(os.path.abspath(save_txt_path)), os.path.splitext(os.path.basename(save_txt_path))[0])
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def aggregate_map_to_units(value_map, unit_width=4, unit_height=3, reduce='mean'):
    value_map = value_map.astype(np.float32)
    h, w = value_map.shape[:2]
    uw, uh = max(1, int(unit_width)), max(1, int(unit_height))
    rows, cols = int(math.ceil(h / float(uh))), int(math.ceil(w / float(uw)))
    unit_map = np.zeros((rows, cols), dtype=np.float32)
    for r in range(rows):
        y0, y1 = r * uh, min(h, (r + 1) * uh)
        for c in range(cols):
            x0, x1 = c * uw, min(w, (c + 1) * uw)
            patch = value_map[y0:y1, x0:x1]
            if reduce == 'max':
                unit_map[r, c] = float(patch.max())
            elif reduce == 'median':
                unit_map[r, c] = float(np.median(patch))
            else:
                unit_map[r, c] = float(patch.mean())
    expanded = cv2.resize(unit_map, (w, h), interpolation=cv2.INTER_NEAREST).astype(np.float32)
    return unit_map, expanded


def top_percent_mean(unit_map, percent, largest=True):
    flat = unit_map.reshape(-1).astype(np.float32)
    if flat.size == 0:
        return float('nan'), 0
    percent = max(0.0, min(100.0, float(percent)))
    k = max(1, int(math.ceil(flat.size * percent / 100.0)))
    values = np.sort(flat)
    chosen = values[-k:] if largest else values[:k]
    return float(chosen.mean()), k


def temporal_variation_map(seed_img, frames, reduce='max'):
    if not frames:
        return np.zeros(seed_img.shape[:2], dtype=np.float32)
    diffs = []
    for f in frames:
        diff = np.abs(seed_img - f).mean(axis=2)
        diffs.append(diff)
    stack = np.stack(diffs, axis=0)
    reduce = (reduce or 'max').lower()
    if reduce == 'mean':
        return stack.mean(axis=0).astype(np.float32)
    if reduce == 'median':
        return np.median(stack, axis=0).astype(np.float32)
    if reduce == 'p75':
        return np.percentile(stack, 75, axis=0).astype(np.float32)
    if reduce == 'max':
        return stack.max(axis=0).astype(np.float32)
    raise ValueError(f'Unsupported temporal_reduce: {reduce}')


def stable_map_from_stack(frames, unit_width, unit_height, stable_variation_scale=3.0):
    if len(frames) < 2:
        gray = gray_from_bgr(frames[0])
        return np.ones_like(gray, dtype=np.float32), np.zeros_like(gray, dtype=np.float32)
    arr = np.stack(frames, axis=0).astype(np.float32)
    std_map = arr.std(axis=0).mean(axis=2).astype(np.float32)
    _, std_unit_map = aggregate_map_to_units(std_map, unit_width, unit_height)
    stable = np.exp(-std_unit_map / max(float(stable_variation_scale), 1e-6)).astype(np.float32)
    return stable, std_map


def canny_edge_map(gray, low=40, high=120):
    blur = cv2.GaussianBlur(gray.astype(np.uint8), (3, 3), 0)
    return (cv2.Canny(blur, int(low), int(high)) > 0).astype(np.float32)


def calculate_noise_metric(frames, args):
    # 与 inference_lens_multi_metric_results.py 中 noise 分支保持一致：
    # all_grays = [gray_from_bgr(seed)] + [gray_from_bgr(neighbor) ...]，
    # 对所有帧的高频残差取均值后再计算 noise_map/noise_score。
    grays = [gray_from_bgr(f) for f in frames]
    k = ensure_odd_kernel(args.noise_blur_ksize, 3)
    maps = []
    for g in grays:
        low = cv2.GaussianBlur(g, (k, k), 0)
        maps.append(np.abs(g - low))
    residual = np.mean(np.stack(maps, axis=0), axis=0).astype(np.float32)
    # Low-light prior: when illumination is insufficient, sensor noise probability rises.
    # Old behavior was noise_dark_weight=0.5, i.e. max dark amplification ~=1.5.
    dark_weight = float(getattr(args, 'noise_dark_weight', 0.5))
    dark_scale = float(getattr(args, 'noise_dark_scale', 60.0))
    dark_amp = 1.0 + dark_weight * np.exp(-grays[0] / max(dark_scale, 1e-6))
    noise_map = np.clip(residual * dark_amp / 20.0, 0.0, 1.0).astype(np.float32)
    unit_noise, heat = aggregate_map_to_units(noise_map, args.unit_width, args.unit_height)
    score, kcnt = top_percent_mean(unit_noise, max(args.top_unit_percent, 5.0), largest=True)
    return {'noise_score': 100.0 * score, 'noise_response': float(noise_map.mean()), 'noise_heat': heat, 'noise_top_unit_count': kcnt}


def calculate_black_occlusion_metric(frames, args):
    # 与 inference_lens_multi_metric_results.py 中 black_occlusion 分支保持一致：
    # temporal_variation_map(seed, neighbor_images) -> unit 聚合 -> exp 稳定度
    # -> 接近黑色亮度 -> top units。
    seed = frames[0]
    gray = gray_from_bgr(seed)
    if len(frames) > 1:
        pix_var = temporal_variation_map(seed, frames[1:], reduce=args.temporal_reduce)
        _, variation_map = aggregate_map_to_units(pix_var, args.unit_width, args.unit_height)
        stable_map = np.exp(-variation_map / max(float(args.stable_variation_scale), 1e-6)).astype(np.float32)
        _, stable_map = aggregate_map_to_units(stable_map, args.unit_width, args.unit_height)
    else:
        # multi_metric 在没有 neighbor_images 时将 stable_map 置 0；
        # 这里保持同样行为，避免单帧黑色区域被误当成“安装遮挡”。
        stable_map = np.zeros_like(gray, dtype=np.float32)
    black_degree = np.exp(-gray / max(float(args.black_luma_scale), 1e-6)).astype(np.float32)
    black_map = stable_map * black_degree
    unit_black, heat = aggregate_map_to_units(black_map, args.unit_width, args.unit_height)
    score, kcnt = top_percent_mean(unit_black, args.top_unit_percent, largest=True)
    return {'black_occlusion_score': 100.0 * score, 'black_response': float(black_map.mean()), 'black_heat': heat, 'black_top_unit_count': kcnt}


def fft_bandpass_response(gray, inner_ratio=0.04, outer_ratio=0.28):
    gray = gray.astype(np.float32)
    h, w = gray.shape
    f = np.fft.fftshift(np.fft.fft2(gray - gray.mean()))
    yy, xx = np.ogrid[:h, :w]
    cy, cx = h // 2, w // 2
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    maxr = max(1.0, math.sqrt(cy ** 2 + cx ** 2))
    mask = (rr >= inner_ratio * maxr) & (rr <= outer_ratio * maxr)
    band = np.zeros_like(f)
    band[mask] = f[mask]
    rec = np.abs(np.fft.ifft2(np.fft.ifftshift(band))).astype(np.float32)
    return normalize_01(rec, 0, np.percentile(rec, 98) if rec.size else 1.0)


def make_line_kernels(length=21, width=3):
    length = ensure_odd_kernel(length, 3)
    width = max(1, int(width))
    kernels = []
    kh = np.zeros((width, length), dtype=np.uint8); kh[:, :] = 1; kernels.append(kh)
    kv = np.zeros((length, width), dtype=np.uint8); kv[:, :] = 1; kernels.append(kv)
    kd1 = np.zeros((length, length), dtype=np.uint8); cv2.line(kd1, (0, 0), (length - 1, length - 1), 1, width); kernels.append(kd1)
    kd2 = np.zeros((length, length), dtype=np.uint8); cv2.line(kd2, (0, length - 1), (length - 1, 0), 1, width); kernels.append(kd2)
    return kernels


def calculate_dust_metric(frames, args):
    # 时域: 附着物固定在图像坐标，低 std 更可信。
    # 空域: 灰尘近似小颗粒/椒盐/软斑点，用 black-hat/top-hat + LoG/DoG 响应。
    # 频域: 用中高频 band-pass 强调小尺度固定纹理。
    seed = frames[0]
    gray = gray_from_bgr(seed)
    stable, std_map = stable_map_from_stack(frames, args.unit_width, args.unit_height, args.dust_stable_scale)
    spot_k = ensure_odd_kernel(args.dust_spot_ksize, 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (spot_k, spot_k))
    blackhat = cv2.morphologyEx(gray.astype(np.uint8), cv2.MORPH_BLACKHAT, kernel).astype(np.float32)
    tophat = cv2.morphologyEx(gray.astype(np.uint8), cv2.MORPH_TOPHAT, kernel).astype(np.float32)
    spot = np.maximum(blackhat, args.dust_bright_weight * tophat)
    blur_small = cv2.GaussianBlur(gray, (ensure_odd_kernel(args.dust_log_ksize, 3), ensure_odd_kernel(args.dust_log_ksize, 3)), 0)
    blur_large = cv2.GaussianBlur(gray, (ensure_odd_kernel(args.dust_bg_ksize, 7), ensure_odd_kernel(args.dust_bg_ksize, 7)), 0)
    dog = np.abs(blur_small - blur_large)
    freq = fft_bandpass_response(gray, args.dust_fft_inner, args.dust_fft_outer)
    spot_norm = np.maximum(normalize_01(spot, 0, max(1.0, np.percentile(spot, 98))), normalize_01(dog, 0, max(1.0, np.percentile(dog, 98))))
    edge = canny_edge_map(gray, args.edge_low, args.edge_high)
    anti_long_edge = 1.0 - np.clip(cv2.GaussianBlur(edge, (15, 15), 0) * 1.5, 0.0, 0.65)
    dust_map = np.clip(stable * spot_norm * (0.6 + 0.4 * freq) * anti_long_edge, 0.0, 1.0).astype(np.float32)
    unit, heat = aggregate_map_to_units(dust_map, args.unit_width, args.unit_height)
    score, kcnt = top_percent_mean(unit, args.top_unit_percent, largest=True)
    return {
        'dust_score': 100.0 * score,
        'dust_temporal_stable': float(stable.mean()),
        'dust_spot_response': float(spot_norm.mean()),
        'dust_freq_response': float(freq.mean()),
        'dust_heat': heat,
        'dust_std_map': std_map,
        'dust_top_unit_count': kcnt,
    }


def calculate_hair_metric(frames, args):
    # 时域: 毛发固定；空域: 细长暗线/虚影；频域: 方向性/结构张量一致性 + 边缘持久性。
    seed = frames[0]
    gray = gray_from_bgr(seed)
    stable, std_map = stable_map_from_stack(frames, args.unit_width, args.unit_height, args.hair_stable_scale)
    line_len = ensure_odd_kernel(args.hair_line_length, 5)
    line_responses = []
    for k in make_line_kernels(line_len, args.hair_line_width):
        bh = cv2.morphologyEx(gray.astype(np.uint8), cv2.MORPH_BLACKHAT, k).astype(np.float32)
        line_responses.append(bh)
    line = np.max(np.stack(line_responses, axis=0), axis=0)
    line_norm = normalize_01(line, 0, max(1.0, np.percentile(line, 98)))

    grays = [gray_from_bgr(f) for f in frames]
    edge_stack = np.stack([canny_edge_map(g, args.edge_low, args.edge_high) for g in grays], axis=0)
    edge_persistence = edge_stack.mean(axis=0).astype(np.float32)
    edge_persistence = cv2.GaussianBlur(edge_persistence, (7, 7), 0)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    jxx = cv2.GaussianBlur(gx * gx, (15, 15), 0)
    jyy = cv2.GaussianBlur(gy * gy, (15, 15), 0)
    jxy = cv2.GaussianBlur(gx * gy, (15, 15), 0)
    coherence = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy ** 2) / (jxx + jyy + 1e-6)
    coherence = np.clip(coherence, 0.0, 1.0).astype(np.float32)
    band = fft_bandpass_response(gray, args.hair_fft_inner, args.hair_fft_outer)
    freq_dir = np.clip(0.5 * coherence + 0.5 * band, 0.0, 1.0)

    hair_map = np.clip(stable * np.sqrt(np.clip(line_norm * (0.35 + 0.65 * edge_persistence), 0, 1)) * (0.4 + 0.6 * freq_dir), 0.0, 1.0).astype(np.float32)
    unit, heat = aggregate_map_to_units(hair_map, args.unit_width, args.unit_height)
    score, kcnt = top_percent_mean(unit, max(args.top_unit_percent, 3.0), largest=True)
    return {
        'hair_score': 100.0 * score,
        'hair_temporal_stable': float(stable.mean()),
        'hair_line_response': float(line_norm.mean()),
        'hair_freq_response': float(freq_dir.mean()),
        'hair_edge_persistence': float(edge_persistence.mean()),
        'hair_heat': heat,
        'hair_std_map': std_map,
        'hair_top_unit_count': kcnt,
    }


def calculate_all_metrics(metric_stacks, args):
    start = time()
    noise = calculate_noise_metric(metric_stacks['noise'], args)
    black = calculate_black_occlusion_metric(metric_stacks['black'], args)
    dust = calculate_dust_metric(metric_stacks['dust'], args)
    hair = calculate_hair_metric(metric_stacks['hair'], args)
    result = {}
    result.update(noise); result.update(black); result.update(dust); result.update(hair)
    result['valid_noise_frames'] = len(metric_stacks['noise'])
    result['valid_black_frames'] = len(metric_stacks['black'])
    result['valid_dust_frames'] = len(metric_stacks['dust'])
    result['valid_hair_frames'] = len(metric_stacks['hair'])
    result['elapsed_time'] = time() - start
    return result


def colorize_01(map01, colormap=cv2.COLORMAP_JET):
    return cv2.applyColorMap(np.clip(map01 * 255.0, 0, 255).astype(np.uint8), colormap)


def put_label(img, text, y=26):
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)


def overlay_heat(base_bgr, heat01, alpha=0.55):
    base = np.clip(base_bgr, 0, 255).astype(np.uint8)
    heat = np.clip(heat01, 0, 1).astype(np.float32)
    color = colorize_01(heat)
    blended = cv2.addWeighted(base, 1.0 - alpha, color, alpha, 0)
    out = base.copy()
    mask = heat > 0
    out[mask] = blended[mask]
    return out


def save_visualization(img_path, seed_img, result, out_dir, index, scene, alpha=0.55):
    os.makedirs(out_dir, exist_ok=True)
    base = np.clip(seed_img, 0, 255).astype(np.uint8)
    panels = [
        base.copy(),
        overlay_heat(base, result['noise_heat'], alpha),
        overlay_heat(base, result['black_heat'], alpha),
        overlay_heat(base, result['dust_heat'], alpha),
        overlay_heat(base, result['hair_heat'], alpha),
    ]
    labels = [
        'seed',
        f'noise={format_float(result["noise_score"])}',
        f'black={format_float(result["black_occlusion_score"])}',
        f'dust={format_float(result["dust_score"])}',
        f'hair={format_float(result["hair_score"])}',
    ]
    labeled = []
    for p, label in zip(panels, labels):
        p = p.copy(); put_label(p, label); labeled.append(p)
    blank = np.zeros_like(base)
    put_label(blank, 'per-branch abnormal scores, no composite score', 26)
    put_label(blank, f'n/b/d/h frames={result["valid_noise_frames"]}/{result["valid_black_frames"]}/{result["valid_dust_frames"]}/{result["valid_hair_frames"]}', 54)
    put_label(blank, f'dust: stable={format_float(result["dust_temporal_stable"])} spot={format_float(result["dust_spot_response"])} freq={format_float(result["dust_freq_response"])}', 82)
    put_label(blank, f'hair: stable={format_float(result["hair_temporal_stable"])} line={format_float(result["hair_line_response"])} freq={format_float(result["hair_freq_response"])}', 110)
    labeled.append(blank)
    row1 = np.concatenate(labeled[:3], axis=1)
    row2 = np.concatenate(labeled[3:6], axis=1)
    canvas = np.concatenate([row1, row2], axis=0)
    scene_part = make_safe_name(scene) if scene else 'no_scene'
    stem = make_safe_name(os.path.splitext(os.path.basename(img_path))[0])
    out_path = os.path.join(out_dir, f'{index:06d}_{scene_part}_{stem}_temporal_multi.jpg')
    cv2.imwrite(out_path, canvas)
    return out_path


def update_scene_stat(scene_stats, scene, score, path):
    if not scene or math.isnan(float(score)):
        return
    stat = scene_stats[scene]
    stat['sum'] += float(score)
    stat['count'] += 1
    if stat['max_score'] is None or score > stat['max_score']:
        stat['max_score'], stat['max_path'] = score, path
    if stat['min_score'] is None or score < stat['min_score']:
        stat['min_score'], stat['min_path'] = score, path


def scene_separability_msgs(scene_stats_by_metric, normal_scene='normal'):
    msgs = ['Normal-vs-other separability by metric; larger means more abnormal.']
    for metric_name, scene_stats in scene_stats_by_metric.items():
        msgs.append(f'[{metric_name}]')
        normal_key = None
        for s in scene_stats:
            if s.lower() == normal_scene.lower():
                normal_key = s
                break
        if normal_key is None:
            msgs.append(f'  Scene [{normal_scene}] not found, skip.')
            continue
        total, sep = 0, 0
        n = scene_stats[normal_key]
        for s, stat in scene_stats.items():
            if s == normal_key:
                continue
            total += 1
            overlap_min = max(n['min_score'], stat['min_score'])
            overlap_max = min(n['max_score'], stat['max_score'])
            ok = overlap_min > overlap_max
            sep += int(ok)
            if ok:
                worse = normal_key if n['min_score'] > stat['max_score'] else s
                msgs.append(f'  [{normal_key}] vs [{s}]: SEPARABLE, normal=[{format_float(n["min_score"])}, {format_float(n["max_score"])}], scene=[{format_float(stat["min_score"])}, {format_float(stat["max_score"])}], worse=[{worse}]')
            else:
                msgs.append(f'  [{normal_key}] vs [{s}]: NOT separable, normal=[{format_float(n["min_score"])}, {format_float(n["max_score"])}], scene=[{format_float(stat["min_score"])}, {format_float(stat["max_score"])}], overlap=[{format_float(overlap_min)}, {format_float(overlap_max)}]')
        ratio = sep / total if total else 0.0
        msgs.append(f'  Summary: {sep}/{total} ({format_float(ratio)}) normal-other scene pairs are separable.')
    return msgs


def split_aliases(text):
    return [x.strip().lower() for x in str(text).split(',') if x.strip()]


def scene_matches_aliases(scene, aliases):
    if not scene:
        return False
    scene = scene.lower()
    return any(alias in scene for alias in aliases)


def build_best_binary_confusion(valid):
    scores = sorted(set(x[1] for x in valid))
    candidates = [scores[0] - 1e-6]
    candidates += [(scores[i] + scores[i + 1]) / 2.0 for i in range(len(scores) - 1)]
    candidates += [scores[-1] + 1e-6]

    best = None
    for th in candidates:
        tp = tn = fp = fn = 0
        for is_positive, score, _, _ in valid:
            pred_positive = score > th
            if is_positive and pred_positive:
                tp += 1
            elif is_positive and not pred_positive:
                fn += 1
            elif (not is_positive) and pred_positive:
                fp += 1
            else:
                tn += 1
        total = tp + tn + fp + fn
        acc = (tp + tn) / float(total) if total else 0.0
        negative_recall = tn / float(tn + fp) if (tn + fp) else 0.0
        positive_recall = tp / float(tp + fn) if (tp + fn) else 0.0
        balanced_acc = 0.5 * (negative_recall + positive_recall)
        # 独立问题度量通常类别不均衡，优先 balanced_acc，再看 accuracy。
        # 最后偏向稍高阈值，减少误报。
        key = (balanced_acc, acc, th)
        if best is None or key > best[0]:
            best = (key, th, tp, tn, fp, fn, acc, balanced_acc)
    return best


def format_confusion_matrix_msgs(metric_name, positive_name, valid):
    msgs = [f'[{metric_name}] independent binary problem: {positive_name} vs not_{positive_name}']
    if not valid:
        msgs.append('  No labeled valid samples, skip.')
        return msgs
    has_pos = any(x[0] for x in valid)
    has_neg = any(not x[0] for x in valid)
    if not has_pos or not has_neg:
        msgs.append(f'  Need both positive and negative samples, got positive={has_pos}, negative={has_neg}; skip.')
        return msgs

    _, th, tp, tn, fp, fn, acc, balanced_acc = build_best_binary_confusion(valid)
    precision = tp / float(tp + fp) if (tp + fp) else 0.0
    recall = tp / float(tp + fn) if (tp + fn) else 0.0
    f1 = 2.0 * precision * recall / float(precision + recall) if (precision + recall) else 0.0
    msgs.append(f'  threshold={format_float(th)}; rule: score > threshold => predict_{positive_name}')
    msgs.append('  confusion_matrix:')
    msgs.append(f'                    pred_not_{positive_name}  pred_{positive_name}')
    msgs.append(f'    true_not_{positive_name:<12s} {tn:6d}        {fp:6d}')
    msgs.append(f'    true_{positive_name:<16s} {fn:6d}        {tp:6d}')
    msgs.append(
        f'  accuracy={format_float(acc)}, balanced_acc={format_float(balanced_acc)}, '
        f'precision_{positive_name}={format_float(precision)}, '
        f'recall_{positive_name}={format_float(recall)}, f1_{positive_name}={format_float(f1)}'
    )
    return msgs


def independent_problem_confusion_matrix_msgs(sample_scores_by_metric, args):
    """Build per-defect independent confusion matrices.

    每个 seed 同时参与多个相互独立的问题判断：
      - noise_score: 当前 seed 是否属于噪声问题
      - black_occlusion_score: 当前 seed 是否属于安装遮挡问题
      - dust_score: 当前 seed 是否属于灰尘附着问题
      - hair_score: 当前 seed 是否属于毛发附着问题

    对某一个问题来说，其他问题场景不是 abnormal 正类，而是该问题的
    negative 类。例如 hair_score 的正类只看 hair/毛发场景，dust/noise/normal
    都属于 not_hair。
    """
    defect_specs = [
        ('noise_score', 'noise', split_aliases(args.noise_positive_scenes)),
        ('black_occlusion_score', 'black_occlusion', split_aliases(args.black_positive_scenes)),
        ('dust_score', 'dust', split_aliases(args.dust_positive_scenes)),
        ('hair_score', 'hair', split_aliases(args.hair_positive_scenes)),
    ]
    msgs = [
        'Independent per-defect binary confusion matrices; all scores are abnormal scores.',
        'For each metric, only its own scene aliases are positive; normal and other defect scenes are negative.',
    ]
    for metric_name, positive_name, aliases in defect_specs:
        samples = sample_scores_by_metric.get(metric_name, [])
        valid = []
        for scene, path, score in samples:
            if not scene or math.isnan(float(score)):
                continue
            is_positive = scene_matches_aliases(scene, aliases)
            valid.append((is_positive, float(score), scene, path))
        msgs.extend(format_confusion_matrix_msgs(metric_name, positive_name, valid))

    return msgs



def any_branch_alarm_confusion_matrix_msgs(sample_results, args):
    """Normal-vs-abnormal alarm by OR-ing independent defect branches.

    No composite score is used. Each defect metric gets its own threshold from
    its corresponding independent binary problem, then final alarm is:
        noise_hit OR black_hit OR dust_hit OR hair_hit
    """
    defect_specs = [
        ('noise_score', 'noise', split_aliases(args.noise_positive_scenes)),
        ('black_occlusion_score', 'black_occlusion', split_aliases(args.black_positive_scenes)),
        ('dust_score', 'dust', split_aliases(args.dust_positive_scenes)),
        ('hair_score', 'hair', split_aliases(args.hair_positive_scenes)),
    ]
    msgs = [
        'Any-branch abnormal alarm matrix; no composite score is used.',
        'Rule: alarm = noise_hit OR black_occlusion_hit OR dust_hit OR hair_hit.',
    ]
    thresholds = {}
    for metric_name, positive_name, aliases in defect_specs:
        valid = []
        for scene, path, res in sample_results:
            if not scene:
                continue
            score = float(res.get(metric_name, float('nan')))
            if math.isnan(score):
                continue
            valid.append((scene_matches_aliases(scene, aliases), score, scene, path))
        if valid and any(x[0] for x in valid) and any(not x[0] for x in valid):
            _, th, tp, tn, fp, fn, acc, balanced_acc = build_best_binary_confusion(valid)
            thresholds[metric_name] = th
            msgs.append(
                f'  branch_threshold[{metric_name}/{positive_name}]={format_float(th)} '
                f'(branch_balanced_acc={format_float(balanced_acc)}, branch_acc={format_float(acc)})'
            )
        else:
            thresholds[metric_name] = float('inf')
            msgs.append(f'  branch_threshold[{metric_name}/{positive_name}]=inf (insufficient labels)')

    tp = tn = fp = fn = 0
    hit_counter = {name: 0 for _, name, _ in defect_specs}
    false_alarm_paths = []
    miss_paths = []
    for scene, path, res in sample_results:
        if not scene:
            continue
        true_alarm = scene.lower() != args.normal_scene.lower()
        hits = []
        for metric_name, positive_name, _ in defect_specs:
            score = float(res.get(metric_name, float('nan')))
            th = thresholds.get(metric_name, float('inf'))
            if not math.isnan(score) and not math.isinf(th) and score > th:
                hits.append(positive_name)
                hit_counter[positive_name] += 1
        pred_alarm = bool(hits)
        if true_alarm and pred_alarm:
            tp += 1
        elif true_alarm and not pred_alarm:
            fn += 1
            miss_paths.append((scene, path))
        elif (not true_alarm) and pred_alarm:
            fp += 1
            false_alarm_paths.append((scene, path, ','.join(hits)))
        else:
            tn += 1

    total = tp + tn + fp + fn
    acc = (tp + tn) / float(total) if total else 0.0
    normal_recall = tn / float(tn + fp) if (tn + fp) else 0.0
    abnormal_recall = tp / float(tp + fn) if (tp + fn) else 0.0
    precision_alarm = tp / float(tp + fp) if (tp + fp) else 0.0
    f1_alarm = 2.0 * precision_alarm * abnormal_recall / float(precision_alarm + abnormal_recall) if (precision_alarm + abnormal_recall) else 0.0
    msgs.append('  confusion_matrix:')
    msgs.append('                         pred_normal  pred_alarm')
    msgs.append(f'    true_normal              {tn:6d}      {fp:6d}')
    msgs.append(f'    true_abnormal            {fn:6d}      {tp:6d}')
    msgs.append(
        f'  total={total}, accuracy={format_float(acc)}, normal_recall={format_float(normal_recall)}, '
        f'abnormal_recall={format_float(abnormal_recall)}, precision_alarm={format_float(precision_alarm)}, '
        f'f1_alarm={format_float(f1_alarm)}'
    )
    msgs.append('  branch_hit_count: ' + ', '.join(f'{k}={v}' for k, v in hit_counter.items()))
    if false_alarm_paths:
        msgs.append('  false_alarm_examples:')
        for scene, path, hits in false_alarm_paths[:10]:
            msgs.append(f'    [{scene}] hits={hits}: {path}')
    if miss_paths:
        msgs.append('  miss_examples:')
        for scene, path in miss_paths[:10]:
            msgs.append(f'    [{scene}] {path}')
    return msgs

def scene_to_multiclass_label(scene, args):
    if not scene:
        return None
    scene_l = scene.lower()
    if scene_l == args.normal_scene.lower():
        return 'normal'
    if scene_matches_aliases(scene, split_aliases(args.noise_positive_scenes)):
        return 'noise'
    if scene_matches_aliases(scene, split_aliases(args.black_positive_scenes)):
        return 'black_occlusion'
    if scene_matches_aliases(scene, split_aliases(args.dust_positive_scenes)):
        return 'dust'
    if scene_matches_aliases(scene, split_aliases(args.hair_positive_scenes)):
        return 'hair'
    return scene_l


def multiclass_confusion_matrix_msgs(sample_results, args):
    """Build one unified multi-class confusion matrix.

    每个 seed 只落到一个最终预测类别：
      1. 先为 noise/black/dust/hair 四个 score 各自从当前验证集学习一个
         one-vs-rest 阈值；
      2. 对单个 seed，若所有 defect score 都没有超过各自阈值，则预测 normal；
      3. 若多个 defect score 同时超过阈值，取相对超阈值幅度最大的类别。

    这样输出的是一个统一二维混淆矩阵，而不是多个独立二分类矩阵。
    """
    defect_specs = [
        ('noise', 'noise_score'),
        ('black_occlusion', 'black_occlusion_score'),
        ('dust', 'dust_score'),
        ('hair', 'hair_score'),
    ]
    canonical_labels = ['normal', 'noise', 'black_occlusion', 'dust', 'hair']
    labeled = []
    for scene, path, result in sample_results:
        true_label = scene_to_multiclass_label(scene, args)
        if true_label is None:
            continue
        labeled.append((true_label, path, result))

    msgs = [
        'Unified multi-class confusion matrix:',
        '  true class: scene marker mapped to normal/noise/black_occlusion/dust/hair by aliases.',
        '  pred class: normal if no defect score exceeds its threshold; otherwise choose the defect with largest relative over-threshold margin.',
        '  threshold search: each defect score uses one-vs-rest threshold on current validation set; all scores larger means more suspicious.',
    ]
    if not labeled:
        msgs.append('  No labeled valid samples, skip.')
        return msgs

    thresholds = {}
    for label, metric_name in defect_specs:
        valid = []
        for true_label, path, result in labeled:
            score = float(result[metric_name])
            if math.isnan(score):
                continue
            valid.append((true_label == label, score, true_label, path))
        has_pos = any(x[0] for x in valid)
        has_neg = any(not x[0] for x in valid)
        if valid and has_pos and has_neg:
            _, th, tp, tn, fp, fn, acc, balanced_acc = build_best_binary_confusion(valid)
            thresholds[label] = th
            msgs.append(
                f'  threshold[{label}/{metric_name}]={format_float(th)} '
                f'(one-vs-rest balanced_acc={format_float(balanced_acc)}, acc={format_float(acc)})'
            )
        else:
            thresholds[label] = float('inf')
            msgs.append(f'  threshold[{label}/{metric_name}]=inf, because positive/negative samples are incomplete.')

    labels = list(canonical_labels)
    for true_label, _, _ in labeled:
        if true_label not in labels:
            labels.append(true_label)

    matrix = {r: {c: 0 for c in labels} for r in labels}
    total = correct = 0
    for true_label, path, result in labeled:
        best_label, best_margin = 'normal', 0.0
        for label, metric_name in defect_specs:
            th = thresholds.get(label, float('inf'))
            score = float(result[metric_name])
            if math.isnan(score) or math.isinf(th) or score <= th:
                continue
            margin = (score - th) / max(abs(th), 1.0)
            if margin > best_margin:
                best_label, best_margin = label, margin
        pred_label = best_label
        if pred_label not in matrix[true_label]:
            for r in labels:
                matrix[r][pred_label] = 0
            labels.append(pred_label)
            matrix[pred_label] = {c: 0 for c in labels}
        matrix[true_label][pred_label] += 1
        total += 1
        correct += int(true_label == pred_label)

    width = max(14, max(len(x) for x in labels) + 2)
    msgs.append('  confusion_matrix:')
    msgs.append(' ' * width + ''.join(f'{("pred_" + c):>{width}s}' for c in labels))
    for r in labels:
        msgs.append(f'{("true_" + r):>{width}s}' + ''.join(f'{matrix[r].get(c, 0):>{width}d}' for c in labels))
    acc = correct / float(total) if total else 0.0
    recalls = []
    for r in labels:
        row_sum = sum(matrix[r].values())
        if row_sum > 0:
            recalls.append(matrix[r].get(r, 0) / float(row_sum))
    balanced_acc = sum(recalls) / float(len(recalls)) if recalls else 0.0
    msgs.append(f'  total={total}, accuracy={format_float(acc)}, macro_recall/balanced_acc={format_float(balanced_acc)}')
    return msgs


def main():
    parser = argparse.ArgumentParser(description='Lens abnormality validation with temporal multi-metric noise/black/dust/hair hypotheses.')
    parser.add_argument('-t', '--target', type=str, default=None, help='input image/folder path.')
    parser.add_argument('--target_txt', type=str, default=None, help='txt file containing seed image paths and scene markers.')
    parser.add_argument('-m', '--metric_name', type=str, default='lens_temporal_multi_metric', help='metric name used in logs/result file name.')
    parser.add_argument('--metric_mode', type=str, default='NR', help='kept for compatibility; only NR is supported.')
    parser.add_argument('--device', type=str, default='cpu', help='kept for compatibility; CPU/OpenCV only.')
    parser.add_argument('--save_txt_dir', type=str, default=None, help='directory to save txt results and visualizations.')
    parser.add_argument('--save_file', type=str, default=None, help='optional CSV output path.')
    parser.add_argument('--no_vis', action='store_true', help='disable visualization saving. Default saves when --save_txt_dir is set.')
    parser.add_argument('--sequence_mode', type=str, default='even_span', choices=['even_span', 'previous_next'], help='global sequence mode.')
    parser.add_argument('--noise_sequence_mode', type=str, default='previous_next', choices=['inherit', 'even_span', 'previous_next'], help='noise metric sequence mode; default previous_next to match multi_metric behavior.')
    parser.add_argument('--black_sequence_mode', type=str, default='previous_next', choices=['inherit', 'even_span', 'previous_next'], help='black occlusion sequence mode; default continuous nearby frames.')
    parser.add_argument('--dust_sequence_mode', type=str, default='even_span', choices=['inherit', 'even_span', 'previous_next'], help='dust metric sequence mode; default wide even-span frames.')
    parser.add_argument('--hair_sequence_mode', type=str, default='even_span', choices=['inherit', 'even_span', 'previous_next'], help='hair metric sequence mode; default wide even-span frames.')
    parser.add_argument('--sample_count', type=int, default=7, help='global M for even_span.')
    parser.add_argument('--noise_sample_count', type=int, default=0, help='override M for noise; <=0 uses --sample_count.')
    parser.add_argument('--black_sample_count', type=int, default=0, help='override M for black occlusion; <=0 uses --sample_count.')
    parser.add_argument('--dust_sample_count', type=int, default=0, help='override M for dust; <=0 uses --sample_count.')
    parser.add_argument('--hair_sample_count', type=int, default=0, help='override M for hair; <=0 uses --sample_count.')
    parser.add_argument('--window', type=int, default=3, help='previous/next N for previous_next mode.')
    parser.add_argument('--resize_width', type=int, default=640, help='resize width; <=0 disables.')
    parser.add_argument('--unit_width', type=int, default=4, help='super-pixel unit width.')
    parser.add_argument('--unit_height', type=int, default=3, help='super-pixel unit height.')
    parser.add_argument('--top_unit_percent', type=float, default=1.0, help='top suspicious unit percentage used for scores.')
    parser.add_argument('--temporal_reduce', type=str, default='max', choices=['max', 'p75', 'median', 'mean'], help='temporal reduce used by black metric.')
    parser.add_argument('--stable_variation_scale', type=float, default=3.0, help='black metric exp(-variation/scale).')
    parser.add_argument('--black_luma_scale', type=float, default=35.0, help='black occlusion luminance scale.')
    parser.add_argument('--noise_blur_ksize', type=int, default=5, help='small blur kernel for noise residual.')
    parser.add_argument('--noise_dark_weight', type=float, default=0.5, help='low-light amplification weight for noise; old/default max amplification is 1+0.5.')
    parser.add_argument('--noise_dark_scale', type=float, default=60.0, help='luma decay scale for low-light noise amplification.')
    parser.add_argument('--edge_low', type=float, default=40.0, help='Canny low threshold.')
    parser.add_argument('--edge_high', type=float, default=120.0, help='Canny high threshold.')
    parser.add_argument('--dust_stable_scale', type=float, default=8.0, help='dust temporal std stability scale; larger is more tolerant.')
    parser.add_argument('--dust_spot_ksize', type=int, default=7, help='morphological kernel for dust salt/pepper spots.')
    parser.add_argument('--dust_log_ksize', type=int, default=5, help='small Gaussian for dust DoG.')
    parser.add_argument('--dust_bg_ksize', type=int, default=31, help='large Gaussian for dust DoG/background.')
    parser.add_argument('--dust_bright_weight', type=float, default=0.6, help='weight for bright tophat vs dark blackhat in dust response.')
    parser.add_argument('--dust_fft_inner', type=float, default=0.04, help='inner radius ratio of FFT band-pass for dust.')
    parser.add_argument('--dust_fft_outer', type=float, default=0.32, help='outer radius ratio of FFT band-pass for dust.')
    parser.add_argument('--hair_stable_scale', type=float, default=10.0, help='hair temporal std stability scale; larger is more tolerant.')
    parser.add_argument('--hair_line_length', type=int, default=25, help='elongated morphology kernel length for hair/vague-shadow line.')
    parser.add_argument('--hair_line_width', type=int, default=3, help='elongated morphology kernel width.')
    parser.add_argument('--hair_fft_inner', type=float, default=0.015, help='inner radius ratio of FFT band-pass for hair.')
    parser.add_argument('--hair_fft_outer', type=float, default=0.18, help='outer radius ratio of FFT band-pass for hair.')
    parser.add_argument('--score_metric', type=str, default='dust_score', choices=SCORE_COLUMNS, help='metric used only for progress/overall average; final alarm uses per-branch OR logic.')
    parser.add_argument('--normal_scene', type=str, default='normal', help='scene name used as normal class.')
    parser.add_argument('--noise_positive_scenes', type=str, default='noise,noisy,low_light,dark,weak_light,噪声,暗光,弱光', help='comma-separated scene-name aliases treated as positive for noise_score.')
    parser.add_argument('--black_positive_scenes', type=str, default='black,occlusion,block,cover,install,安装遮挡,遮挡,黑屏', help='comma-separated scene-name aliases treated as positive for black_occlusion_score.')
    parser.add_argument('--dust_positive_scenes', type=str, default='dust,dirty,灰尘,尘,脏污,污渍', help='comma-separated scene-name aliases treated as positive for dust_score.')
    parser.add_argument('--hair_positive_scenes', type=str, default='hair,毛发,头发', help='comma-separated scene-name aliases treated as positive for hair_score.')
    parser.add_argument('--heatmap_alpha', type=float, default=0.55, help='overlay alpha for heatmaps.')
    args = parser.parse_args()

    if args.metric_mode != 'NR':
        raise ValueError('This script only supports NR mode.')
    if args.target is None and args.target_txt is None:
        raise ValueError('Please specify --target or --target_txt.')

    input_paths, input_scenes = get_input_paths(args.target, args.target_txt)
    if not input_paths:
        raise ValueError('No input images found.')
    resize_width = args.resize_width if args.resize_width > 0 else None

    metric_sequences, all_load_paths, modes, counts = build_metric_sequences(input_paths, args)
    print('Loading seed images and same-directory temporal sequence frames...')
    print(f'Seed images: {len(input_paths)}')
    print(f'Images to load including metric-specific frames: {len(all_load_paths)}')
    print(f'Sequence modes: {modes}; sample counts: {counts}')
    cache = {}
    for p in tqdm(all_load_paths, total=len(all_load_paths), unit='image'):
        cache[os.path.abspath(p)] = imread_image_resize(p, resize_width)

    save_txt_path, vis_dir, txt_f = None, None, None
    if args.save_txt_dir:
        save_txt_path = build_auto_txt_save_path(args.save_txt_dir, args.metric_name)
        if not args.no_vis:
            vis_dir = build_vis_save_dir(save_txt_path)
        txt_f = open(save_txt_path, 'w', encoding='utf-8')
        txt_f.write(f'metric_name: {args.metric_name}\nmetric_mode: NR\n')
        txt_f.write(f'target: {args.target}\ntarget_txt: {args.target_txt}\ndevice: {args.device}\n')
        txt_f.write('score_direction: all scores are abnormal scores; larger means more suspicious/worse\n')
        txt_f.write(f'score_metric_for_overall_progress: {args.score_metric}\n')
        txt_f.write('method:\n')
        txt_f.write('  sequence: temporal_stack style; each anomaly type may use previous_next or even_span sampling.\n')
        txt_f.write('  noise: same as multi_metric noise branch: average high-frequency residual over seed and neighbor frames.\n')
        txt_f.write('  black_occlusion: same as multi_metric black branch: temporal-stable + near-black; no temporal neighbor means zero stable prior.\n')
        txt_f.write('  dust: temporal stability + salt/pepper morphology + DoG + FFT band-pass response.\n')
        txt_f.write('  hair: temporal stability + elongated dark-line morphology + persistent edge + directional/frequency response.\n')
        for k in vars(args):
            txt_f.write(f'{k}: {getattr(args, k)}\n')
        txt_f.write(f'resolved_sequence_modes: {modes}\nresolved_sample_counts: {counts}\n')
        txt_f.write(f'seed_count: {len(input_paths)}\nloaded_image_count: {len(all_load_paths)}\nvis_dir: {vis_dir}\n')
        txt_f.write(f'time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')
        txt_f.write('scene\timage\t' + '\t'.join(RESULT_COLUMNS) + '\ttime\tvisualization\n')

    sf, writer = None, None
    if args.save_file:
        sf = open(args.save_file, 'w', newline='')
        writer = csv.writer(sf)
        writer.writerow(['scene', 'image'] + RESULT_COLUMNS + ['time', 'visualization'])

    scene_stats_by_metric = {m: defaultdict(lambda: {'sum': 0.0, 'count': 0, 'max_score': None, 'max_path': None, 'min_score': None, 'min_path': None}) for m in SCORE_COLUMNS}
    sample_scores_by_metric = {m: [] for m in SCORE_COLUMNS}
    sample_results = []
    avg, count = 0.0, 0
    pbar = tqdm(total=len(input_paths), unit='image')
    for idx, img_path in enumerate(input_paths):
        scene = input_scenes[idx]
        stacks = {}
        for name in ['noise', 'black', 'dust', 'hair']:
            stacks[name] = [cache[os.path.abspath(p)] for p in metric_sequences[name][idx]]
        result = calculate_all_metrics(stacks, args)
        vis_path = ''
        if vis_dir is not None:
            vis_path = save_visualization(img_path, cache[os.path.abspath(img_path)], result, vis_dir, idx, scene, args.heatmap_alpha)

        score = float(result[args.score_metric])
        if not math.isnan(score):
            avg += score; count += 1
        for m in SCORE_COLUMNS:
            update_scene_stat(scene_stats_by_metric[m], scene, float(result[m]), img_path)
            sample_scores_by_metric[m].append((scene, img_path, float(result[m])))
        sample_results.append((scene, img_path, result))

        values = [format_float(result[c]) for c in RESULT_COLUMNS]
        elapsed = format_float(result['elapsed_time'])
        prefix = f'[{scene}] ' if scene else ''
        pbar.update(1)
        pbar.set_description(f'{prefix}{args.metric_name}/{args.score_metric} of {os.path.basename(img_path)}: {format_float(score)}')
        pbar.write(f'{prefix}{args.metric_name} {os.path.basename(img_path)}: noise={format_float(result["noise_score"])} black={format_float(result["black_occlusion_score"])} dust={format_float(result["dust_score"])} hair={format_float(result["hair_score"])} Time: {elapsed}s')

        row = [scene or '', img_path] + values + [elapsed, vis_path]
        if writer is not None:
            writer.writerow(row)
        if txt_f is not None:
            txt_f.write('\t'.join(map(str, row)) + '\n')
    pbar.close()

    avg = avg / count if count else float('nan')
    msg = f'Average {args.metric_name}/{args.score_metric} score of {args.target or args.target_txt} with {count}/{len(input_paths)} valid images is: {format_float(avg)}'
    print(msg)

    scene_msgs = []
    if scene_stats_by_metric:
        print('Scene statistics by metric:')
        for metric_name in SCORE_COLUMNS:
            scene_stats = scene_stats_by_metric[metric_name]
            scene_msgs.append(f'[{metric_name}]')
            print(f'[{metric_name}]')
            for scene, stat in scene_stats.items():
                if stat['count'] <= 0:
                    continue
                scene_avg = stat['sum'] / stat['count']
                m = f'  [{scene}] count={stat["count"]}: avg={format_float(scene_avg)}, max={format_float(stat["max_score"])} ({stat["max_path"]}), min={format_float(stat["min_score"])} ({stat["min_path"]})'
                scene_msgs.append(m); print(m)
    sep_msgs = scene_separability_msgs(scene_stats_by_metric, args.normal_scene) if scene_stats_by_metric else []
    for m in sep_msgs:
        print(m)
    alarm_msgs = any_branch_alarm_confusion_matrix_msgs(sample_results, args) if sample_results else []
    cm_msgs = multiclass_confusion_matrix_msgs(sample_results, args) if sample_results else []
    for m in cm_msgs:
        print(m)

    if txt_f is not None:
        txt_f.write('\n' + msg + '\n')
        if scene_msgs:
            txt_f.write('Scene statistics by metric:\n')
            for m in scene_msgs:
                txt_f.write(m + '\n')
        if sep_msgs:
            txt_f.write('\n')
            for m in sep_msgs:
                txt_f.write(m + '\n')
        if cm_msgs:
            txt_f.write('\n')
            for m in cm_msgs:
                txt_f.write(m + '\n')
        txt_f.close()
    if sf is not None:
        sf.close()
    if args.save_file:
        print(f'Done! CSV results are in {args.save_file}.')
    if save_txt_path:
        print(f'Done! TXT results are in {save_txt_path}.')
    if vis_dir:
        print(f'Done! Visualizations are in {vis_dir}.')
    if not args.save_file and not save_txt_path:
        print('Done!')


if __name__ == '__main__':
    main()

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
METRIC_COLUMNS = [
    'composite_score',
    'black_occlusion_score',
    'hair_score',
    'dust_score',
    'noise_score',
    'mean_temporal_variation',
    'min_unit_variation',
    'mean_luma',
    'edge_density',
    'dust_response',
    'noise_response',
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
        for line in f:
            line = line.strip()
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


def get_input_paths(input_path, input_txt=None):
    if input_txt is not None:
        return read_paths_from_txt(input_txt)
    if os.path.isfile(input_path):
        return [input_path], [None]
    paths = sorted(glob.glob(os.path.join(input_path, '*')))
    paths = [p for p in paths if is_image_file(p)]
    return paths, [None] * len(paths)


def build_auto_txt_save_path(save_txt_dir, metric_name):
    os.makedirs(save_txt_dir, exist_ok=True)
    time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    safe_metric_name = metric_name.replace(os.sep, '_')
    return os.path.join(save_txt_dir, f'{time_str}.{safe_metric_name}.txt')


def format_float(value):
    try:
        value = float(value)
    except Exception:
        return str(value)
    if math.isnan(value):
        return 'nan'
    return f'{value:.4f}'


def format_scene_prefix(scene):
    return f'[{scene}] ' if scene else ''


def natural_key(path):
    import re
    name = os.path.basename(path)
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r'(\d+)', name)]


def is_image_file(path):
    return os.path.splitext(path)[1].lower() in IMAGE_EXTENSIONS


def list_images_in_same_dir(img_path):
    img_dir = os.path.dirname(os.path.abspath(img_path))
    if not os.path.isdir(img_dir):
        return []
    paths = [os.path.join(img_dir, name) for name in os.listdir(img_dir)
             if is_image_file(name) and os.path.isfile(os.path.join(img_dir, name))]
    return sorted(paths, key=natural_key)


def build_seed_context_paths(seed_paths, window):
    seed_context_paths, all_paths, missing_seed_paths = [], set(), []
    for seed_path in seed_paths:
        seed_abs = os.path.abspath(seed_path)
        dir_images = list_images_in_same_dir(seed_abs)
        abs_to_path = {os.path.abspath(path): path for path in dir_images}
        if seed_abs not in abs_to_path:
            missing_seed_paths.append(seed_path)
            seed_context_paths.append([])
            continue
        seed_real_path = abs_to_path[seed_abs]
        seed_index = dir_images.index(seed_real_path)
        start_index = max(0, seed_index - window)
        end_index = min(len(dir_images), seed_index + window + 1)
        context_paths = [path for path in dir_images[start_index:end_index] if path != seed_real_path]
        seed_context_paths.append(context_paths)
        all_paths.add(seed_real_path)
        all_paths.update(context_paths)
    if missing_seed_paths:
        raise FileNotFoundError('Some seed images do not exist: ' + ', '.join(missing_seed_paths[:5]))
    return seed_context_paths, sorted(all_paths, key=natural_key)


def imread_image_resize(img_path, resize_width):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f'Failed to read image: {img_path}')
    if resize_width is not None and resize_width > 0 and img.shape[1] != resize_width:
        scale = resize_width / float(img.shape[1])
        resize_height = max(1, int(round(img.shape[0] * scale)))
        img = cv2.resize(img, (resize_width, resize_height), interpolation=cv2.INTER_AREA)
    return img.astype(np.float32)


def ensure_odd_kernel(kernel_size, min_value=3):
    kernel_size = int(kernel_size)
    if kernel_size < min_value:
        kernel_size = min_value
    if kernel_size % 2 == 0:
        kernel_size += 1
    return kernel_size


def normalize_01(x, low=None, high=None):
    x = x.astype(np.float32)
    if low is None:
        low = float(np.nanmin(x))
    if high is None:
        high = float(np.nanmax(x))
    if high - low < 1e-6:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - low) / (high - low), 0.0, 1.0).astype(np.float32)


def sigmoid01(x, center, scale):
    scale = max(float(scale), 1e-6)
    return (1.0 / (1.0 + np.exp(-(x - center) / scale))).astype(np.float32)


def aggregate_map_to_units(value_map, unit_width=4, unit_height=3, reduce='mean'):
    value_map = value_map.astype(np.float32)
    height, width = value_map.shape[:2]
    unit_width = max(1, int(unit_width))
    unit_height = max(1, int(unit_height))
    unit_rows = int(math.ceil(height / float(unit_height)))
    unit_cols = int(math.ceil(width / float(unit_width)))
    unit_map = np.zeros((unit_rows, unit_cols), dtype=np.float32)
    for row in range(unit_rows):
        y0, y1 = row * unit_height, min(height, (row + 1) * unit_height)
        for col in range(unit_cols):
            x0, x1 = col * unit_width, min(width, (col + 1) * unit_width)
            patch = value_map[y0:y1, x0:x1]
            if reduce == 'max':
                unit_map[row, col] = float(patch.max())
            elif reduce == 'median':
                unit_map[row, col] = float(np.median(patch))
            else:
                unit_map[row, col] = float(patch.mean())
    expanded_map = cv2.resize(unit_map, (width, height), interpolation=cv2.INTER_NEAREST).astype(np.float32)
    return unit_map, expanded_map


def top_percent_mean(unit_map, percent, largest=True):
    flat = unit_map.reshape(-1).astype(np.float32)
    if flat.size == 0:
        return float('nan'), 0
    percent = max(0.0, min(100.0, float(percent)))
    k = max(1, int(math.ceil(flat.size * percent / 100.0)))
    values = np.sort(flat)
    chosen = values[-k:] if largest else values[:k]
    return float(chosen.mean()), k


def gray_from_bgr(img):
    return cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)


def temporal_variation_map(img, neighbor_images, reduce='max'):
    if not neighbor_images:
        return None
    diffs = []
    for neighbor in neighbor_images:
        diff = np.abs(img - neighbor)
        if diff.ndim == 3:
            diff = diff.mean(axis=2)
        diffs.append(diff)
    stack = np.stack(diffs, axis=0)
    reduce = (reduce or 'max').lower()
    if reduce == 'mean':
        return np.mean(stack, axis=0).astype(np.float32)
    if reduce == 'median':
        return np.median(stack, axis=0).astype(np.float32)
    if reduce == 'p75':
        return np.percentile(stack, 75, axis=0).astype(np.float32)
    if reduce == 'max':
        return np.max(stack, axis=0).astype(np.float32)
    raise ValueError(f'Unsupported temporal_reduce: {reduce}')


def canny_edge_map(gray, low=40, high=120):
    blur = cv2.GaussianBlur(gray.astype(np.uint8), (3, 3), 0)
    return (cv2.Canny(blur, int(low), int(high)) > 0).astype(np.float32)


def calculate_lens_multi_metrics(
    img,
    neighbor_images,
    unit_width=4,
    unit_height=3,
    top_unit_percent=1.0,
    temporal_reduce='max',
    stable_variation_scale=3.0,
    black_luma_scale=35.0,
    edge_low=40,
    edge_high=120,
    dust_blur_ksize=31,
    dust_dark_scale=18.0,
    noise_blur_ksize=5,
    composite_weights=(1.0, 1.0, 1.0, 1.0),
    return_maps=False,
):
    """Return four hypothesis-specific abnormal scores in [0, 100].

    All scores are abnormal scores: larger means more suspicious/worse.
    - black_occlusion_score: small temporal variation + near-black luminance.
    - hair_score: persistent cluttered edge units across many nearby frames.
    - dust_score: temporally stable, soft dark blob / local attenuation response.
    - noise_score: high spatial high-frequency grain, especially unstable over frames.
    """
    start_time = time()
    gray = gray_from_bgr(img)
    all_frames = [img] + list(neighbor_images)
    all_grays = [gray_from_bgr(frame) for frame in all_frames]

    if neighbor_images:
        pix_var = temporal_variation_map(img, neighbor_images, reduce=temporal_reduce)
    else:
        pix_var = np.full(gray.shape, np.nan, dtype=np.float32)

    valid_temporal = np.isfinite(pix_var).any()
    if valid_temporal:
        unit_var, variation_map = aggregate_map_to_units(pix_var, unit_width, unit_height)
        stable_map = np.exp(-variation_map / max(float(stable_variation_scale), 1e-6)).astype(np.float32)
        unit_stable, stable_map = aggregate_map_to_units(stable_map, unit_width, unit_height)
    else:
        unit_var = np.full((int(math.ceil(gray.shape[0] / float(unit_height))), int(math.ceil(gray.shape[1] / float(unit_width)))), np.nan, dtype=np.float32)
        variation_map = None
        stable_map = np.zeros_like(gray, dtype=np.float32)
        unit_stable, _ = aggregate_map_to_units(stable_map, unit_width, unit_height)

    # 1) Installation/black occlusion: stable + close to black.
    black_degree = np.exp(-gray / max(float(black_luma_scale), 1e-6)).astype(np.float32)
    black_map = stable_map * black_degree
    unit_black, black_heat = aggregate_map_to_units(black_map, unit_width, unit_height)
    black_score, black_k = top_percent_mean(unit_black, top_unit_percent, largest=True)
    black_score *= 100.0

    # 2) Hair: cluttered and persistent thin edges. Use edge persistence across frames.
    edge_maps = [canny_edge_map(g, edge_low, edge_high) for g in all_grays]
    edge_stack = np.stack(edge_maps, axis=0)
    edge_persistence = edge_stack.mean(axis=0).astype(np.float32)
    edge_current = edge_maps[0]
    # Local edge density captures clutter; persistence suppresses moving-background edges.
    edge_density = cv2.GaussianBlur(edge_current, (9, 9), 0).astype(np.float32)
    hair_map = np.sqrt(np.clip(edge_density * edge_persistence, 0, 1)).astype(np.float32)
    unit_hair, hair_heat = aggregate_map_to_units(hair_map, unit_width, unit_height)
    hair_score, hair_k = top_percent_mean(unit_hair, max(top_unit_percent, 3.0), largest=True)
    hair_score *= 100.0

    # 3) Dust: stable soft dark/local-attenuation blob. DoG-like dark response.
    dust_blur_ksize = ensure_odd_kernel(dust_blur_ksize)
    local_mean = cv2.GaussianBlur(gray, (dust_blur_ksize, dust_blur_ksize), 0)
    dark_blob = np.clip((local_mean - gray) / max(float(dust_dark_scale), 1e-6), 0.0, 1.0)
    # Soft dust is usually not a sharp Canny edge; down-weight heavy edge units a little.
    anti_edge = 1.0 - np.clip(cv2.GaussianBlur(edge_current, (9, 9), 0) * 2.0, 0.0, 0.7)
    dust_map = (stable_map * dark_blob * anti_edge).astype(np.float32)
    unit_dust, dust_heat = aggregate_map_to_units(dust_map, unit_width, unit_height)
    dust_score, dust_k = top_percent_mean(unit_dust, top_unit_percent, largest=True)
    dust_score *= 100.0

    # 4) Noise: high-frequency grain. Use all frames; noise should be spatially high-pass strong.
    noise_blur_ksize = ensure_odd_kernel(noise_blur_ksize)
    noise_maps = []
    for g in all_grays:
        low = cv2.GaussianBlur(g, (noise_blur_ksize, noise_blur_ksize), 0)
        residual = np.abs(g - low)
        noise_maps.append(residual)
    noise_residual = np.mean(np.stack(noise_maps, axis=0), axis=0).astype(np.float32)
    # Dark-area amplification, but not mandatory; capped to avoid full-black occlusion becoming noise.
    dark_amp = 1.0 + 0.5 * np.exp(-gray / 60.0)
    noise_map = np.clip(noise_residual * dark_amp / 20.0, 0.0, 1.0).astype(np.float32)
    unit_noise, noise_heat = aggregate_map_to_units(noise_map, unit_width, unit_height)
    noise_score, noise_k = top_percent_mean(unit_noise, max(top_unit_percent, 5.0), largest=True)
    noise_score *= 100.0

    weights = np.asarray(composite_weights, dtype=np.float32)
    if weights.size != 4 or float(weights.sum()) <= 1e-6:
        weights = np.ones(4, dtype=np.float32)
    scores = np.asarray([black_score, hair_score, dust_score, noise_score], dtype=np.float32)
    composite_score = float(np.sum(scores * weights) / np.sum(weights))

    result = {
        'composite_score': composite_score,
        'black_occlusion_score': float(black_score),
        'hair_score': float(hair_score),
        'dust_score': float(dust_score),
        'noise_score': float(noise_score),
        'mean_temporal_variation': float(np.nanmean(pix_var)) if valid_temporal else float('nan'),
        'min_unit_variation': float(np.nanmin(unit_var)) if valid_temporal else float('nan'),
        'mean_luma': float(gray.mean()),
        'edge_density': float(edge_current.mean()),
        'dust_response': float(dust_map.mean()),
        'noise_response': float(noise_map.mean()),
        'valid_neighbors': len(neighbor_images),
        'elapsed_time': time() - start_time,
        'black_top_unit_count': black_k,
        'hair_top_unit_count': hair_k,
        'dust_top_unit_count': dust_k,
        'noise_top_unit_count': noise_k,
    }
    if return_maps:
        result.update({
            'black_heat': black_heat,
            'hair_heat': hair_heat,
            'dust_heat': dust_heat,
            'noise_heat': noise_heat,
            'variation_map': variation_map,
        })
    return result


def make_safe_name(text):
    import re
    text = str(text) if text is not None else ''
    text = re.sub(r'[^0-9A-Za-z._-]+', '_', text.strip())
    return text.strip('_') or 'none'


def build_heatmap_save_dir(save_txt_path):
    txt_dir = os.path.dirname(os.path.abspath(save_txt_path))
    heatmap_dir_name = os.path.splitext(os.path.basename(save_txt_path))[0]
    heatmap_dir = os.path.join(txt_dir, heatmap_dir_name)
    os.makedirs(heatmap_dir, exist_ok=True)
    return heatmap_dir


def colorize_heat(base_bgr, heat, alpha=0.55):
    heat_u8 = np.clip(heat * 255.0, 0, 255).astype(np.uint8)
    color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    blended = cv2.addWeighted(base_bgr, 1.0 - alpha, color, alpha, 0)
    out = base_bgr.copy()
    mask = heat_u8 > 0
    out[mask] = blended[mask]
    return out


def save_lens_multi_heatmap(img_path, raw_img, result, heatmap_dir, index, scene, alpha=0.55):
    os.makedirs(heatmap_dir, exist_ok=True)
    base = np.clip(raw_img, 0, 255).astype(np.uint8)
    h, w = base.shape[:2]
    panels = []
    labels = [
        ('black', 'black_occlusion_score', result.get('black_heat')),
        ('hair', 'hair_score', result.get('hair_heat')),
        ('dust', 'dust_score', result.get('dust_heat')),
        ('noise', 'noise_score', result.get('noise_heat')),
    ]
    for name, score_key, heat in labels:
        panel = base.copy() if heat is None else colorize_heat(base, heat, alpha)
        text = f'{name}={format_float(result[score_key])}'
        cv2.putText(panel, text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(panel, text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        panels.append(panel)
    top = np.concatenate(panels[:2], axis=1)
    bottom = np.concatenate(panels[2:], axis=1)
    canvas = np.concatenate([top, bottom], axis=0)
    title = f'composite={format_float(result["composite_score"])}  neighbors={result["valid_neighbors"]}'
    cv2.putText(canvas, title, (10, h * 2 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(canvas, title, (10, h * 2 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    scene_part = make_safe_name(scene) if scene else 'no_scene'
    stem = make_safe_name(os.path.splitext(os.path.basename(img_path))[0])
    out_path = os.path.join(heatmap_dir, f'{index:06d}_{scene_part}_{stem}_multi_heatmap.jpg')
    cv2.imwrite(out_path, canvas)
    return out_path


def get_scene_order_relation(scene_a, stat_a, scene_b, stat_b, lower_better):
    min_a, max_a = stat_a['min_score'], stat_a['max_score']
    min_b, max_b = stat_b['min_score'], stat_b['max_score']
    overlap_min, overlap_max = max(min_a, min_b), min(max_a, max_b)
    is_separable = overlap_min > overlap_max
    if not is_separable:
        return False, (f'  [{scene_a}] vs [{scene_b}]: NOT separable, '
                       f'range_a=[{format_float(min_a)}, {format_float(max_a)}], '
                       f'range_b=[{format_float(min_b)}, {format_float(max_b)}], '
                       f'overlap=[{format_float(overlap_min)}, {format_float(overlap_max)}]')
    if lower_better:
        worse_scene = scene_a if min_a > max_b else scene_b
        better_scene = scene_b if worse_scene == scene_a else scene_a
    else:
        worse_scene = scene_a if max_a < min_b else scene_b
        better_scene = scene_b if worse_scene == scene_a else scene_a
    return True, (f'  [{scene_a}] vs [{scene_b}]: SEPARABLE, '
                  f'range_a=[{format_float(min_a)}, {format_float(max_a)}], '
                  f'range_b=[{format_float(min_b)}, {format_float(max_b)}], '
                  f'worse=[{worse_scene}], better=[{better_scene}]')


def build_scene_separability_msgs(scene_stats, metric_name, lower_better, normal_scene_name='normal'):
    msgs = [f'Normal-vs-other scene separability based on {metric_name} min/max intervals:',
            '  Metric direction: lower_better=True, larger score means worse/abnormal']
    normal_scene = None
    for scene in scene_stats.keys():
        if scene.lower() == normal_scene_name.lower():
            normal_scene = scene
            break
    if normal_scene is None:
        msgs.append(f'  Scene [{normal_scene_name}] not found, skip separability analysis.')
        return msgs
    sep, total = 0, 0
    for scene in scene_stats.keys():
        if scene == normal_scene:
            continue
        ok, msg = get_scene_order_relation(normal_scene, scene_stats[normal_scene], scene, scene_stats[scene], lower_better)
        total += 1
        sep += int(ok)
        msgs.append(msg)
    ratio = sep / total if total else 0
    msgs.append(f'  Normal-vs-other separability summary: {sep}/{total} ({format_float(ratio)}) normal-other scene pairs are separable.')
    return msgs


def update_scene_stat(scene_stats, scene, score, path):
    if not scene or math.isnan(score):
        return
    stat = scene_stats[scene]
    stat['sum'] += score
    stat['count'] += 1
    if stat['max_score'] is None or score > stat['max_score']:
        stat['max_score'], stat['max_path'] = score, path
    if stat['min_score'] is None or score < stat['min_score']:
        stat['min_score'], stat['min_path'] = score, path


def main():
    parser = argparse.ArgumentParser(description='Lens abnormality validation with multiple hand-crafted metrics.')
    parser.add_argument('-t', '--target', type=str, default=None, help='input image/folder path.')
    parser.add_argument('--target_txt', type=str, default=None, help='txt file containing target image paths and scene markers.')
    parser.add_argument('-m', '--metric_name', type=str, default='lens_multi_metric', help='metric name used in logs/result file name.')
    parser.add_argument('--metric_mode', type=str, default='NR', help='kept for compatibility; this script only supports NR.')
    parser.add_argument('--device', type=str, default='cpu', help='kept for compatibility; CPU/OpenCV only.')
    parser.add_argument('--save_file', type=str, default=None, help='path to save csv results.')
    parser.add_argument('--save_txt_dir', type=str, default=None, help='directory to save txt results as current_time.metric_name.txt.')
    parser.add_argument('--no_heatmap', action='store_true', help='disable heatmap saving. Default saves when --save_txt_dir is set.')
    parser.add_argument('--heatmap_alpha', type=float, default=0.55, help='heatmap overlay alpha.')
    parser.add_argument('--window', type=int, default=3, help='use previous/next N frames as context.')
    parser.add_argument('--resize_width', type=int, default=640, help='resize width; <=0 disables.')
    parser.add_argument('--unit_width', type=int, default=4, help='super-pixel unit width.')
    parser.add_argument('--unit_height', type=int, default=3, help='super-pixel unit height.')
    parser.add_argument('--top_unit_percent', type=float, default=1.0, help='top suspicious unit percentage used for scores.')
    parser.add_argument('--temporal_reduce', type=str, default='max', choices=['max', 'p75', 'median', 'mean'], help='reduce temporal differences across neighbor frames.')
    parser.add_argument('--stable_variation_scale', type=float, default=3.0, help='exp(-temporal_variation/scale) scale; smaller requires stronger static prior.')
    parser.add_argument('--black_luma_scale', type=float, default=35.0, help='black occlusion luminance scale.')
    parser.add_argument('--edge_low', type=float, default=40.0, help='Canny low threshold for hair metric.')
    parser.add_argument('--edge_high', type=float, default=120.0, help='Canny high threshold for hair metric.')
    parser.add_argument('--dust_blur_ksize', type=int, default=31, help='large blur kernel for dust local attenuation response.')
    parser.add_argument('--dust_dark_scale', type=float, default=18.0, help='dark local attenuation scale for dust metric.')
    parser.add_argument('--noise_blur_ksize', type=int, default=5, help='small blur kernel for high-frequency noise residual.')
    parser.add_argument('--composite_weights', type=str, default='1,1,1,1', help='weights for black,hair,dust,noise composite score.')
    parser.add_argument('--score_metric', type=str, default='composite_score', choices=METRIC_COLUMNS[:5], help='which score is used for avg and scene separability.')
    parser.add_argument('--normal_scene', type=str, default='normal', help='scene name used as normal class.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output.')
    args = parser.parse_args()

    if args.metric_mode != 'NR':
        raise ValueError('This script only supports NR mode.')
    if args.target is None and args.target_txt is None:
        raise ValueError('Please specify --target or --target_txt.')

    weights = tuple(float(x.strip()) for x in args.composite_weights.split(',') if x.strip())
    if len(weights) != 4:
        raise ValueError('--composite_weights must contain four comma-separated numbers: black,hair,dust,noise')

    input_paths, input_scenes = get_input_paths(args.target, args.target_txt)
    if not input_paths:
        raise ValueError('No input images found.')
    resize_width = args.resize_width if args.resize_width > 0 else None
    lower_better = True  # abnormal score: lower is cleaner/better.

    seed_context_paths, all_load_paths = build_seed_context_paths(input_paths, args.window)
    print('Loading seed images and same-directory neighbor images...')
    print(f'Seed images: {len(input_paths)}')
    print(f'Images to load including neighbors: {len(all_load_paths)}')
    image_cache = {}
    for img_path in tqdm(all_load_paths, total=len(all_load_paths), unit='image'):
        image_cache[os.path.abspath(img_path)] = imread_image_resize(img_path, resize_width)

    save_txt_path, heatmap_dir, txt_f = None, None, None
    if args.save_txt_dir:
        save_txt_path = build_auto_txt_save_path(args.save_txt_dir, args.metric_name)
        if not args.no_heatmap:
            heatmap_dir = build_heatmap_save_dir(save_txt_path)
        txt_f = open(save_txt_path, 'w', encoding='utf-8')
        txt_f.write(f'metric_name: {args.metric_name}\n')
        txt_f.write('metric_mode: NR\n')
        txt_f.write(f'target: {args.target}\n')
        txt_f.write(f'target_txt: {args.target_txt}\n')
        txt_f.write(f'device: {args.device}\n')
        txt_f.write(f'lower_better: {lower_better}\n')
        txt_f.write('score_direction: abnormal_score_larger_means_worse\n')
        txt_f.write(f'score_metric_for_summary: {args.score_metric}\n')
        txt_f.write('metric_definitions:\n')
        txt_f.write('  black_occlusion_score: static same-position units AND low luminance, for installation/physical occlusion.\n')
        txt_f.write('  hair_score: persistent cluttered thin-edge units across nearby frames.\n')
        txt_f.write('  dust_score: static soft dark local-attenuation/blob response, down-weighting sharp edges.\n')
        txt_f.write('  noise_score: high-frequency grain residual, slightly amplified in dark areas.\n')
        for k in ['window', 'resize_width', 'unit_width', 'unit_height', 'top_unit_percent', 'temporal_reduce',
                  'stable_variation_scale', 'black_luma_scale', 'edge_low', 'edge_high', 'dust_blur_ksize',
                  'dust_dark_scale', 'noise_blur_ksize', 'composite_weights', 'normal_scene']:
            txt_f.write(f'{k}: {getattr(args, k)}\n')
        txt_f.write('seed_expand_mode: same_directory_previous_next_existing_files\n')
        txt_f.write(f'seed_count: {len(input_paths)}\n')
        txt_f.write(f'loaded_image_count: {len(all_load_paths)}\n')
        txt_f.write(f'heatmap_dir: {heatmap_dir}\n')
        txt_f.write(f'time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')
        txt_f.write('scene\timage\t' + '\t'.join(METRIC_COLUMNS) + '\tvalid_neighbors\ttime\theatmap\n')

    sf, sfwriter = None, None
    if args.save_file:
        sf = open(args.save_file, 'w', newline='')
        sfwriter = csv.writer(sf)
        sfwriter.writerow(['scene', 'image'] + METRIC_COLUMNS + ['valid_neighbors', 'time', 'heatmap'])

    avg_score, valid_count = 0.0, 0
    scene_stats = defaultdict(lambda: {'sum': 0.0, 'count': 0, 'max_score': None, 'max_path': None, 'min_score': None, 'min_path': None})

    pbar = tqdm(total=len(input_paths), unit='image')
    for idx, img_path in enumerate(input_paths):
        scene = input_scenes[idx]
        img_abs = os.path.abspath(img_path)
        neighbors = [image_cache[os.path.abspath(p)] for p in seed_context_paths[idx]]
        result = calculate_lens_multi_metrics(
            image_cache[img_abs], neighbors,
            unit_width=args.unit_width,
            unit_height=args.unit_height,
            top_unit_percent=args.top_unit_percent,
            temporal_reduce=args.temporal_reduce,
            stable_variation_scale=args.stable_variation_scale,
            black_luma_scale=args.black_luma_scale,
            edge_low=args.edge_low,
            edge_high=args.edge_high,
            dust_blur_ksize=args.dust_blur_ksize,
            dust_dark_scale=args.dust_dark_scale,
            noise_blur_ksize=args.noise_blur_ksize,
            composite_weights=weights,
            return_maps=heatmap_dir is not None,
        )
        heatmap_path = ''
        if heatmap_dir is not None:
            heatmap_path = save_lens_multi_heatmap(img_path, image_cache[img_abs], result, heatmap_dir, idx, scene, args.heatmap_alpha)

        score = float(result[args.score_metric])
        if not math.isnan(score):
            avg_score += score
            valid_count += 1
            update_scene_stat(scene_stats, scene, score, img_path)

        values = [format_float(result[col]) for col in METRIC_COLUMNS]
        elapsed = format_float(result['elapsed_time'])
        pbar.update(1)
        pbar.set_description(f'{format_scene_prefix(scene)}{args.metric_name} of {os.path.basename(img_path)}: {format_float(score)}')
        pbar.write(
            f'{format_scene_prefix(scene)}{args.metric_name} of {os.path.basename(img_path)}: {format_float(score)}\t'
            f'black={format_float(result["black_occlusion_score"])}\t'
            f'hair={format_float(result["hair_score"])}\t'
            f'dust={format_float(result["dust_score"])}\t'
            f'noise={format_float(result["noise_score"])}\tTime: {elapsed}s'
        )
        row = [scene or '', img_path] + values + [result['valid_neighbors'], elapsed, heatmap_path]
        if sfwriter is not None:
            sfwriter.writerow(row)
        if txt_f is not None:
            txt_f.write('\t'.join(map(str, row)) + '\n')
    pbar.close()

    avg_score = avg_score / valid_count if valid_count > 0 else float('nan')
    msg = (f'Average {args.metric_name}/{args.score_metric} score of {args.target or args.target_txt} '
           f'with {valid_count}/{len(input_paths)} valid images is: {format_float(avg_score)}')
    print(msg)

    scene_msgs = []
    if scene_stats:
        print('Scene statistics:')
        for scene, stat in scene_stats.items():
            if stat['count'] <= 0:
                continue
            scene_avg = stat['sum'] / stat['count']
            scene_msg = (f'  [{scene}] {args.metric_name}/{args.score_metric} statistics with {stat["count"]} images: '
                         f'avg={format_float(scene_avg)}, max={format_float(stat["max_score"])} ({stat["max_path"]}), '
                         f'min={format_float(stat["min_score"])} ({stat["min_path"]})')
            scene_msgs.append(scene_msg)
            print(scene_msg)

    separability_msgs = []
    if scene_stats:
        separability_msgs = build_scene_separability_msgs(scene_stats, f'{args.metric_name}/{args.score_metric}', lower_better, args.normal_scene)
        for m in separability_msgs:
            print(m)

    if txt_f is not None:
        txt_f.write('\n' + msg + '\n')
        if scene_msgs:
            txt_f.write('Scene statistics:\n')
            for m in scene_msgs:
                txt_f.write(m + '\n')
        if separability_msgs:
            txt_f.write('\n')
            for m in separability_msgs:
                txt_f.write(m + '\n')
        txt_f.close()
    if sf is not None:
        sf.close()
    if args.save_file:
        print(f'Done! CSV results are in {args.save_file}.')
    if save_txt_path:
        print(f'Done! TXT results are in {save_txt_path}.')
    if heatmap_dir:
        print(f'Done! Heatmaps are in {heatmap_dir}.')
    if not args.save_file and not save_txt_path:
        print('Done!')


if __name__ == '__main__':
    main()

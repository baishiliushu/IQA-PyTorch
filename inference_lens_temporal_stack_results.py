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
    'noise_score',
    'temporal_std_mean',
    'temporal_std_p95',
    'temporal_std_top_mean',
    'stable_decay_score',
    'stable_ratio',
    'black_pixel_ratio',
    'valid_frames',
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


def get_input_paths(input_path, input_txt=None):
    if input_txt is not None:
        return read_paths_from_txt(input_txt)
    if os.path.isfile(input_path):
        return [input_path], [None]
    paths = sorted(glob.glob(os.path.join(input_path, '*')), key=natural_key)
    paths = [p for p in paths if is_image_file(p)]
    return paths, [None] * len(paths)


def list_images_in_same_dir(img_path):
    img_dir = os.path.dirname(os.path.abspath(img_path))
    if not os.path.isdir(img_dir):
        return []
    paths = [
        os.path.join(img_dir, name)
        for name in os.listdir(img_dir)
        if is_image_file(name) and os.path.isfile(os.path.join(img_dir, name))
    ]
    return sorted(paths, key=natural_key)


def build_seed_context_paths(seed_paths, window, include_seed=True):
    seed_context_paths, all_paths, missing = [], set(), []
    for seed_path in seed_paths:
        seed_abs = os.path.abspath(seed_path)
        dir_images = list_images_in_same_dir(seed_abs)
        abs_to_path = {os.path.abspath(p): p for p in dir_images}
        if seed_abs not in abs_to_path:
            missing.append(seed_path)
            seed_context_paths.append([])
            continue
        seed_real = abs_to_path[seed_abs]
        idx = dir_images.index(seed_real)
        start = max(0, idx - window)
        end = min(len(dir_images), idx + window + 1)
        ctx = dir_images[start:end]
        if not include_seed:
            ctx = [p for p in ctx if os.path.abspath(p) != seed_abs]
        seed_context_paths.append(ctx)
        all_paths.update(ctx)
    if missing:
        raise FileNotFoundError('Some seed images do not exist: ' + ', '.join(missing[:5]))
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


def ensure_odd_kernel(v, min_value=1):
    v = int(v)
    if v < min_value:
        v = min_value
    if v % 2 == 0:
        v += 1
    return v


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


def normalize_01(x, low=None, high=None):
    x = x.astype(np.float32)
    if low is None:
        low = float(np.nanmin(x))
    if high is None:
        high = float(np.nanmax(x))
    if high - low < 1e-6:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - low) / (high - low), 0.0, 1.0).astype(np.float32)


def colorize_01(map01, colormap=cv2.COLORMAP_JET):
    return cv2.applyColorMap(np.clip(map01 * 255.0, 0, 255).astype(np.uint8), colormap)


def put_label(img, text, y=26):
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)


def resize_to(img, size):
    if img.shape[1] == size[0] and img.shape[0] == size[1]:
        return img
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def calculate_noise_from_stack(stack, noise_blur_ksize=5, noise_top_percent=5.0):
    grays = [cv2.cvtColor(np.clip(f, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) for f in stack]
    k = ensure_odd_kernel(noise_blur_ksize, 3)
    residuals = []
    for g in grays:
        low = cv2.GaussianBlur(g, (k, k), 0)
        residuals.append(np.abs(g - low))
    noise_map = np.mean(np.stack(residuals, axis=0), axis=0).astype(np.float32)
    flat = np.sort(noise_map.reshape(-1))
    top_k = max(1, int(math.ceil(flat.size * max(0.0, min(100.0, noise_top_percent)) / 100.0)))
    top_mean = float(flat[-top_k:].mean())
    # 20 灰度级以上的高频残差已经很可疑；该分数是启发式可视化分数。
    noise_score = 100.0 * (1.0 - math.exp(-top_mean / 20.0))
    return noise_map, float(noise_score)


def calculate_temporal_decay_black(stack, close_thresh=6.0, dilate_kernel=9, decay_rate=0.82, black_threshold=80):
    """Temporal decay dilation.

    For every adjacent pair, pixels whose RGB/BGR distance is within close_thresh
    are treated as same-position stable candidates. Their dilated neighborhood
    is multiplied by decay_rate; repeated hits become progressively darker.
    """
    if len(stack) < 2:
        h, w = stack[0].shape[:2]
        return np.full((h, w), 255, dtype=np.uint8), np.zeros((h, w), dtype=np.float32), 0.0, 0.0

    h, w = stack[0].shape[:2]
    decay_img = np.full((h, w), 255.0, dtype=np.float32)
    hit_count = np.zeros((h, w), dtype=np.float32)
    k = ensure_odd_kernel(dilate_kernel, 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    stable_total = 0.0

    for i in range(1, len(stack)):
        diff = np.abs(stack[i].astype(np.float32) - stack[i - 1].astype(np.float32)).mean(axis=2)
        stable = (diff <= close_thresh).astype(np.uint8)
        stable_total += float(stable.mean())
        if k > 1:
            stable = cv2.dilate(stable, kernel, iterations=1)
        mask = stable > 0
        decay_img[mask] *= float(decay_rate)
        hit_count[mask] += 1.0

    stable_ratio = stable_total / float(len(stack) - 1)
    black_pixel_ratio = float((decay_img <= black_threshold).mean())
    decay_score = 100.0 * (1.0 - float(decay_img.mean()) / 255.0)
    decay_u8 = np.clip(decay_img, 0, 255).astype(np.uint8)
    hit_norm = hit_count / max(1.0, float(len(stack) - 1))
    return decay_u8, hit_norm.astype(np.float32), float(decay_score), black_pixel_ratio, float(stable_ratio)


def calculate_stack_visualization(stack, args):
    start = time()
    stack_np = np.stack(stack, axis=0).astype(np.float32)
    mean_img = np.clip(stack_np.mean(axis=0), 0, 255).astype(np.uint8)
    median_img = np.clip(np.median(stack_np, axis=0), 0, 255).astype(np.uint8)
    temporal_std_map = stack_np.std(axis=0).mean(axis=2).astype(np.float32)
    temporal_std_mean = float(temporal_std_map.mean())
    temporal_std_p95 = float(np.percentile(temporal_std_map, 95))
    flat_std = np.sort(temporal_std_map.reshape(-1))
    top_k = max(1, int(math.ceil(flat_std.size * max(0.0, min(100.0, args.noise_top_percent)) / 100.0)))
    temporal_std_top_mean = float(flat_std[-top_k:].mean())

    noise_map, noise_score = calculate_noise_from_stack(stack, args.noise_blur_ksize, args.noise_top_percent)
    decay_img, hit_norm, stable_decay_score, black_pixel_ratio, stable_ratio = calculate_temporal_decay_black(
        stack,
        close_thresh=args.close_thresh,
        dilate_kernel=args.dilate_kernel,
        decay_rate=args.decay_rate,
        black_threshold=args.black_threshold,
    )

    return {
        'mean_img': mean_img,
        'median_img': median_img,
        'temporal_std_map': temporal_std_map,
        'noise_map': noise_map,
        'decay_img': decay_img,
        'hit_norm': hit_norm,
        'noise_score': noise_score,
        'temporal_std_mean': temporal_std_mean,
        'temporal_std_p95': temporal_std_p95,
        'temporal_std_top_mean': temporal_std_top_mean,
        'stable_decay_score': stable_decay_score,
        'stable_ratio': stable_ratio,
        'black_pixel_ratio': black_pixel_ratio,
        'valid_frames': len(stack),
        'elapsed_time': time() - start,
    }


def save_stack_visualization(img_path, seed_img, result, out_dir, index, scene, alpha=0.55):
    os.makedirs(out_dir, exist_ok=True)
    base = np.clip(seed_img, 0, 255).astype(np.uint8)
    h, w = base.shape[:2]
    size = (w, h)

    mean_img = resize_to(result['mean_img'], size)
    median_img = resize_to(result['median_img'], size)
    std_color = colorize_01(normalize_01(result['temporal_std_map'], 0, max(1.0, result['temporal_std_p95'])))
    noise_color = colorize_01(normalize_01(result['noise_map'], 0, max(1.0, np.percentile(result['noise_map'], 98))))
    decay_bgr = cv2.cvtColor(result['decay_img'], cv2.COLOR_GRAY2BGR)
    hit_color = colorize_01(result['hit_norm'])

    # Overlay hit map on the mean image; repeated stable/dilated hits are expected to reveal lens-attached pollution.
    hit_overlay = cv2.addWeighted(mean_img, 1.0 - alpha, hit_color, alpha, 0)
    panels = [base, mean_img, median_img, std_color, noise_color, decay_bgr, hit_overlay]
    labels = [
        'seed',
        'temporal_mean',
        'temporal_median',
        f'temporal_std p95={format_float(result["temporal_std_p95"])}',
        f'noise={format_float(result["noise_score"])}',
        f'decay_black={format_float(result["stable_decay_score"])}',
        'stable_hit_overlay',
    ]
    labeled = []
    for p, label in zip(panels, labels):
        p = p.copy()
        put_label(p, label)
        labeled.append(p)

    blank = np.zeros_like(base)
    put_label(blank, f'frames={result["valid_frames"]} stable_ratio={format_float(result["stable_ratio"])}', 26)
    put_label(blank, f'black_ratio={format_float(result["black_pixel_ratio"])}', 54)
    labeled.append(blank)

    row1 = np.concatenate(labeled[:4], axis=1)
    row2 = np.concatenate(labeled[4:8], axis=1)
    canvas = np.concatenate([row1, row2], axis=0)

    scene_part = make_safe_name(scene) if scene else 'no_scene'
    stem = make_safe_name(os.path.splitext(os.path.basename(img_path))[0])
    out_path = os.path.join(out_dir, f'{index:06d}_{scene_part}_{stem}_stack.jpg')
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


def scene_separability_msgs(scene_stats, metric_name, normal_scene='normal'):
    msgs = [f'Normal-vs-other separability based on {metric_name}; larger means more abnormal.']
    normal_key = None
    for s in scene_stats:
        if s.lower() == normal_scene.lower():
            normal_key = s
            break
    if normal_key is None:
        msgs.append(f'  Scene [{normal_scene}] not found, skip.')
        return msgs
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


def main():
    parser = argparse.ArgumentParser(description='Visualize temporal image stacking, noise, and temporal-decay dilation for lens pollution.')
    parser.add_argument('-t', '--target', type=str, default=None, help='input image/folder path.')
    parser.add_argument('--target_txt', type=str, default=None, help='txt file containing seed image paths and scene markers.')
    parser.add_argument('-m', '--metric_name', type=str, default='lens_temporal_stack', help='metric name used in logs/result file name.')
    parser.add_argument('--metric_mode', type=str, default='NR', help='kept for compatibility; only NR is supported.')
    parser.add_argument('--device', type=str, default='cpu', help='kept for compatibility; CPU/OpenCV only.')
    parser.add_argument('--save_txt_dir', type=str, default=None, help='directory to save txt results and stack visualizations.')
    parser.add_argument('--save_file', type=str, default=None, help='optional CSV output path.')
    parser.add_argument('--no_vis', action='store_true', help='disable visualization saving. Default saves when --save_txt_dir is set.')
    parser.add_argument('--window', type=int, default=3, help='use previous/next N frames plus seed for stacking.')
    parser.add_argument('--resize_width', type=int, default=640, help='resize width; <=0 disables.')
    parser.add_argument('--noise_blur_ksize', type=int, default=5, help='small blur kernel for high-frequency noise residual.')
    parser.add_argument('--noise_top_percent', type=float, default=5.0, help='top percentage used for noise/std summary.')
    parser.add_argument('--close_thresh', type=float, default=6.0, help='adjacent-frame RGB/BGR mean absolute difference threshold for stable pixels.')
    parser.add_argument('--dilate_kernel', type=int, default=9, help='dilation kernel for stable pixels before temporal black decay.')
    parser.add_argument('--decay_rate', type=float, default=0.82, help='stable dilated region is multiplied by this value each hit; smaller becomes black faster.')
    parser.add_argument('--black_threshold', type=float, default=80.0, help='threshold to report black_pixel_ratio in decay image.')
    parser.add_argument('--heatmap_alpha', type=float, default=0.55, help='overlay alpha for stable-hit visualization.')
    parser.add_argument('--score_metric', type=str, default='stable_decay_score', choices=['noise_score', 'stable_decay_score', 'temporal_std_top_mean'], help='metric used for average/scene separability.')
    parser.add_argument('--normal_scene', type=str, default='normal', help='scene name used as normal class.')
    args = parser.parse_args()

    if args.metric_mode != 'NR':
        raise ValueError('This script only supports NR mode.')
    if args.target is None and args.target_txt is None:
        raise ValueError('Please specify --target or --target_txt.')

    input_paths, input_scenes = get_input_paths(args.target, args.target_txt)
    if not input_paths:
        raise ValueError('No input images found.')
    resize_width = args.resize_width if args.resize_width > 0 else None

    seed_context_paths, all_load_paths = build_seed_context_paths(input_paths, args.window, include_seed=True)
    print('Loading seed images and same-directory previous/next frames...')
    print(f'Seed images: {len(input_paths)}')
    print(f'Images to load including stack frames: {len(all_load_paths)}')
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
        txt_f.write('score_direction: larger_means_more_abnormal\n')
        txt_f.write(f'score_metric_for_summary: {args.score_metric}\n')
        txt_f.write('method:\n')
        txt_f.write('  stack: save seed, temporal mean, temporal median, temporal std map.\n')
        txt_f.write('  noise: average high-frequency residual over the N-window stack.\n')
        txt_f.write('  temporal_decay_dilation: adjacent-frame close pixels are dilated, then accumulated by multiplicative darkening.\n')
        for k in ['window', 'resize_width', 'noise_blur_ksize', 'noise_top_percent', 'close_thresh', 'dilate_kernel', 'decay_rate', 'black_threshold', 'normal_scene']:
            txt_f.write(f'{k}: {getattr(args, k)}\n')
        txt_f.write('seed_expand_mode: same_directory_previous_next_existing_files_include_seed\n')
        txt_f.write(f'seed_count: {len(input_paths)}\nloaded_image_count: {len(all_load_paths)}\nvis_dir: {vis_dir}\n')
        txt_f.write(f'time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')
        txt_f.write('scene\timage\t' + '\t'.join(RESULT_COLUMNS) + '\ttime\tvisualization\n')

    sf, writer = None, None
    if args.save_file:
        sf = open(args.save_file, 'w', newline='')
        writer = csv.writer(sf)
        writer.writerow(['scene', 'image'] + RESULT_COLUMNS + ['time', 'visualization'])

    scene_stats = defaultdict(lambda: {'sum': 0.0, 'count': 0, 'max_score': None, 'max_path': None, 'min_score': None, 'min_path': None})
    avg, count = 0.0, 0

    pbar = tqdm(total=len(input_paths), unit='image')
    for idx, img_path in enumerate(input_paths):
        scene = input_scenes[idx]
        stack_paths = seed_context_paths[idx]
        stack = [cache[os.path.abspath(p)] for p in stack_paths]
        result = calculate_stack_visualization(stack, args)
        vis_path = ''
        if vis_dir is not None:
            vis_path = save_stack_visualization(img_path, cache[os.path.abspath(img_path)], result, vis_dir, idx, scene, args.heatmap_alpha)

        score = float(result[args.score_metric])
        if not math.isnan(score):
            avg += score
            count += 1
            update_scene_stat(scene_stats, scene, score, img_path)

        values = [format_float(result[c]) for c in RESULT_COLUMNS]
        elapsed = format_float(result['elapsed_time'])
        prefix = f'[{scene}] ' if scene else ''
        pbar.update(1)
        pbar.set_description(f'{prefix}{args.metric_name} of {os.path.basename(img_path)}: {format_float(score)}')
        pbar.write(f'{prefix}{args.metric_name} of {os.path.basename(img_path)}: {format_float(score)}\tnoise={format_float(result["noise_score"])}\tdecay={format_float(result["stable_decay_score"])}\tstd_top={format_float(result["temporal_std_top_mean"])}\tframes={result["valid_frames"]}\tTime: {elapsed}s')

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
    if scene_stats:
        print('Scene statistics:')
        for scene, stat in scene_stats.items():
            if stat['count'] <= 0:
                continue
            scene_avg = stat['sum'] / stat['count']
            m = f'  [{scene}] {args.metric_name}/{args.score_metric} statistics with {stat["count"]} images: avg={format_float(scene_avg)}, max={format_float(stat["max_score"])} ({stat["max_path"]}), min={format_float(stat["min_score"])} ({stat["min_path"]})'
            scene_msgs.append(m)
            print(m)
    sep_msgs = scene_separability_msgs(scene_stats, f'{args.metric_name}/{args.score_metric}', args.normal_scene) if scene_stats else []
    for m in sep_msgs:
        print(m)

    if txt_f is not None:
        txt_f.write('\n' + msg + '\n')
        if scene_msgs:
            txt_f.write('Scene statistics:\n')
            for m in scene_msgs:
                txt_f.write(m + '\n')
        if sep_msgs:
            txt_f.write('\n')
            for m in sep_msgs:
                txt_f.write(m + '\n')
        txt_f.close()
    if sf is not None:
        sf.close()
    if args.save_file:
        print(f'Done! CSV results are in {args.save_file}.')
    if save_txt_path:
        print(f'Done! TXT results are in {save_txt_path}.')
    if vis_dir:
        print(f'Done! Stack visualizations are in {vis_dir}.')
    if not args.save_file and not save_txt_path:
        print('Done!')


if __name__ == '__main__':
    main()

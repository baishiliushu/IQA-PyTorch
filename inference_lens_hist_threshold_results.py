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
RESULT_COLUMNS = ['threshold', 'foreground_ratio', 'background_ratio', 'gray_mean', 'gray_std']


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


def imread_image(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f'Failed to read image: {path}')
    return img


def gray_from_bgr(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def calculate_hist_threshold(seed_img, method='otsu'):
    gray = gray_from_bgr(seed_img)
    start = time()
    if method == 'otsu':
        threshold, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == 'triangle':
        threshold, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
    else:
        threshold = float(method)
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    fg = float((binary > 0).mean())
    return {
        'threshold': float(threshold),
        'foreground_ratio': fg,
        'background_ratio': 1.0 - fg,
        'gray_mean': float(gray.mean()),
        'gray_std': float(gray.std()),
        'elapsed_time': time() - start,
        'gray': gray,
        'binary': binary,
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


def draw_histogram(gray, threshold, width, height):
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).reshape(-1)
    hist = hist / max(float(hist.max()), 1e-6)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    xs = np.linspace(0, width - 1, 256).astype(np.int32)
    for i in range(255):
        x1, x2 = xs[i], xs[i + 1]
        y1 = height - 1 - int(hist[i] * (height - 35))
        y2 = height - 1 - int(hist[i + 1] * (height - 35))
        cv2.line(canvas, (x1, y1), (x2, y2), (200, 200, 200), 1, cv2.LINE_AA)
    tx = int(np.clip(threshold / 255.0 * (width - 1), 0, width - 1))
    cv2.line(canvas, (tx, 0), (tx, height - 1), (0, 0, 255), 2)
    put_label(canvas, f'histogram threshold={format_float(threshold)}')
    return canvas


def resize_panel(img, size):
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def save_visualization(img_path, seed_img, result, out_dir, index, scene):
    h, w = seed_img.shape[:2]
    panel_w = min(640, max(320, w))
    panel_h = int(round(panel_w * h / float(w)))
    size = (panel_w, panel_h)

    seed = resize_panel(seed_img, size)
    gray_bgr = cv2.cvtColor(resize_panel(result['gray'], size), cv2.COLOR_GRAY2BGR)
    binary_bgr = cv2.cvtColor(resize_panel(result['binary'], size), cv2.COLOR_GRAY2BGR)
    hist = draw_histogram(result['gray'], result['threshold'], panel_w, panel_h)

    put_label(seed, 'seed')
    put_label(gray_bgr, f'gray mean={format_float(result["gray_mean"])} std={format_float(result["gray_std"])}')
    put_label(binary_bgr, f'binary threshold={format_float(result["threshold"])} fg={format_float(result["foreground_ratio"])}')

    canvas = np.concatenate([
        np.concatenate([seed, gray_bgr], axis=1),
        np.concatenate([binary_bgr, hist], axis=1),
    ], axis=0)
    out = os.path.join(out_dir, f'{index:06d}_{make_safe_name(scene) if scene else "no_scene"}_{make_safe_name(os.path.splitext(os.path.basename(img_path))[0])}.jpg')
    cv2.imwrite(out, canvas)
    return out


def update_scene_stat(stats, scene, threshold, path):
    if not scene:
        return
    st = stats[scene]
    st['sum'] += threshold
    st['count'] += 1
    if st['max'] is None or threshold > st['max']:
        st['max'], st['max_path'] = threshold, path
    if st['min'] is None or threshold < st['min']:
        st['min'], st['min_path'] = threshold, path


def main():
    parser = argparse.ArgumentParser(description='Show seed image histogram binarization result and print threshold.')
    parser.add_argument('-t', '--target', default=None)
    parser.add_argument('--target_txt', default=None)
    parser.add_argument('-m', '--metric_name', default='lens_hist_threshold')
    parser.add_argument('--metric_mode', default='NR')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--save_txt_dir', default=None)
    parser.add_argument('--save_file', default=None)
    parser.add_argument('--method', default='otsu', help='otsu, triangle, or a numeric threshold like 80')
    parser.add_argument('--no_vis', action='store_true')
    args = parser.parse_args()

    if args.metric_mode != 'NR':
        raise ValueError('This script only supports NR mode.')

    paths, scenes = get_input_paths(args.target, args.target_txt)
    save_txt_path = vis_dir = txt_f = None
    if args.save_txt_dir:
        save_txt_path = build_auto_txt_save_path(args.save_txt_dir, args.metric_name)
        if not args.no_vis:
            vis_dir = build_vis_save_dir(save_txt_path)
        txt_f = open(save_txt_path, 'w', encoding='utf-8')
        txt_f.write(f'metric_name: {args.metric_name}\nmetric_mode: NR\nmethod: histogram binarization ({args.method})\n')
        for k in vars(args):
            txt_f.write(f'{k}: {getattr(args, k)}\n')
        txt_f.write(f'seed_count: {len(paths)}\nvis_dir: {vis_dir}\ntime: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
        txt_f.write('\nscene\timage\t' + '\t'.join(RESULT_COLUMNS) + '\ttime\tvisualization\n')

    sf = writer = None
    if args.save_file:
        sf = open(args.save_file, 'w', newline='', encoding='utf-8')
        writer = csv.writer(sf)
        writer.writerow(['scene', 'image'] + RESULT_COLUMNS + ['time', 'visualization'])

    stats = defaultdict(lambda: {'sum': 0.0, 'count': 0, 'max': None, 'max_path': None, 'min': None, 'min_path': None})
    print('Loading seed images and calculating histogram thresholds...')
    for i, p in enumerate(tqdm(paths, unit='image')):
        scene = scenes[i]
        img = imread_image(p)
        res = calculate_hist_threshold(img, args.method)
        vis = save_visualization(p, img, res, vis_dir, i, scene) if vis_dir else ''
        update_scene_stat(stats, scene, float(res['threshold']), p)
        elapsed = format_float(res['elapsed_time'])
        prefix = f'[{scene}] ' if scene else ''
        print(f'{prefix}{os.path.basename(p)} threshold={format_float(res["threshold"])} fg={format_float(res["foreground_ratio"])} bg={format_float(res["background_ratio"])} Time={elapsed}s')
        row = [scene or '', p] + [format_float(res[c]) for c in RESULT_COLUMNS] + [elapsed, vis]
        if writer:
            writer.writerow(row)
        if txt_f:
            txt_f.write('\t'.join(map(str, row)) + '\n')

    scene_msgs = []
    if stats:
        print('Scene threshold statistics:')
        scene_msgs.append('Scene threshold statistics:')
        for scene, st in stats.items():
            if st['count'] <= 0:
                continue
            line = f'  [{scene}] count={st["count"]}: avg={format_float(st["sum"] / st["count"])} max={format_float(st["max"])} ({st["max_path"]}) min={format_float(st["min"])} ({st["min_path"]})'
            print(line)
            scene_msgs.append(line)

    if txt_f:
        txt_f.write('\n')
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

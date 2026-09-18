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


IMAGE_EXTENSIONS = {
    '.jpg',
    '.jpeg',
    '.png',
    '.bmp',
    '.tif',
    '.tiff',
    '.webp',
}


def normalize_input_path(path):
    """Normalize normal local paths and file:// URI paths to filesystem paths."""
    path = path.strip()
    if path.startswith('file://'):
        parsed = urlparse(path)
        if parsed.netloc not in ('', 'localhost'):
            raise ValueError(f'Unsupported non-local file URI: {path}')
        path = unquote(parsed.path)
    return path


def parse_scene_marker(line):
    """Parse scene marker like '#-[hair_few]'. Return None if not a marker."""
    if line.startswith('#-[') and line.endswith(']'):
        return line[3:-1].strip()
    return None


def read_paths_from_txt(txt_path):
    """Read image paths and scene markers from a txt file."""
    txt_dir = os.path.dirname(os.path.abspath(txt_path))
    paths = []
    scenes = []
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
    """Get input paths from a single image, folder, or txt file."""
    if input_txt is not None:
        return read_paths_from_txt(input_txt)

    if os.path.isfile(input_path):
        return [input_path], [None]

    paths = sorted(glob.glob(os.path.join(input_path, '*')))
    return paths, [None] * len(paths)


def build_auto_txt_save_path(save_txt_dir, metric_name):
    """Build save path with format: current_time.metric_name.txt."""
    os.makedirs(save_txt_dir, exist_ok=True)
    time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    safe_metric_name = metric_name.replace(os.sep, '_')
    return os.path.join(save_txt_dir, f'{time_str}.{safe_metric_name}.txt')


def format_float(value):
    """Format numeric result with 4 decimal places."""
    return f'{float(value):.4f}'


def format_scene_prefix(scene):
    """Format scene name for terminal output."""
    return f'[{scene}] ' if scene else ''


def natural_key(path):
    """Natural sort key for file names with numbers."""
    import re

    name = os.path.basename(path)
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r'(\d+)', name)
    ]


def is_image_file(path):
    """Check common image file extensions."""
    return os.path.splitext(path)[1].lower() in IMAGE_EXTENSIONS


def list_images_in_same_dir(img_path):
    """List actually existing images in the same directory."""
    img_dir = os.path.dirname(os.path.abspath(img_path))
    if not os.path.isdir(img_dir):
        return []

    paths = [
        os.path.join(img_dir, name)
        for name in os.listdir(img_dir)
        if is_image_file(name) and os.path.isfile(os.path.join(img_dir, name))
    ]
    return sorted(paths, key=natural_key)


def build_seed_context_paths(seed_paths, window):
    """Use txt image paths as seed paths and collect existing same-dir neighbors.

    For every seed image, find its position in the naturally sorted image list
    of the same directory, then collect previous/next N actually existing image
    files as temporal context. Only seed images are scored; added neighbor
    images are used only for temporal variation calculation.
    """
    seed_context_paths = []
    all_paths = set()
    missing_seed_paths = []

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
        context_paths = [
            path for path in dir_images[start_index:end_index] if path != seed_real_path
        ]

        seed_context_paths.append(context_paths)
        all_paths.add(seed_real_path)
        for path in context_paths:
            all_paths.add(path)

    if missing_seed_paths:
        raise FileNotFoundError(
            'Some seed images from txt do not exist in their directories: '
            + ', '.join(missing_seed_paths[:5])
            + (' ...' if len(missing_seed_paths) > 5 else '')
        )

    return seed_context_paths, sorted(all_paths, key=natural_key)


def imread_image_resize(img_path, resize_width, use_rgb=True):
    """Read image as float32 [0, 255], optionally resize by width.

    By default this keeps 3 color channels. OpenCV stores the data as BGR in
    memory, but all three RGB/BGR channels are used consistently during temporal
    difference calculation.
    """
    flag = cv2.IMREAD_COLOR if use_rgb else cv2.IMREAD_GRAYSCALE
    img = cv2.imread(img_path, flag)
    if img is None:
        raise FileNotFoundError(f'Failed to read image: {img_path}')

    if resize_width is not None and resize_width > 0 and img.shape[1] != resize_width:
        scale = resize_width / float(img.shape[1])
        resize_height = max(1, int(round(img.shape[0] * scale)))
        img = cv2.resize(img, (resize_width, resize_height), interpolation=cv2.INTER_AREA)
    return img.astype(np.float32)


def imread_gray_resize(img_path, resize_width):
    """Backward-compatible wrapper. New logic uses imread_image_resize()."""
    return imread_image_resize(img_path, resize_width, use_rgb=False)


def ensure_odd_kernel(kernel_size, min_value=3):
    """Normalize OpenCV kernel size to a positive odd integer."""
    kernel_size = int(kernel_size)
    if kernel_size < min_value:
        kernel_size = min_value
    if kernel_size % 2 == 0:
        kernel_size += 1
    return kernel_size


def normalize_to_255(img):
    """Normalize an image to float32 [0, 255]."""
    img = img.astype(np.float32)
    min_val = float(np.min(img))
    max_val = float(np.max(img))
    if max_val - min_val < 1e-6:
        return np.zeros_like(img, dtype=np.float32)
    return ((img - min_val) / (max_val - min_val) * 255.0).astype(np.float32)


def preprocess_each_frame(
    img,
    method='gray',
    blur_ksize=9,
    clahe_clip_limit=2.0,
    clahe_tile_grid_size=8,
):
    """Preprocess every frame before temporal difference calculation.

    The temporal-static idea is sensitive to illumination/noise/background
    texture. This function keeps all per-frame preprocessing in one place so
    different assumptions can be validated by changing only --preprocess_method.

    Supported methods:
        gray/raw/none:
            Use resized image directly. With default color input, RGB/BGR three
            channels are kept and used in temporal difference calculation.
        gaussian_blur:
            Smooth noise before temporal difference. Useful when weak-light noise
            causes unstable false negatives.
        clahe:
            Local contrast enhancement. Useful when dust/hair contrast is weak.
        highpass:
            Remove low-frequency illumination/background by subtracting local
            Gaussian blur. Often better for lens hair/dust edge-like artifacts.
        sobel:
            Use gradient magnitude. Suppresses flat regions and highlights edges.
        laplacian:
            Use second-order edge response. More sensitive than sobel, also more
            sensitive to noise.
    """
    method = (method or 'gray').lower()
    img = np.clip(img, 0, 255).astype(np.float32)

    if method in ('none', 'raw', 'gray'):
        return img

    blur_ksize = ensure_odd_kernel(blur_ksize)

    if method == 'gaussian_blur':
        return cv2.GaussianBlur(img, (blur_ksize, blur_ksize), 0).astype(np.float32)

    if method == 'clahe':
        tile_size = max(1, int(clahe_tile_grid_size))
        clahe = cv2.createCLAHE(
            clipLimit=float(clahe_clip_limit),
            tileGridSize=(tile_size, tile_size),
        )
        if img.ndim == 2:
            return clahe.apply(img.astype(np.uint8)).astype(np.float32)
        lab = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR).astype(np.float32)

    if method == 'highpass':
        low_freq = cv2.GaussianBlur(img, (blur_ksize, blur_ksize), 0)
        highpass = img - low_freq + 128.0
        return np.clip(highpass, 0, 255).astype(np.float32)

    if method == 'sobel':
        grad_x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
        if img.ndim == 2:
            magnitude = cv2.magnitude(grad_x, grad_y)
        else:
            magnitude = np.sqrt(grad_x * grad_x + grad_y * grad_y)
        return normalize_to_255(magnitude)

    if method == 'laplacian':
        lap = cv2.Laplacian(img, cv2.CV_32F, ksize=3)
        return normalize_to_255(np.abs(lap))

    raise ValueError(f'Unsupported preprocess method: {method}')


def aggregate_map_to_units(value_map, unit_width=4, unit_height=3):
    """Aggregate a per-pixel map to small super-pixel units.

    This reduces pixel-level jitter while keeping the unit small enough to avoid
    becoming coarse image blocks. Border units are kept with their actual size.
    """
    value_map = value_map.astype(np.float32)
    height, width = value_map.shape[:2]
    unit_width = max(1, int(unit_width))
    unit_height = max(1, int(unit_height))

    unit_rows = int(math.ceil(height / float(unit_height)))
    unit_cols = int(math.ceil(width / float(unit_width)))
    unit_map = np.zeros((unit_rows, unit_cols), dtype=np.float32)

    for row in range(unit_rows):
        y0 = row * unit_height
        y1 = min(height, y0 + unit_height)
        for col in range(unit_cols):
            x0 = col * unit_width
            x1 = min(width, x0 + unit_width)
            unit_map[row, col] = float(value_map[y0:y1, x0:x1].mean())

    expanded_map = cv2.resize(
        unit_map,
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    ).astype(np.float32)
    return unit_map, expanded_map


def calculate_lens_static_score(
    img,
    neighbor_images,
    variation_thresh,
    min_area_ratio,
    morph_kernel,
    unit_width=4,
    unit_height=3,
    top_unit_percent=1.0,
    score_variation_scale=1.0,
    temporal_reduce='max',
    return_maps=False,
):
    """Calculate lens-attached-object score for one frame.

    Prior assumption:
        The camera is moving. Therefore normal background pixels tend to vary
        over nearby frames at the same image coordinate, while lens-attached
        dust/hair/occlusion stays fixed and has smaller temporal variation.

    Score:
        1. Calculate same-position temporal variation using RGB/BGR 3 channels.
        2. Aggregate pixel variation into small super-pixel units, e.g. 4x3.
        3. Focus on units with the smallest variation, but score directly from
           their absolute variation value instead of clipped stable-degree.

        This is not an area-ratio score. The default temporal_reduce=max also
        requires a unit to stay stable against every neighbor frame, reducing
        accidental 100 scores caused by one coincidentally similar frame.
    """
    start_time = time()

    if not neighbor_images:
        elapsed_time = time() - start_time
        result = {
            'score': float('nan'),
            'stable_area_ratio': float('nan'),
            'largest_area_ratio': float('nan'),
            'mean_variation': float('nan'),
            'mean_stable_degree': float('nan'),
            'low_unit_variation': float('nan'),
            'min_unit_variation': float('nan'),
            'max_unit_stable_degree': float('nan'),
            'top_unit_stable_degree': float('nan'),
            'suspicious_unit_count': 0,
            'component_count': 0,
            'valid_neighbors': 0,
            'elapsed_time': elapsed_time,
        }
        if return_maps:
            result.update(
                {
                    'variation_map': None,
                    'unit_variation_map': None,
                    'stable_degree_map': None,
                    'unit_stable_degree_map': None,
                    'stable_mask': None,
                    'valid_component_mask': None,
                    'largest_component_mask': None,
                    'top_unit_mask': None,
                }
            )
        return result

    diffs = []
    for neighbor_img in neighbor_images:
        diff = np.abs(img - neighbor_img)
        if diff.ndim == 3:
            # Use all three color channels. The result is one temporal
            # variation value for each pixel location.
            diff = diff.mean(axis=2)
        diffs.append(diff)

    diff_stack = np.stack(diffs, axis=0)
    temporal_reduce = (temporal_reduce or 'max').lower()
    if temporal_reduce == 'median':
        pixel_variation_map = np.median(diff_stack, axis=0)
    elif temporal_reduce == 'mean':
        pixel_variation_map = np.mean(diff_stack, axis=0)
    elif temporal_reduce == 'p75':
        pixel_variation_map = np.percentile(diff_stack, 75, axis=0)
    elif temporal_reduce == 'max':
        pixel_variation_map = np.max(diff_stack, axis=0)
    else:
        raise ValueError(f'Unsupported temporal_reduce: {temporal_reduce}')

    unit_variation_map, variation_map = aggregate_map_to_units(
        pixel_variation_map,
        unit_width=unit_width,
        unit_height=unit_height,
    )
    unit_stable_degree_map = 1.0 - np.clip(
        unit_variation_map / max(float(variation_thresh), 1e-6),
        0.0,
        1.0,
    )
    stable_degree_map = cv2.resize(
        unit_stable_degree_map,
        (pixel_variation_map.shape[1], pixel_variation_map.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    ).astype(np.float32)
    stable_mask = (variation_map <= variation_thresh).astype(np.uint8)

    if morph_kernel and morph_kernel > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel)
        )
        stable_mask = cv2.morphologyEx(stable_mask, cv2.MORPH_OPEN, kernel)
        stable_mask = cv2.morphologyEx(stable_mask, cv2.MORPH_CLOSE, kernel)

    height, width = stable_mask.shape
    image_area = float(height * width)
    min_area = max(1, int(round(image_area * min_area_ratio)))

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        stable_mask, connectivity=8
    )
    largest_area = 0
    largest_label = 0
    valid_component_count = 0
    valid_component_mask = np.zeros_like(stable_mask, dtype=np.uint8)
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_area:
            valid_component_count += 1
            valid_component_mask[labels == label] = 1
            if area > largest_area:
                largest_area = area
                largest_label = label

    stable_area = int(stable_mask.sum())
    stable_area_ratio = stable_area / image_area
    largest_area_ratio = largest_area / image_area
    flat_unit_variation = unit_variation_map.reshape(-1)
    top_unit_percent = max(0.0, min(100.0, float(top_unit_percent)))
    top_k = max(1, int(math.ceil(flat_unit_variation.size * top_unit_percent / 100.0)))
    low_unit_variations = np.sort(flat_unit_variation)[:top_k]
    low_unit_variation = float(low_unit_variations.mean())
    score_variation_scale = max(1e-6, float(score_variation_scale))
    # Avoid the previous saturation problem:
    # clipped stable degree makes all units below threshold almost equally high.
    # Exponential mapping keeps sensitivity around near-zero variation.
    score = float(100.0 * math.exp(-low_unit_variation / score_variation_scale))
    top_unit_stable_degree = float(
        np.mean(np.exp(-low_unit_variations / score_variation_scale))
    )
    top_unit_threshold = float(low_unit_variations.max())
    top_unit_mask_small = (unit_variation_map <= top_unit_threshold).astype(np.uint8)
    top_unit_mask = cv2.resize(
        top_unit_mask_small,
        (pixel_variation_map.shape[1], pixel_variation_map.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    ).astype(np.uint8)
    elapsed_time = time() - start_time

    result = {
        'score': score,
        'stable_area_ratio': stable_area_ratio,
        'largest_area_ratio': largest_area_ratio,
        'mean_variation': float(variation_map.mean()),
        'mean_stable_degree': float(stable_degree_map.mean()),
        'low_unit_variation': low_unit_variation,
        'min_unit_variation': float(unit_variation_map.min()),
        'max_unit_stable_degree': float(unit_stable_degree_map.max()),
        'top_unit_stable_degree': top_unit_stable_degree,
        'suspicious_unit_count': top_k,
        'component_count': valid_component_count,
        'valid_neighbors': len(neighbor_images),
        'elapsed_time': elapsed_time,
    }
    if return_maps:
        largest_component_mask = (labels == largest_label).astype(np.uint8)
        result.update(
            {
                'variation_map': variation_map,
                'unit_variation_map': unit_variation_map,
                'stable_degree_map': stable_degree_map,
                'unit_stable_degree_map': unit_stable_degree_map,
                'stable_mask': stable_mask,
                'valid_component_mask': valid_component_mask,
                'largest_component_mask': largest_component_mask,
                'top_unit_mask': top_unit_mask,
            }
        )
    return result


def make_safe_name(text):
    """Convert arbitrary path/scene text to a safe file-name component."""
    import re

    text = str(text) if text is not None else ''
    text = text.strip()
    text = re.sub(r'[^0-9A-Za-z._-]+', '_', text)
    return text.strip('_') or 'none'


def build_heatmap_save_dir(save_txt_path):
    """Use result txt basename as heatmap sub-directory name."""
    txt_dir = os.path.dirname(os.path.abspath(save_txt_path))
    txt_name = os.path.basename(save_txt_path)
    heatmap_dir_name = os.path.splitext(txt_name)[0]
    heatmap_dir = os.path.join(txt_dir, heatmap_dir_name)
    os.makedirs(heatmap_dir, exist_ok=True)
    return heatmap_dir


def save_lens_static_heatmap(
    img_path,
    resized_gray,
    result,
    heatmap_dir,
    index,
    scene,
    variation_thresh,
    alpha=0.55,
):
    """Save one visualization image.

    Visualization convention:
        - Red/yellow heatmap: stronger low-variation degree, more suspicious.
        - Magenta contour: top suspicious super-pixel units used by score.
        - Green contour: valid low-variation connected components.
        - Cyan contour: largest low-variation component, shown only as an
          auxiliary shape cue. It no longer determines final score.
    """
    os.makedirs(heatmap_dir, exist_ok=True)

    if resized_gray.ndim == 2:
        base_bgr = cv2.cvtColor(np.clip(resized_gray, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    else:
        base_bgr = np.clip(resized_gray, 0, 255).astype(np.uint8)

    variation_map = result.get('variation_map')
    stable_degree_map = result.get('stable_degree_map')
    stable_mask = result.get('stable_mask')
    valid_component_mask = result.get('valid_component_mask')
    largest_component_mask = result.get('largest_component_mask')
    top_unit_mask = result.get('top_unit_mask')

    if variation_map is None or stable_mask is None:
        overlay = base_bgr
        cv2.putText(
            overlay,
            'No valid neighbor frames',
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    else:
        if stable_degree_map is None:
            variation_thresh = max(1e-6, float(variation_thresh))
            stable_degree_map = 1.0 - np.clip(
                variation_map / variation_thresh, 0.0, 1.0
            )
        heat_u8 = np.clip(stable_degree_map * 255.0, 0, 255).astype(np.uint8)
        heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
        overlay = base_bgr.copy()
        heat_mask = heat_u8 > 0
        if heat_mask.any():
            blended = cv2.addWeighted(base_bgr, 1.0 - alpha, heat_color, alpha, 0)
            overlay[heat_mask] = blended[heat_mask]

        if valid_component_mask is not None and valid_component_mask.any():
            contours, _ = cv2.findContours(
                (valid_component_mask * 255).astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            cv2.drawContours(overlay, contours, -1, (0, 255, 0), 1)

        if largest_component_mask is not None and largest_component_mask.any():
            contours, _ = cv2.findContours(
                (largest_component_mask * 255).astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            cv2.drawContours(overlay, contours, -1, (255, 255, 0), 2)

        if top_unit_mask is not None and top_unit_mask.any():
            contours, _ = cv2.findContours(
                (top_unit_mask * 255).astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            cv2.drawContours(overlay, contours, -1, (255, 0, 255), 1)

    info_lines = [
        f'score={format_float(result["score"])}',
        f'low_unit_var={format_float(result["low_unit_variation"])}',
        f'min_unit_var={format_float(result["min_unit_variation"])}',
        f'mean_var={format_float(result["mean_variation"])}',
        f'neighbors={result["valid_neighbors"]}',
    ]
    x, y = 10, 24
    for line in info_lines:
        cv2.putText(
            overlay,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += 24

    scene_part = make_safe_name(scene) if scene else 'no_scene'
    stem = make_safe_name(os.path.splitext(os.path.basename(img_path))[0])
    out_name = f'{index:06d}_{scene_part}_{stem}_heatmap.jpg'
    out_path = os.path.join(heatmap_dir, out_name)
    cv2.imwrite(out_path, overlay)
    return out_path


def get_scene_order_relation(scene_a, stat_a, scene_b, stat_b, lower_better):
    """Return pairwise scene separability relation based on min/max intervals."""
    min_a = stat_a['min_score']
    max_a = stat_a['max_score']
    min_b = stat_b['min_score']
    max_b = stat_b['max_score']

    overlap_min = max(min_a, min_b)
    overlap_max = min(max_a, max_b)
    is_separable = overlap_min > overlap_max

    if not is_separable:
        return {
            'is_separable': False,
            'msg': (
                f'  [{scene_a}] vs [{scene_b}]: NOT separable, '
                f'range_a=[{format_float(min_a)}, {format_float(max_a)}], '
                f'range_b=[{format_float(min_b)}, {format_float(max_b)}], '
                f'overlap=[{format_float(overlap_min)}, {format_float(overlap_max)}]'
            ),
        }

    if lower_better:
        if min_a > max_b:
            worse_scene, better_scene = scene_a, scene_b
        else:
            worse_scene, better_scene = scene_b, scene_a
    else:
        if max_a < min_b:
            worse_scene, better_scene = scene_a, scene_b
        else:
            worse_scene, better_scene = scene_b, scene_a

    return {
        'is_separable': True,
        'msg': (
            f'  [{scene_a}] vs [{scene_b}]: SEPARABLE, '
            f'range_a=[{format_float(min_a)}, {format_float(max_a)}], '
            f'range_b=[{format_float(min_b)}, {format_float(max_b)}], '
            f'worse=[{worse_scene}], better=[{better_scene}]'
        ),
    }


def build_scene_separability_msgs(
    scene_stats, metric_name, lower_better, normal_scene_name='normal'
):
    """Build normal-vs-other scene separability summary messages."""
    scenes = list(scene_stats.keys())
    direction_msg = (
        'lower_better=True, larger score means worse quality'
        if lower_better
        else 'lower_better=False, smaller score means worse quality'
    )
    msgs = [
        f'Normal-vs-other scene separability based on {metric_name} min/max intervals:',
        f'  Metric direction: {direction_msg}',
    ]

    normal_scene = None
    for scene in scenes:
        if scene.lower() == normal_scene_name.lower():
            normal_scene = scene
            break

    if normal_scene is None:
        msgs.append(
            f'  Scene [{normal_scene_name}] not found, skip separability analysis.'
        )
        return msgs

    other_scenes = [scene for scene in scenes if scene != normal_scene]
    if not other_scenes:
        msgs.append(f'  No other scenes except [{normal_scene}], skip analysis.')
        return msgs

    separable_count = 0
    total_count = 0
    for scene in other_scenes:
        relation = get_scene_order_relation(
            normal_scene,
            scene_stats[normal_scene],
            scene,
            scene_stats[scene],
            lower_better,
        )
        total_count += 1
        if relation['is_separable']:
            separable_count += 1
        msgs.append(relation['msg'])

    ratio = separable_count / total_count if total_count > 0 else 0
    msgs.append(
        f'  Normal-vs-other separability summary: {separable_count}/{total_count} '
        f'({format_float(ratio)}) normal-other scene pairs are separable.'
    )
    return msgs


def main():
    parser = argparse.ArgumentParser(
        description='Lens dust/hair/occlusion validation by temporal pixel stability.'
    )
    parser.add_argument(
        '-t', '--target', type=str, default=None, help='input image/folder path.'
    )
    parser.add_argument(
        '--target_txt',
        type=str,
        default=None,
        help='txt file containing target image paths and scene markers.',
    )
    parser.add_argument(
        '-m',
        '--metric_name',
        type=str,
        default='lens_static',
        help='metric name used in logs and result file name.',
    )
    parser.add_argument(
        '--metric_mode',
        type=str,
        default='NR',
        help='kept for compatibility; this script only supports NR.',
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        help='kept for compatibility; this script runs on CPU with OpenCV.',
    )
    parser.add_argument(
        '--save_file', type=str, default=None, help='path to save csv results.'
    )
    parser.add_argument(
        '--save_txt_dir',
        type=str,
        default=None,
        help='directory to save txt results as current_time.metric_name.txt.',
    )
    parser.add_argument(
        '--no_heatmap',
        action='store_true',
        help='disable heatmap visualization saving. By default heatmaps are saved when --save_txt_dir is set.',
    )
    parser.add_argument(
        '--heatmap_alpha',
        type=float,
        default=0.55,
        help='heatmap overlay alpha in [0, 1].',
    )
    parser.add_argument(
        '--window',
        type=int,
        default=3,
        help='use previous/next N frames for temporal variation.',
    )
    parser.add_argument(
        '--variation_thresh',
        type=float,
        default=5.0,
        help='low-variation threshold in RGB/BGR averaged value range [0, 255].',
    )
    parser.add_argument(
        '--unit_width',
        type=int,
        default=4,
        help='small super-pixel unit width for temporal variation aggregation.',
    )
    parser.add_argument(
        '--unit_height',
        type=int,
        default=3,
        help='small super-pixel unit height for temporal variation aggregation.',
    )
    parser.add_argument(
        '--top_unit_percent',
        type=float,
        default=1.0,
        help='score uses units with the lowest variation by this percentage.',
    )
    parser.add_argument(
        '--score_variation_scale',
        type=float,
        default=1.0,
        help='scale of exp(-low_unit_variation / scale) score mapping. Smaller value means only extremely stable units get high score.',
    )
    parser.add_argument(
        '--temporal_reduce',
        type=str,
        default='max',
        choices=['max', 'p75', 'median', 'mean'],
        help='how to reduce temporal differences across neighbor frames. max is strict and helps avoid accidental 100 scores.',
    )
    parser.add_argument(
        '--min_area_ratio',
        type=float,
        default=0.001,
        help='minimum connected component area ratio to count as valid region.',
    )
    parser.add_argument(
        '--morph_kernel',
        type=int,
        default=5,
        help='morphological kernel size. Set <=1 to disable.',
    )
    parser.add_argument(
        '--resize_width',
        type=int,
        default=640,
        help='resize image width for faster validation. Set <=0 to disable.',
    )
    parser.add_argument(
        '--preprocess_method',
        type=str,
        default='gray',
        choices=[
            'gray',
            'raw',
            'none',
            'gaussian_blur',
            'clahe',
            'highpass',
            'sobel',
            'laplacian',
        ],
        help='per-frame preprocessing before temporal difference.',
    )
    parser.add_argument(
        '--preprocess_blur_ksize',
        type=int,
        default=9,
        help='Gaussian kernel size used by gaussian_blur/highpass preprocessing.',
    )
    parser.add_argument(
        '--clahe_clip_limit',
        type=float,
        default=2.0,
        help='CLAHE clip limit when --preprocess_method clahe is used.',
    )
    parser.add_argument(
        '--clahe_tile_grid_size',
        type=int,
        default=8,
        help='CLAHE tile grid size when --preprocess_method clahe is used.',
    )
    parser.add_argument(
        '--normal_scene',
        type=str,
        default='normal',
        help='scene name used as normal class in separability analysis.',
    )
    parser.add_argument(
        '--no_scene_isolated',
        action='store_true',
        help='deprecated compatibility flag; seed neighbors are collected from the same directory.',
    )
    parser.add_argument(
        '-v', '--verbose', action='store_true', help='Enable verbose output.'
    )

    args = parser.parse_args()

    if args.metric_mode != 'NR':
        raise ValueError('This script only supports NR mode.')
    if args.target is None and args.target_txt is None:
        raise ValueError('Please specify --target or --target_txt.')

    input_paths, input_scenes = get_input_paths(args.target, args.target_txt)
    if not input_paths:
        raise ValueError('No input images found.')

    metric_name = args.metric_name
    lower_better = True  # This score is an abnormal score: lower is cleaner/better.
    resize_width = args.resize_width if args.resize_width > 0 else None

    seed_context_paths, all_load_paths = build_seed_context_paths(input_paths, args.window)

    print('Loading seed images and same-directory neighbor images...')
    print(f'Seed images: {len(input_paths)}')
    print(f'Images to load including neighbors: {len(all_load_paths)}')
    raw_image_cache = {}
    processed_image_cache = {}
    for img_path in tqdm(all_load_paths, total=len(all_load_paths), unit='image'):
        img_abs_path = os.path.abspath(img_path)
        raw_img = imread_image_resize(img_path, resize_width, use_rgb=True)
        raw_image_cache[img_abs_path] = raw_img
        processed_image_cache[img_abs_path] = preprocess_each_frame(
            raw_img,
            method=args.preprocess_method,
            blur_ksize=args.preprocess_blur_ksize,
            clahe_clip_limit=args.clahe_clip_limit,
            clahe_tile_grid_size=args.clahe_tile_grid_size,
        )

    if args.save_file:
        sf = open(args.save_file, 'w', newline='')
        sfwriter = csv.writer(sf)
        sfwriter.writerow(
            [
                'scene',
                'image',
                'score',
                'stable_area_ratio',
                'largest_area_ratio',
                'mean_variation',
                'mean_stable_degree',
                'low_unit_variation',
                'min_unit_variation',
                'max_unit_stable_degree',
                'top_unit_stable_degree',
                'suspicious_unit_count',
                'component_count',
                'valid_neighbors',
                'time',
                'heatmap',
            ]
        )
    else:
        sf = None
        sfwriter = None

    save_txt_path = None
    heatmap_dir = None
    txt_f = None
    if args.save_txt_dir:
        save_txt_path = build_auto_txt_save_path(args.save_txt_dir, metric_name)
        if not args.no_heatmap:
            heatmap_dir = build_heatmap_save_dir(save_txt_path)
        txt_f = open(save_txt_path, 'w', encoding='utf-8')
        txt_f.write(f'metric_name: {metric_name}\n')
        txt_f.write(f'metric_mode: NR\n')
        txt_f.write(f'target: {args.target}\n')
        txt_f.write(f'target_txt: {args.target_txt}\n')
        txt_f.write(f'device: {args.device}\n')
        txt_f.write(f'lower_better: {lower_better}\n')
        txt_f.write(f'window: {args.window}\n')
        txt_f.write(f'variation_thresh: {format_float(args.variation_thresh)}\n')
        txt_f.write(f'unit_width: {args.unit_width}\n')
        txt_f.write(f'unit_height: {args.unit_height}\n')
        txt_f.write(f'top_unit_percent: {format_float(args.top_unit_percent)}\n')
        txt_f.write(f'score_variation_scale: {format_float(args.score_variation_scale)}\n')
        txt_f.write(f'temporal_reduce: {args.temporal_reduce}\n')
        txt_f.write(f'min_area_ratio: {format_float(args.min_area_ratio)}\n')
        txt_f.write(f'morph_kernel: {args.morph_kernel}\n')
        txt_f.write(f'resize_width: {args.resize_width}\n')
        txt_f.write(f'preprocess_method: {args.preprocess_method}\n')
        txt_f.write(f'preprocess_blur_ksize: {args.preprocess_blur_ksize}\n')
        txt_f.write(f'clahe_clip_limit: {format_float(args.clahe_clip_limit)}\n')
        txt_f.write(f'clahe_tile_grid_size: {args.clahe_tile_grid_size}\n')
        txt_f.write('seed_expand_mode: same_directory_previous_next_existing_files\n')
        txt_f.write(f'seed_count: {len(input_paths)}\n')
        txt_f.write(f'loaded_image_count: {len(all_load_paths)}\n')
        txt_f.write(f'normal_scene: {args.normal_scene}\n')
        txt_f.write('score_definition: 100_exp_minus_low_unit_variation_over_scale\n')
        txt_f.write('color_channels: RGB/BGR_3_channels_used\n')
        txt_f.write(f'heatmap_dir: {heatmap_dir}\n')
        txt_f.write(f'heatmap_alpha: {format_float(args.heatmap_alpha)}\n')
        txt_f.write(f'time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
        txt_f.write('\n')
        txt_f.write(
            'scene\timage\tscore\tstable_area_ratio\tlargest_area_ratio\t'
            'mean_variation\tmean_stable_degree\tlow_unit_variation\t'
            'min_unit_variation\t'
            'max_unit_stable_degree\ttop_unit_stable_degree\t'
            'suspicious_unit_count\tcomponent_count\t'
            'valid_neighbors\ttime\theatmap\n'
        )

    avg_score = 0.0
    valid_score_count = 0
    scene_stats = defaultdict(
        lambda: {
            'sum': 0.0,
            'count': 0,
            'max_score': None,
            'max_path': None,
            'min_score': None,
            'min_path': None,
        }
    )

    pbar = tqdm(total=len(input_paths), unit='image')
    for idx, img_path in enumerate(input_paths):
        scene = input_scenes[idx]
        scene_prefix = format_scene_prefix(scene)
        img_name = os.path.basename(img_path)
        img_abs_path = os.path.abspath(img_path)
        context_paths = seed_context_paths[idx]
        neighbor_images = [
            processed_image_cache[os.path.abspath(path)] for path in context_paths
        ]

        result = calculate_lens_static_score(
            processed_image_cache[img_abs_path],
            neighbor_images,
            args.variation_thresh,
            args.min_area_ratio,
            args.morph_kernel,
            unit_width=args.unit_width,
            unit_height=args.unit_height,
            top_unit_percent=args.top_unit_percent,
            score_variation_scale=args.score_variation_scale,
            temporal_reduce=args.temporal_reduce,
            return_maps=heatmap_dir is not None,
        )

        heatmap_path = ''
        if heatmap_dir is not None:
            heatmap_path = save_lens_static_heatmap(
                img_path,
                raw_image_cache[img_abs_path],
                result,
                heatmap_dir,
                idx,
                scene,
                args.variation_thresh,
                alpha=args.heatmap_alpha,
            )

        score = result['score']
        if not math.isnan(score):
            avg_score += score
            valid_score_count += 1
            if scene:
                stat = scene_stats[scene]
                stat['sum'] += score
                stat['count'] += 1
                if stat['max_score'] is None or score > stat['max_score']:
                    stat['max_score'] = score
                    stat['max_path'] = img_path
                if stat['min_score'] is None or score < stat['min_score']:
                    stat['min_score'] = score
                    stat['min_path'] = img_path

        score_str = format_float(score)
        stable_area_ratio_str = format_float(result['stable_area_ratio'])
        largest_area_ratio_str = format_float(result['largest_area_ratio'])
        mean_variation_str = format_float(result['mean_variation'])
        mean_stable_degree_str = format_float(result['mean_stable_degree'])
        low_unit_variation_str = format_float(result['low_unit_variation'])
        min_unit_variation_str = format_float(result['min_unit_variation'])
        max_unit_stable_degree_str = format_float(result['max_unit_stable_degree'])
        top_unit_stable_degree_str = format_float(result['top_unit_stable_degree'])
        elapsed_time_str = format_float(result['elapsed_time'])

        pbar.update(1)
        pbar.set_description(f'{scene_prefix}{metric_name} of {img_name}: {score_str}')
        pbar.write(
            f'{scene_prefix}{metric_name} of {img_name}: {score_str}\t'
            f'low_unit_variation: {low_unit_variation_str}\t'
            f'min_unit_variation: {min_unit_variation_str}\t'
            f'largest_area_ratio(aux): {largest_area_ratio_str}\t'
            f'mean_variation: {mean_variation_str}\t'
            f'Time: {elapsed_time_str}s'
        )

        if sfwriter is not None:
            sfwriter.writerow(
                [
                    scene or '',
                    img_path,
                    score_str,
                    stable_area_ratio_str,
                    largest_area_ratio_str,
                    mean_variation_str,
                    mean_stable_degree_str,
                    low_unit_variation_str,
                    min_unit_variation_str,
                    max_unit_stable_degree_str,
                    top_unit_stable_degree_str,
                    result['suspicious_unit_count'],
                    result['component_count'],
                    result['valid_neighbors'],
                    elapsed_time_str,
                    heatmap_path,
                ]
            )

        if txt_f is not None:
            txt_f.write(
                f'{scene or ""}\t{img_path}\t{score_str}\t'
                f'{stable_area_ratio_str}\t{largest_area_ratio_str}\t'
                f'{mean_variation_str}\t{mean_stable_degree_str}\t'
                f'{low_unit_variation_str}\t'
                f'{min_unit_variation_str}\t{max_unit_stable_degree_str}\t'
                f'{top_unit_stable_degree_str}\t{result["suspicious_unit_count"]}\t'
                f'{result["component_count"]}\t'
                f'{result["valid_neighbors"]}\t{elapsed_time_str}s\t'
                f'{heatmap_path}\n'
            )

    pbar.close()

    avg_score = avg_score / valid_score_count if valid_score_count > 0 else float('nan')
    msg = (
        f'Average {metric_name} score of {args.target or args.target_txt} '
        f'with {valid_score_count}/{len(input_paths)} valid images is: '
        f'{format_float(avg_score)}'
    )
    print(msg)

    scene_msgs = []
    if scene_stats:
        print('Scene statistics:')
        for scene, stat in scene_stats.items():
            if stat['count'] <= 0:
                continue
            scene_avg = stat['sum'] / stat['count']
            scene_msg = (
                f'  [{scene}] {metric_name} statistics with {stat["count"]} images: '
                f'avg={format_float(scene_avg)}, '
                f'max={format_float(stat["max_score"])} ({stat["max_path"]}), '
                f'min={format_float(stat["min_score"])} ({stat["min_path"]})'
            )
            scene_msgs.append(scene_msg)
            print(scene_msg)

    separability_msgs = []
    if scene_stats:
        separability_msgs = build_scene_separability_msgs(
            scene_stats, metric_name, lower_better, args.normal_scene
        )
        for separability_msg in separability_msgs:
            print(separability_msg)

    if txt_f is not None:
        txt_f.write('\n')
        txt_f.write(msg + '\n')
        if scene_msgs:
            txt_f.write('Scene statistics:\n')
            for scene_msg in scene_msgs:
                txt_f.write(scene_msg + '\n')
        if separability_msgs:
            txt_f.write('\n')
            for separability_msg in separability_msgs:
                txt_f.write(separability_msg + '\n')
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

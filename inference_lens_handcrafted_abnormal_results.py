import argparse
import csv
import math
import os
from collections import defaultdict
from datetime import datetime
from time import time

import cv2
import numpy as np
from tqdm import tqdm

import inference_lens_tile_dirty_results as tile
import inference_lens_temporal_multi_metric_results as multi
import inference_lens_pixel_stability_observe as pix

SCORE_COLUMNS = ['noise_score', 'black_occlusion_score', 'dust_score', 'hair_score']
RESULT_COLUMNS = SCORE_COLUMNS + [
    'temporal_static_mean', 'overexp_excluded_ratio',
    'dust_blob_response', 'dust_low_contrast_response', 'dust_blur_response', 'dust_contour_response',
    'hair_line_response', 'hair_coherence_response', 'hair_contour_response', 'hair_semitransparent_response',
    'unit_width', 'unit_height', 'valid_frames'
]


def format_float(v):
    try:
        v = float(v)
    except Exception:
        return str(v)
    return 'nan' if math.isnan(v) else f'{v:.4f}'


def build_auto_txt_save_path(save_txt_dir, metric_name):
    os.makedirs(save_txt_dir, exist_ok=True)
    return os.path.join(save_txt_dir, f'{datetime.now().strftime("%Y%m%d_%H%M%S")}.{metric_name.replace(os.sep, "_")}.txt')


def build_vis_save_dir(save_txt_path):
    d = os.path.join(os.path.dirname(os.path.abspath(save_txt_path)), os.path.splitext(os.path.basename(save_txt_path))[0])
    os.makedirs(d, exist_ok=True)
    return d


def update_scene_stat(stats, scene, score, path):
    if not scene or math.isnan(float(score)):
        return
    st = stats[scene]
    st['sum'] += float(score)
    st['count'] += 1
    if st['max_score'] is None or score > st['max_score']:
        st['max_score'], st['max_path'] = float(score), path
    if st['min_score'] is None or score < st['min_score']:
        st['min_score'], st['min_path'] = float(score), path


def local_std_map(gray, ksize):
    k = tile.ensure_odd(ksize, 3)
    g = gray.astype(np.float32)
    mean = cv2.blur(g, (k, k))
    mean2 = cv2.blur(g * g, (k, k))
    return np.sqrt(np.maximum(mean2 - mean * mean, 0.0)).astype(np.float32)


def laplacian_var_map(gray, ksize):
    k = tile.ensure_odd(ksize, 3)
    lap = cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F, ksize=3)
    return local_std_map(lap, k)


def elongated_contour_prior(response, temporal_static, unit_w, unit_h):
    h, w = response.shape[:2]
    cand = np.clip(response * (0.30 + 0.70 * temporal_static), 0, 1)
    mask = tile.threshold_response_map(cand, 92.0)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    prior = np.zeros((h, w), dtype=np.float32)
    min_area = max(4.0, 0.10 * unit_w * unit_h)
    max_area = max(min_area + 1.0, 60.0 * unit_w * unit_h)
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < min_area or area > max_area:
            continue
        rect = cv2.minAreaRect(cnt)
        rw, rh = rect[1]
        if rw <= 1e-3 or rh <= 1e-3:
            continue
        length, width = max(rw, rh), min(rw, rh)
        aspect = length / (width + 1e-6)
        if aspect < 3.5:
            continue
        tmp = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(tmp, [cnt], -1, 255, -1)
        local = float(cand[tmp > 0].mean()) if np.any(tmp > 0) else 0.0
        score = np.clip(local * np.clip((aspect - 3.0) / 9.0, 0, 1), 0, 1)
        prior[tmp > 0] = np.maximum(prior[tmp > 0], score)
    return prior


def round_blob_contour_prior(response, temporal_static, unit_w, unit_h):
    h, w = response.shape[:2]
    cand = np.clip(response * (0.30 + 0.70 * temporal_static), 0, 1)
    mask = tile.threshold_response_map(cand, 94.0)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    prior = np.zeros((h, w), dtype=np.float32)
    min_area = max(3.0, 0.08 * unit_w * unit_h)
    max_area = max(min_area + 1.0, 35.0 * unit_w * unit_h)
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
        if aspect > 4.5:
            continue
        circ = np.clip(4.0 * math.pi * area / (per * per + 1e-6), 0, 1)
        tmp = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(tmp, [cnt], -1, 255, -1)
        local = float(cand[tmp > 0].mean()) if np.any(tmp > 0) else 0.0
        score = np.clip(local * (0.45 + 0.55 * circ) * math.exp(-(aspect - 1.0) / 2.0), 0, 1)
        prior[tmp > 0] = np.maximum(prior[tmp > 0], score)
    return prior


def calculate_handcrafted_metrics(frames, args):
    start = time()
    seed = frames[0]
    gray = tile.gray_from_bgr(seed)
    h, w = gray.shape[:2]
    unit_w, unit_h = tile.adaptive_unit_size(h, w)

    # Existing robust branches: keep noise and black occlusion identical to previous multi-metric logic.
    noise = multi.calculate_noise_metric(frames, args)
    black = multi.calculate_black_occlusion_metric(frames, args)

    # Stronger temporal prior: same-position stability over whole sequence, with long overexposure excluded.
    pix_res = pix.calculate_pixel_stability(frames, args)
    temporal_static = pix_res['similarity_map'].astype(np.float32)
    overexp_mask = pix_res['overexp_mask_u8'] > 0

    gray_u8 = np.clip(gray, 0, 255).astype(np.uint8)

    # ---------- dust: stable + soft blob + local low contrast/blur + anti-line ----------
    blob_rs = []
    for k in [7, 11, 17, 25]:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tile.ensure_odd(k), tile.ensure_odd(k)))
        blackhat = cv2.morphologyEx(gray_u8, cv2.MORPH_BLACKHAT, kernel).astype(np.float32)
        tophat = cv2.morphologyEx(gray_u8, cv2.MORPH_TOPHAT, kernel).astype(np.float32)
        spot = np.maximum(blackhat, args.dust_bright_weight * tophat)
        blob_rs.append(tile.normalize_01(spot, 0, max(1.0, np.percentile(spot, 98))))
    for s1, s2 in [(1.2, 2.5), (2.0, 4.5), (3.0, 8.0)]:
        dog = np.abs(cv2.GaussianBlur(gray, (0, 0), s1) - cv2.GaussianBlur(gray, (0, 0), s2))
        blob_rs.append(tile.normalize_01(dog, 0, max(1.0, np.percentile(dog, 98))))
    dust_blob = np.max(np.stack(blob_rs, axis=0), axis=0).astype(np.float32)

    contrast = local_std_map(gray, args.local_window)
    low_contrast = 1.0 - tile.normalize_01(contrast, 0, max(1.0, np.percentile(contrast, 90)))
    lap_var = laplacian_var_map(gray, args.local_window)
    blur_prior = 1.0 - tile.normalize_01(lap_var, 0, max(1.0, np.percentile(lap_var, 90)))

    # ---------- hair: stable + multi-orientation line + coherence + elongated contour ----------
    line_rs = []
    for k in tile.make_oriented_line_kernels(args.hair_line_length, args.hair_line_width, 15):
        dark = cv2.morphologyEx(gray_u8, cv2.MORPH_BLACKHAT, k).astype(np.float32)
        bright = cv2.morphologyEx(gray_u8, cv2.MORPH_TOPHAT, k).astype(np.float32)
        line_rs.append(np.maximum(dark, 0.55 * bright))
    line_raw = np.max(np.stack(line_rs, axis=0), axis=0)
    hair_line = tile.normalize_01(line_raw, 0, max(1.0, np.percentile(line_raw, 98)))

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    jxx = cv2.GaussianBlur(gx * gx, (15, 15), 0)
    jyy = cv2.GaussianBlur(gy * gy, (15, 15), 0)
    jxy = cv2.GaussianBlur(gx * gy, (15, 15), 0)
    coherence = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / (jxx + jyy + 1e-6)
    coherence = np.clip(coherence, 0, 1).astype(np.float32)

    # Hair/dust should not be saturated highlight or very black area. Prefer semi-transparent middle luma.
    mid = np.exp(-((gray - 128.0) ** 2) / (2.0 * args.semitransparent_luma_sigma ** 2)).astype(np.float32)
    not_overexp = (~overexp_mask).astype(np.float32)

    hair_contour = elongated_contour_prior(hair_line * (0.35 + 0.65 * coherence), temporal_static, unit_w, unit_h)
    dust_contour = round_blob_contour_prior(dust_blob, temporal_static, unit_w, unit_h)

    anti_line = 1.0 - tile.normalize_01(hair_line * coherence, 0, max(0.05, np.percentile(hair_line * coherence, 98)))
    dust_shape = np.clip((0.50 * dust_blob + 0.20 * low_contrast + 0.15 * blur_prior + 0.15 * dust_contour) * (0.55 + 0.45 * anti_line), 0, 1)
    hair_shape = np.clip((0.48 * hair_line + 0.22 * coherence + 0.20 * hair_contour + 0.10 * mid), 0, 1)

    dust_map = np.clip(temporal_static * dust_shape * not_overexp, 0, 1)
    hair_map = np.clip(temporal_static * hair_shape * mid * not_overexp, 0, 1)

    dust_unit, dust_heat = multi.aggregate_map_to_units(dust_map, unit_w, unit_h)
    hair_unit, hair_heat = multi.aggregate_map_to_units(hair_map, unit_w, unit_h)
    dust_score, _ = tile.adaptive_top_mean(dust_unit, True, percent=2.0, min_tiles=8)
    hair_score, _ = tile.adaptive_top_mean(hair_unit, True, percent=3.0, min_tiles=12)

    return {
        'noise_score': float(noise['noise_score']),
        'black_occlusion_score': float(black['black_occlusion_score']),
        'dust_score': 100.0 * float(dust_score),
        'hair_score': 100.0 * float(hair_score),
        'temporal_static_mean': float(temporal_static[~overexp_mask].mean()) if np.any(~overexp_mask) else 0.0,
        'overexp_excluded_ratio': float(overexp_mask.mean()),
        'dust_blob_response': float(dust_blob.mean()),
        'dust_low_contrast_response': float(low_contrast.mean()),
        'dust_blur_response': float(blur_prior.mean()),
        'dust_contour_response': float(dust_contour.mean()),
        'hair_line_response': float(hair_line.mean()),
        'hair_coherence_response': float(coherence.mean()),
        'hair_contour_response': float(hair_contour.mean()),
        'hair_semitransparent_response': float(mid.mean()),
        'unit_width': float(unit_w),
        'unit_height': float(unit_h),
        'valid_frames': float(len(frames)),
        'elapsed_time': time() - start,
        'noise_heat': noise['noise_heat'],
        'black_heat': black['black_heat'],
        'temporal_static_heat': temporal_static,
        'overexp_heat': overexp_mask.astype(np.float32),
        'dust_heat': dust_heat,
        'hair_heat': hair_heat,
        'dust_blob_heat': dust_blob,
        'hair_line_heat': hair_line,
    }


def make_prior_judgment(result, args):
    specs = [
        ('noise_score', 'noise', float(args.prior_noise_threshold)),
        ('black_occlusion_score', 'occ', float(args.prior_occlusion_threshold)),
        ('dust_score', 'dust', float(args.prior_dust_threshold)),
        ('hair_score', 'hair', float(args.prior_hair_threshold)),
    ]
    hits = []
    vals = {}
    for key, name, th in specs:
        try:
            score = float(result.get(key, float('nan')))
        except Exception:
            score = float('nan')
        vals[name] = score
        if not math.isnan(score) and score > th:
            hits.append(f'{name}:{format_float(score)}>{format_float(th)}')
    valid = [(v, k) for k, v in vals.items() if not math.isnan(v)]
    top_text = 'top=unknown'
    if valid:
        top_score, top_name = max(valid, key=lambda x: x[0])
        top_text = f'top={top_name}:{format_float(top_score)}'
    pred = 'ABNORMAL' if hits else 'NORMAL'
    return pred, hits, vals, top_text


def make_visual_judgment_lines(result, args):
    pred, hits, vals, top_text = make_prior_judgment(result, args)
    hit_text = ','.join(h.split(':', 1)[0] for h in hits) if hits else 'none'
    return [
        f'prior_judge={pred} hits={hit_text} {top_text}',
        'scores: ' + ' '.join(f'{k}={format_float(v)}' for k, v in vals.items()),
        'thr: n>{} o>{} d>{} h>{}'.format(
            format_float(args.prior_noise_threshold),
            format_float(args.prior_occlusion_threshold),
            format_float(args.prior_dust_threshold),
            format_float(args.prior_hair_threshold),
        ),
    ]


def save_visualization(img_path, seed_img, result, out_dir, index, scene, args, alpha=0.55):
    base = np.clip(seed_img, 0, 255).astype(np.uint8)
    panels = [
        base.copy(),
        tile.overlay(base, result['noise_heat'], alpha),
        tile.overlay(base, result['black_heat'], alpha),
        tile.colorize_01(result['temporal_static_heat']),
        tile.colorize_01(result['overexp_heat']),
        tile.overlay(base, result['dust_heat'], alpha),
        tile.overlay(base, result['hair_heat'], alpha),
        tile.colorize_01(np.maximum(result['dust_blob_heat'], result['hair_line_heat'])),
    ]
    seed_name = os.path.basename(img_path)
    labels = [
        f'seed={seed_name} scene={scene or "none"}',
        f'noise={format_float(result["noise_score"])}',
        f'occlusion={format_float(result["black_occlusion_score"])}',
        f'temporal_static={format_float(result["temporal_static_mean"])}',
        f'overexp_excluded={format_float(result["overexp_excluded_ratio"])}',
        f'dust={format_float(result["dust_score"])}',
        f'hair={format_float(result["hair_score"])}',
        'shape clues: dust_blob OR hair_line',
    ]
    labs = []
    judgment_lines = make_visual_judgment_lines(result, args)
    for idx, (p, label) in enumerate(zip(panels, labels)):
        p = p.copy()
        tile.put_label(p, label, 26)
        if idx == 0:
            for line_i, text in enumerate(judgment_lines):
                tile.put_label(p, text, 54 + 28 * line_i)
        labs.append(p)
    canvas = np.concatenate([
        np.concatenate(labs[:4], axis=1),
        np.concatenate(labs[4:8], axis=1),
    ], axis=0)
    out = os.path.join(out_dir, f'{index:06d}_{tile.make_safe_name(scene) if scene else "no_scene"}_{tile.make_safe_name(os.path.splitext(os.path.basename(img_path))[0])}.jpg')
    cv2.imwrite(out, canvas)
    return out


def best_threshold_summary_msgs(samples, args):
    specs = [
        ('noise_score', 'noise', multi.split_aliases(args.noise_positive_scenes)),
        ('black_occlusion_score', 'black_occlusion', multi.split_aliases(args.black_positive_scenes)),
        ('dust_score', 'dust', multi.split_aliases(args.dust_positive_scenes)),
        ('hair_score', 'hair', multi.split_aliases(args.hair_positive_scenes)),
    ]
    msgs = ['Best independent classification thresholds for each abnormal branch:']
    for metric_name, positive_name, aliases in specs:
        valid = []
        for scene, path, res in samples:
            if not scene:
                continue
            score = float(res.get(metric_name, float('nan')))
            if math.isnan(score):
                continue
            valid.append((multi.scene_matches_aliases(scene, aliases), score, scene, path))
        if not valid or not any(x[0] for x in valid) or not any(not x[0] for x in valid):
            msgs.append(f'  {positive_name}\t{metric_name}\tthreshold=nan\tinsufficient_labels')
            continue
        _, th, tp, tn, fp, fn, acc, balanced_acc = multi.build_best_binary_confusion(valid)
        precision = tp / float(tp + fp) if (tp + fp) else 0.0
        recall = tp / float(tp + fn) if (tp + fn) else 0.0
        msgs.append(
            f'  {positive_name}\t{metric_name}\tthreshold={format_float(th)}\t'
            f'tp={tp}\ttn={tn}\tfp={fp}\tfn={fn}\tacc={format_float(acc)}\t'
            f'balanced_acc={format_float(balanced_acc)}\tprecision={format_float(precision)}\trecall={format_float(recall)}'
        )
    return msgs


def main():
    parser = argparse.ArgumentParser(description='Handcrafted lens abnormal detector from typical samples: independent noise/occlusion/dust/hair branches.')
    parser.add_argument('--target_txt', required=True)
    parser.add_argument('--save_txt_dir', required=True)
    parser.add_argument('-m', '--metric_name', default='lens_handcrafted_abnormal')
    parser.add_argument('--sequence_mode', default='previous_next', choices=['previous', 'previous_next', 'even_span'])
    parser.add_argument('--window', type=int, default=5)
    parser.add_argument('--sample_count', type=int, default=7)
    parser.add_argument('--save_csv', default=None)
    parser.add_argument('--no_vis', action='store_true')
    parser.add_argument('--heatmap_alpha', type=float, default=0.55)

    # Compatibility knobs used by imported multi/pixel branches.
    parser.add_argument('--unit_width', type=int, default=4)
    parser.add_argument('--unit_height', type=int, default=3)
    parser.add_argument('--top_unit_percent', type=float, default=1.0)
    parser.add_argument('--noise_blur_ksize', type=int, default=5)
    parser.add_argument('--temporal_reduce', default='max', choices=['max', 'p75', 'median', 'mean'])
    parser.add_argument('--stable_variation_scale', type=float, default=3.0)
    parser.add_argument('--black_luma_scale', type=float, default=35.0)
    parser.add_argument('--brightness_change_ratio', type=float, default=0.30)
    parser.add_argument('--min_pixel_tol', type=float, default=1.0)
    parser.add_argument('--max_pixel_tol', type=float, default=12.0)
    parser.add_argument('--score_percentile', type=float, default=95.0)
    parser.add_argument('--series_threshold_scale', type=float, default=1.0)
    parser.add_argument('--range_threshold_scale', type=float, default=2.0)
    parser.add_argument('--pre_clip_low', type=float, default=1.0)
    parser.add_argument('--pre_clip_high', type=float, default=99.0)
    parser.add_argument('--clahe_clip', type=float, default=2.0)
    parser.add_argument('--clahe_grid', type=int, default=8)
    parser.add_argument('--overexp_suppress_threshold', type=int, default=245)
    parser.add_argument('--overexp_long_threshold', type=int, default=245)
    parser.add_argument('--overexp_persistence_ratio', type=float, default=0.70)
    parser.add_argument('--overexp_neighborhood', type=int, default=31)
    parser.add_argument('--overexp_neighborhood_ratio', type=float, default=0.35)
    parser.add_argument('--overexp_exclude_dilate', type=int, default=5)

    # Handcrafted dust/hair knobs.
    parser.add_argument('--local_window', type=int, default=21)
    parser.add_argument('--dust_bright_weight', type=float, default=0.6)
    parser.add_argument('--hair_line_length', type=int, default=31)
    parser.add_argument('--hair_line_width', type=int, default=3)
    parser.add_argument('--semitransparent_luma_sigma', type=float, default=70.0)

    # Prior thresholds learned from 20260917_144633.lens_handcrafted_abnormal.txt.
    # Final abnormal decision: noise OR occlusion OR dust OR hair score exceeds its threshold.
    parser.add_argument('--prior_noise_threshold', type=float, default=69.0000)
    parser.add_argument('--prior_occlusion_threshold', type=float, default=70.0000)
    parser.add_argument('--prior_dust_threshold', type=float, default=0.1500)
    parser.add_argument('--prior_hair_threshold', type=float, default=1.7170)

    parser.add_argument('--normal_scene', default='normal')
    parser.add_argument('--noise_positive_scenes', default='noise,noisy,low_light,dark,weak_light,噪声,暗光,弱光')
    parser.add_argument('--black_positive_scenes', default='black,occlusion,block,cover,install,安装遮挡,遮挡,黑屏')
    parser.add_argument('--dust_positive_scenes', default='dust,dirty,灰尘,尘,脏污,污渍')
    parser.add_argument('--hair_positive_scenes', default='hair,毛发,头发')
    args = parser.parse_args()

    paths, scenes = tile.get_input_paths(None, args.target_txt)
    paths, seqs, load_paths, seed_adjustments, invalid_seeds, weak_seeds = tile.build_seed_sequences(paths, args.window, args.sequence_mode, args.sample_count)

    print('Loading images for handcrafted abnormal detector...')
    print(f'Seed images: {len(paths)}; images to load: {len(load_paths)}; sequence_mode={args.sequence_mode}; window={args.window}')
    if seed_adjustments:
        print(f'Auto-shifted early seeds: {len(seed_adjustments)}')
    if weak_seeds:
        print(f'WARNING: {len(weak_seeds)} weak seeds have fewer refs than requested.')
    if invalid_seeds:
        print(f'WARNING: {len(invalid_seeds)} invalid seeds have no refs.')
    cache = {os.path.abspath(p): tile.imread_image(p) for p in tqdm(load_paths, unit='image')}

    save_txt_path = build_auto_txt_save_path(args.save_txt_dir, args.metric_name)
    vis_dir = None if args.no_vis else build_vis_save_dir(save_txt_path)
    txt_f = open(save_txt_path, 'w', encoding='utf-8')
    txt_f.write(f'metric_name: {args.metric_name}\nmetric_mode: NR\nscore_direction: larger_means_more_abnormal\n')
    txt_f.write('method: independent handcrafted branches. noise/occlusion reuse previous stable branches; dust=temporal_static*soft_blob*low_contrast*blur*anti_line; hair=temporal_static*line*coherence*elongated_contour*semi_transparent. Long overexposure is excluded from dust/hair temporal prior. Prior abnormal decision uses adjusted fixed thresholds from 20260917_145119.lens_handcrafted_abnormal.txt: noise>69.0000 OR occlusion>70.0000 OR dust>0.1500 OR hair>1.7170. Noise/occlusion are raised to reduce false alarms; dust is lowered to reduce misses.\n')
    for k in vars(args):
        txt_f.write(f'{k}: {getattr(args, k)}\n')
    txt_f.write(f'seed_count: {len(paths)}\nloaded_image_count: {len(load_paths)}\nauto_shifted_seed_count: {len(seed_adjustments)}\nweak_seed_count: {len(weak_seeds)}\ninvalid_seed_count: {len(invalid_seeds)}\nvis_dir: {vis_dir}\ntime: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')
    txt_f.write('scene\timage\t' + '\t'.join(RESULT_COLUMNS) + '\ttime\tvisualization\n')

    sf = writer = None
    if args.save_csv:
        sf = open(args.save_csv, 'w', newline='', encoding='utf-8')
        writer = csv.writer(sf)
        writer.writerow(['scene', 'image'] + RESULT_COLUMNS + ['time', 'visualization'])

    stats = {m: defaultdict(lambda: {'sum': 0.0, 'count': 0, 'max_score': None, 'max_path': None, 'min_score': None, 'min_path': None}) for m in SCORE_COLUMNS}
    samples = []
    pbar = tqdm(total=len(paths), unit='image')
    for i, p in enumerate(paths):
        scene = scenes[i]
        frames = [cache[os.path.abspath(x)] for x in seqs[i]]
        res = calculate_handcrafted_metrics(frames, args)
        vis = save_visualization(p, cache[os.path.abspath(p)], res, vis_dir, i, scene, args, args.heatmap_alpha) if vis_dir else ''
        samples.append((scene, p, res))
        for m in SCORE_COLUMNS:
            update_scene_stat(stats[m], scene, float(res[m]), p)
        elapsed = format_float(res['elapsed_time'])
        vals = [format_float(res[c]) for c in RESULT_COLUMNS]
        row = [scene or '', p] + vals + [elapsed, vis]
        if writer:
            writer.writerow(row)
        txt_f.write('\t'.join(map(str, row)) + '\n')
        prefix = f'[{scene}] ' if scene else ''
        pbar.update(1)
        pbar.set_description(f'{prefix}{args.metric_name}: dust={format_float(res["dust_score"])} hair={format_float(res["hair_score"])}')
        pbar.write(f'{prefix}{os.path.basename(p)} noise={format_float(res["noise_score"])} occ={format_float(res["black_occlusion_score"])} dust={format_float(res["dust_score"])} hair={format_float(res["hair_score"])} Time={elapsed}s')
    pbar.close()

    scene_msgs = ['Scene statistics by metric:']
    print('Scene statistics by metric:')
    for m in SCORE_COLUMNS:
        scene_msgs.append(f'[{m}]')
        print(f'[{m}]')
        for scene, st in stats[m].items():
            if st['count'] <= 0:
                continue
            line = f'  [{scene}] count={st["count"]}: avg={format_float(st["sum"] / st["count"])} max={format_float(st["max_score"])} ({st["max_path"]}) min={format_float(st["min_score"])} ({st["min_path"]})'
            scene_msgs.append(line)
            print(line)

    thresh_msgs = best_threshold_summary_msgs(samples, args)
    alarm_msgs = multi.any_branch_alarm_confusion_matrix_msgs(samples, args)
    for x in thresh_msgs + [''] + alarm_msgs:
        print(x)

    txt_f.write('\n')
    for x in scene_msgs:
        txt_f.write(x + '\n')
    txt_f.write('\n')
    for x in thresh_msgs:
        txt_f.write(x + '\n')
    txt_f.write('\n')
    for x in alarm_msgs:
        txt_f.write(x + '\n')
    txt_f.close()
    if sf:
        sf.close()

    print(f'Done! TXT results are in {save_txt_path}.')
    if vis_dir:
        print(f'Done! Visualizations are in {vis_dir}.')
    if args.save_csv:
        print(f'Done! CSV results are in {args.save_csv}.')


if __name__ == '__main__':
    main()

import argparse
import csv
import math
import os
from collections import defaultdict
from datetime import datetime
from time import time
from types import SimpleNamespace

import numpy as np
from tqdm import tqdm

# Reuse stable IO/sequence/stat code already validated in the two experiment scripts.
import inference_lens_tile_dirty_results as tile
import inference_lens_temporal_multi_metric_results as multi

SCORE_COLUMNS = ['noise_score', 'black_occlusion_score', 'dust_score', 'hair_score']
RESULT_COLUMNS = SCORE_COLUMNS + [
    'noise_response', 'black_response',
    'dirty_temporal_stable',
    'dust_shape_response', 'dust_contour_response',
    'hair_shape_response', 'hair_contour_response', 'hair_coherence_response',
    'unit_width', 'unit_height', 'valid_frames', 'time'
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


def update_scene_stat(stats, scene, score, path):
    if not scene or math.isnan(float(score)):
        return
    st = stats[scene]
    st['sum'] += float(score)
    st['count'] += 1
    if st['max_score'] is None or score > st['max_score']:
        st['max_score'], st['max_path'] = score, path
    if st['min_score'] is None or score < st['min_score']:
        st['min_score'], st['min_path'] = score, path


def calculate_fused_metrics(frames, args):
    """Independent-branch alarm metrics.

    noise/black are exactly reused from temporal_multi/multi_metric-equivalent branches.
    dust/hair use the stronger tile morphology: temporal invariance gated
    multi-scale blob/contour and multi-orientation line/coherence priors.
    """
    start = time()

    # 1) Independent noise and black-occlusion branches from temporal_multi.
    noise = multi.calculate_noise_metric(frames, args)
    black = multi.calculate_black_occlusion_metric(frames, args)

    # 2) Adaptive tile size, then temporal-invariance-gated morphology for dust/hair.
    seed = frames[0]
    gray = tile.gray_from_bgr(seed)
    h, w = gray.shape[:2]
    unit_w, unit_h = tile.adaptive_unit_size(h, w)
    stable, _ = multi.stable_map_from_stack(frames, unit_w, unit_h, args.dirty_stable_scale)
    morph = tile.advanced_dirty_morphology(gray, stable, args, unit_w, unit_h)

    dust_unit, _ = multi.aggregate_map_to_units(morph['dust_prior'], unit_w, unit_h)
    hair_unit, _ = multi.aggregate_map_to_units(morph['hair_prior'], unit_w, unit_h)
    dust_score, _ = tile.adaptive_top_mean(dust_unit, True, percent=2.0, min_tiles=8)
    hair_score, _ = tile.adaptive_top_mean(hair_unit, True, percent=3.0, min_tiles=12)

    return {
        'noise_score': float(noise['noise_score']),
        'black_occlusion_score': float(black['black_occlusion_score']),
        'dust_score': 100.0 * float(dust_score),
        'hair_score': 100.0 * float(hair_score),
        'noise_response': float(noise['noise_response']),
        'black_response': float(black['black_response']),
        'dirty_temporal_stable': float(stable.mean()),
        'dust_shape_response': float(morph['dust_shape'].mean()),
        'dust_contour_response': float(morph['dust_contour'].mean()),
        'hair_shape_response': float(morph['hair_shape'].mean()),
        'hair_contour_response': float(morph['hair_contour'].mean()),
        'hair_coherence_response': float(morph['hair_coherence'].mean()),
        'unit_width': float(unit_w),
        'unit_height': float(unit_h),
        'valid_frames': float(len(frames)),
        'elapsed_time': time() - start,
        'noise_heat': noise['noise_heat'],
        'black_heat': black['black_heat'],
        'dirty_stable_heat': stable.astype(np.float32),
        'dust_blob_heat': morph['dust_blob'],
        'dust_contour_heat': morph['dust_contour'],
        'dust_heat': morph['dust_prior'],
        'hair_line_heat': morph['hair_line'],
        'hair_coherence_heat': morph['hair_coherence'],
        'hair_contour_heat': morph['hair_contour'],
        'hair_heat': morph['hair_prior'],
    }


def build_vis_save_dir(save_txt_path):
    d = os.path.join(os.path.dirname(os.path.abspath(save_txt_path)), os.path.splitext(os.path.basename(save_txt_path))[0])
    os.makedirs(d, exist_ok=True)
    return d


def save_visualization(img_path, seed_img, result, out_dir, index, scene, alpha=0.55):
    """Save a compact diagnostic image per seed: 2 x 3 panels."""
    base = np.clip(seed_img, 0, 255).astype(np.uint8)
    panels = [
        base.copy(),
        tile.overlay(base, result['noise_heat'], alpha),
        tile.overlay(base, result['black_heat'], alpha),
        tile.colorize_01(result['dirty_stable_heat']),
        tile.overlay(base, result['dust_heat'], alpha),
        tile.overlay(base, result['hair_heat'], alpha),
    ]
    labels = [
        f'seed scene={scene or "none"}',
        f'noise={format_float(result["noise_score"])}',
        f'black={format_float(result["black_occlusion_score"])}',
        f'temporal_static={format_float(result["dirty_temporal_stable"])}',
        f'dust={format_float(result["dust_score"])}',
        f'hair={format_float(result["hair_score"])}',
    ]
    labs = []
    for panel, label in zip(panels, labels):
        panel = panel.copy()
        tile.put_label(panel, label)
        labs.append(panel)
    canvas = np.concatenate([
        np.concatenate(labs[:3], axis=1),
        np.concatenate(labs[3:6], axis=1),
    ], axis=0)
    out = os.path.join(out_dir, f'{index:06d}_{tile.make_safe_name(scene) if scene else "no_scene"}_{tile.make_safe_name(os.path.splitext(os.path.basename(img_path))[0])}.jpg')
    if not tile.cv2.imwrite(out, canvas):
        raise IOError(f'Failed to write visualization: {out}')
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
            f'tp={tp}\ttn={tn}\tfp={fp}\tfn={fn}\t'
            f'acc={format_float(acc)}\tbalanced_acc={format_float(balanced_acc)}\t'
            f'precision={format_float(precision)}\trecall={format_float(recall)}'
        )
    return msgs


def main():
    parser = argparse.ArgumentParser(
        description='Fused low-parameter lens abnormal alarm: independent noise/black/dust/hair branches; OR => alarm.'
    )
    parser.add_argument('--target_txt', required=True, help='txt with image paths and #-[] scene markers.')
    parser.add_argument('--save_txt_dir', required=True, help='directory to save timestamp.metric.txt result.')
    parser.add_argument('-m', '--metric_name', default='lens_fused_alarm')
    parser.add_argument('--sequence_mode', default='previous', choices=['previous', 'previous_next', 'even_span'])
    parser.add_argument('--window', type=int, default=10, help='previous N frames for default sequence.')
    parser.add_argument('--sample_count', type=int, default=7, help='frame count for even_span mode.')
    parser.add_argument('--save_csv', default=None)
    parser.add_argument('--no_vis', action='store_true', help='do not save per-seed diagnostic result images.')
    parser.add_argument('--heatmap_alpha', type=float, default=0.55)

    # Hidden-ish expert knobs with stable defaults; kept available but not required.
    parser.add_argument('--unit_width', type=int, default=4)
    parser.add_argument('--unit_height', type=int, default=3)
    parser.add_argument('--top_unit_percent', type=float, default=1.0)
    parser.add_argument('--noise_blur_ksize', type=int, default=5)
    parser.add_argument('--noise_dark_weight', type=float, default=1.0)
    parser.add_argument('--noise_dark_scale', type=float, default=60.0)
    parser.add_argument('--temporal_reduce', default='max', choices=['max', 'p75', 'median', 'mean'])
    parser.add_argument('--stable_variation_scale', type=float, default=3.0)
    parser.add_argument('--black_luma_scale', type=float, default=35.0)
    parser.add_argument('--dirty_stable_scale', type=float, default=8.0)
    parser.add_argument('--dust_spot_ksize', type=int, default=7)
    parser.add_argument('--dust_bright_weight', type=float, default=0.6)
    parser.add_argument('--hair_line_length', type=int, default=25)
    parser.add_argument('--hair_line_width', type=int, default=3)

    parser.add_argument('--normal_scene', default='normal')
    parser.add_argument('--noise_positive_scenes', default='noise,noisy,low_light,dark,weak_light,噪声,暗光,弱光')
    parser.add_argument('--black_positive_scenes', default='black,occlusion,block,cover,install,安装遮挡,遮挡,黑屏')
    parser.add_argument('--dust_positive_scenes', default='dust,dirty,灰尘,尘,脏污,污渍')
    parser.add_argument('--hair_positive_scenes', default='hair,毛发,头发')
    args = parser.parse_args()

    paths, scenes = tile.get_input_paths(None, args.target_txt)
    paths, seqs, load_paths, seed_adjustments, invalid_seeds, weak_seeds = tile.build_seed_sequences(
        paths, args.window, args.sequence_mode, args.sample_count
    )

    print('Loading images for fused alarm...')
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

    csv_f = None
    writer = None
    if args.save_csv:
        csv_f = open(args.save_csv, 'w', newline='', encoding='utf-8')
        writer = csv.writer(csv_f)
        writer.writerow(['scene', 'image'] + RESULT_COLUMNS + ['visualization'])

    stats = {m: defaultdict(lambda: {'sum': 0.0, 'count': 0, 'max_score': None, 'max_path': None, 'min_score': None, 'min_path': None}) for m in SCORE_COLUMNS}
    samples = []

    with open(save_txt_path, 'w', encoding='utf-8') as txt_f:
        txt_f.write(f'metric_name: {args.metric_name}\n')
        txt_f.write('score_direction: larger_means_more_abnormal\n')
        txt_f.write('method: independent branches, no composite score; final alarm is OR(noise, black_occlusion, dust, hair).\n')
        txt_f.write('noise_branch: reused temporal_multi/multi_metric-equivalent high-frequency noise residual.\n')
        txt_f.write('black_branch: reused temporal_multi/multi_metric-equivalent temporal-stable near-black occlusion.\n')
        txt_f.write('dust_hair_branch: fused from tile_dirty morphology: temporal-invariance gated blob/contour and line/coherence priors.\n')
        for k, v in vars(args).items():
            txt_f.write(f'{k}: {v}\n')
        txt_f.write(f'seed_count: {len(paths)}\nloaded_image_count: {len(load_paths)}\nauto_shifted_seed_count: {len(seed_adjustments)}\nweak_seed_count: {len(weak_seeds)}\ninvalid_seed_count: {len(invalid_seeds)}\nvis_dir: {vis_dir}\ntime: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
        if seed_adjustments:
            txt_f.write('auto_shifted_seeds:\n')
            for old_p, new_p, old_i, new_i in seed_adjustments:
                txt_f.write(f'  idx {old_i}->{new_i}: {old_p} -> {new_p}\n')
        txt_f.write('\nscene\timage\t' + '\t'.join(RESULT_COLUMNS) + '\tvisualization\n')

        for i, p in enumerate(tqdm(paths, unit='image')):
            scene = scenes[i]
            frames = [cache[os.path.abspath(x)] for x in seqs[i]]
            res = calculate_fused_metrics(frames, args)
            vis = save_visualization(p, cache[os.path.abspath(p)], res, vis_dir, i, scene, args.heatmap_alpha) if vis_dir else ''
            samples.append((scene, p, res))
            for m in SCORE_COLUMNS:
                update_scene_stat(stats[m], scene, float(res[m]), p)
            row = [scene or '', p] + [format_float(res[c]) if c != 'time' else '' for c in RESULT_COLUMNS]
            row[-1] = format_float(res['elapsed_time'])
            row = row + [vis]
            txt_f.write('\t'.join(map(str, row)) + '\n')
            if writer:
                writer.writerow(row)
            print(f'[{scene}] {os.path.basename(p)} noise={format_float(res["noise_score"])} black={format_float(res["black_occlusion_score"])} dust={format_float(res["dust_score"])} hair={format_float(res["hair_score"])} vis={vis}')

        txt_f.write('\nScene statistics by independent metric:\n')
        print('Scene statistics by independent metric:')
        for m in SCORE_COLUMNS:
            txt_f.write(f'[{m}]\n')
            print(f'[{m}]')
            for scene, st in stats[m].items():
                if st['count'] <= 0:
                    continue
                line = f'  [{scene}] count={st["count"]}: avg={format_float(st["sum"]/st["count"])}, max={format_float(st["max_score"])} ({st["max_path"]}), min={format_float(st["min_score"])} ({st["min_path"]})'
                txt_f.write(line + '\n')
                print(line)

        txt_f.write('\n')
        threshold_msgs = best_threshold_summary_msgs(samples, args)
        for line in threshold_msgs:
            txt_f.write(line + '\n')
            print(line)

        txt_f.write('\n')
        alarm_msgs = multi.any_branch_alarm_confusion_matrix_msgs(samples, args)
        for line in alarm_msgs:
            txt_f.write(line + '\n')
            print(line)

        txt_f.write('\n')
        indep = {m: [(scene, path, res[m]) for scene, path, res in samples] for m in SCORE_COLUMNS}
        independent_msgs = multi.independent_problem_confusion_matrix_msgs(indep, args)
        for line in independent_msgs:
            txt_f.write(line + '\n')
            print(line)

    if csv_f:
        csv_f.close()
    print(f'Done! TXT results are in {save_txt_path}.')
    if vis_dir:
        print(f'Done! Visualizations are in {vis_dir}.')


if __name__ == '__main__':
    main()

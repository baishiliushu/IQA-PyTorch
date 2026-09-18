import argparse
import csv
import glob
import os
from collections import defaultdict
from datetime import datetime
from time import time
from urllib.parse import unquote, urlparse

import torch
from pyiqa import create_metric
from tqdm import tqdm


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
    """Read image paths from a txt file.

    Empty lines and lines starting with '#' will be ignored. Relative paths are
    first interpreted relative to current working directory; if not found, they
    are interpreted relative to the txt file directory. Paths with 'file://'
    prefix are also supported, e.g. file:///home/user/a.jpg.

    Scene marker lines in format '#-[scene_name]' are supported. Images after a
    scene marker belong to that scene until the next scene marker or EOF.
    """
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


def format_scene_prefix(scene):
    """Format scene name for terminal output."""
    return f'[{scene}] ' if scene else ''


def format_float(value):
    """Format numeric result with 4 decimal places."""
    return f'{float(value):.4f}'


def get_scene_order_relation(scene_a, stat_a, scene_b, stat_b, lower_better):
    """Return pairwise scene separability relation based on min/max intervals.

    The raw score intervals are:
        scene_a: [min_a, max_a]
        scene_b: [min_b, max_b]

    If intervals do not overlap, the two scenes are considered separable.
    The worse/better relation depends on metric direction:
        lower_better=True  -> larger score means worse quality
        lower_better=False -> smaller score means worse quality
    """
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


def build_scene_separability_msgs(scene_stats, metric_name, lower_better):
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
        if scene.lower() == 'normal':
            normal_scene = scene
            break

    if normal_scene is None:
        msgs.append('  Scene [normal] not found, skip separability analysis.')
        return msgs

    other_scenes = [scene for scene in scenes if scene != normal_scene]
    if not other_scenes:
        msgs.append('  No other scenes except [normal], skip separability analysis.')
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
    """Inference demo for pyiqa with txt-list input and txt result saving."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-t', '--target', type=str, default=None, help='input image/folder path.'
    )
    parser.add_argument(
        '-r',
        '--ref',
        type=str,
        default=None,
        help='reference image/folder path if needed.',
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='device to run metrics on, e.g. cpu or cuda.',
    )
    parser.add_argument(
        '--metric_mode',
        type=str,
        default='FR',
        help='metric mode Full Reference or No Reference. options: FR|NR.',
    )
    parser.add_argument(
        '-m',
        '--metric_name',
        type=str,
        default='PSNR',
        help='IQA metric name, case sensitive.',
    )
    parser.add_argument(
        '--save_file', type=str, default=None, help='path to save csv results.'
    )
    parser.add_argument(
        '--target_txt',
        type=str,
        default=None,
        help='txt file containing target image paths, one path per line.',
    )
    parser.add_argument(
        '--ref_txt',
        type=str,
        default=None,
        help='txt file containing reference image paths, one path per line.',
    )
    parser.add_argument(
        '--save_txt_dir',
        type=str,
        default=None,
        help='directory to save txt results as current_time.metric_name.txt.',
    )

    # Add a --verbose flag
    parser.add_argument(
        '-v',
        '--verbose',
        action='store_true',
        help='Enable verbose output',
    )

    args = parser.parse_args()

    metric_name = args.metric_name

    if args.target is None and args.target_txt is None:
        raise ValueError('Please specify --target or --target_txt.')

    # set up IQA model
    iqa_model = create_metric(
        metric_name, metric_mode=args.metric_mode, device=args.device
    )
    metric_mode = iqa_model.metric_mode
    lower_better = iqa_model.lower_better

    input_paths, input_scenes = get_input_paths(args.target, args.target_txt)
    ref_paths = None
    ref_scenes = None
    if args.ref is not None or args.ref_txt is not None:
        ref_paths, ref_scenes = get_input_paths(args.ref, args.ref_txt)

    if metric_mode == 'FR':
        assert ref_paths is not None, 'Please specify --ref or --ref_txt for FR metric.'
        assert len(input_paths) == len(ref_paths), (
            f'Number of target images ({len(input_paths)}) and reference images '
            f'({len(ref_paths)}) must be the same.'
        )
        if ref_scenes is not None:
            assert len(input_scenes) == len(ref_scenes), (
                f'Number of target scenes ({len(input_scenes)}) and reference scenes '
                f'({len(ref_scenes)}) must be the same.'
            )

    has_scene_info = any(scene for scene in input_scenes)
    if not has_scene_info and ref_scenes is not None:
        has_scene_info = any(scene for scene in ref_scenes)

    if args.save_file:
        sf = open(args.save_file, 'w', newline='')
        sfwriter = csv.writer(sf)
    else:
        sf = None
        sfwriter = None

    save_txt_path = None
    txt_f = None
    if args.save_txt_dir:
        save_txt_path = build_auto_txt_save_path(args.save_txt_dir, metric_name)
        txt_f = open(save_txt_path, 'w', encoding='utf-8')
        txt_f.write(f'metric_name: {metric_name}\n')
        txt_f.write(f'metric_mode: {metric_mode}\n')
        txt_f.write(f'target: {args.target}\n')
        txt_f.write(f'target_txt: {args.target_txt}\n')
        txt_f.write(f'ref: {args.ref}\n')
        txt_f.write(f'ref_txt: {args.ref_txt}\n')
        txt_f.write(f'device: {args.device}\n')
        txt_f.write(f'lower_better: {lower_better}\n')
        txt_f.write(f'time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
        txt_f.write('\n')

    avg_score = 0
    test_img_num = len(input_paths)
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
    if 'fid' not in metric_name:
        pbar = tqdm(total=test_img_num, unit='image')
        for idx, img_path in enumerate(input_paths):
            img_name = os.path.basename(img_path)
            scene = input_scenes[idx]
            if scene is None and ref_scenes is not None:
                scene = ref_scenes[idx]
            if metric_mode == 'FR':
                ref_img_path = ref_paths[idx]
            else:
                ref_img_path = None

            start_time = time()
            score = iqa_model(img_path, ref_img_path).cpu().item()
            end_time = time()
            elapsed_time = end_time - start_time
            avg_score += score
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
            pbar.update(1)
            scene_prefix = format_scene_prefix(scene)
            score_str = format_float(score)
            elapsed_time_str = format_float(elapsed_time)
            pbar.set_description(
                f'{scene_prefix}{metric_name} of {img_name}: {score_str}'
            )
            pbar.write(
                f'{scene_prefix}{metric_name} of {img_name}: {score_str}\tTime: {elapsed_time_str}s'
            )
            if sfwriter is not None:
                if has_scene_info:
                    sfwriter.writerow([scene or '', img_name, score_str])
                else:
                    sfwriter.writerow([img_name, score_str])
            if txt_f is not None:
                if ref_img_path is None:
                    if has_scene_info:
                        txt_f.write(
                            f'{scene or ""}\t{img_path}\t{score_str}\t{elapsed_time_str}s\n'
                        )
                    else:
                        txt_f.write(f'{img_path}\t{score_str}\t{elapsed_time_str}s\n')
                else:
                    if has_scene_info:
                        txt_f.write(
                            f'{scene or ""}\t{img_path}\t{ref_img_path}\t{score_str}\t{elapsed_time_str}s\n'
                        )
                    else:
                        txt_f.write(
                            f'{img_path}\t{ref_img_path}\t{score_str}\t{elapsed_time_str}s\n'
                        )

        pbar.close()
        avg_score /= test_img_num
    else:
        assert os.path.isdir(args.target) and os.path.isdir(args.ref), (
            'input path must be a folder for FID.'
        )
        avg_score = iqa_model(args.target, args.ref)

    if args.verbose and torch.cuda.is_available():
        print(torch.cuda.memory_summary())

    msg = (
        f'Average {metric_name} score of {args.target or args.target_txt} '
        f'with {test_img_num} images is: {format_float(avg_score)}'
    )
    print(msg)
    scene_msgs = []
    if scene_stats:
        print('Scene statistics:')
        for scene, stat in scene_stats.items():
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
            scene_stats, metric_name, lower_better
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
    if not args.save_file and not save_txt_path:
        print('Done!')


if __name__ == '__main__':
    main()

import argparse
import csv
import glob
import os
from datetime import datetime
from time import time

import torch
from pyiqa import create_metric
from tqdm import tqdm


def read_paths_from_txt(txt_path):
    """Read image paths from a txt file.

    Empty lines and lines starting with '#' will be ignored. Relative paths are
    first interpreted relative to current working directory; if not found, they
    are interpreted relative to the txt file directory.
    """
    txt_dir = os.path.dirname(os.path.abspath(txt_path))
    paths = []
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            if os.path.isabs(line):
                path = line
            elif os.path.exists(line):
                path = line
            else:
                path = os.path.join(txt_dir, line)
            paths.append(path)
    return paths


def get_input_paths(input_path, input_txt=None):
    """Get input paths from a single image, folder, or txt file."""
    if input_txt is not None:
        return read_paths_from_txt(input_txt)

    if os.path.isfile(input_path):
        return [input_path]

    return sorted(glob.glob(os.path.join(input_path, '*')))


def build_auto_txt_save_path(save_txt_dir, metric_name):
    """Build save path with format: current_time.metric_name.txt."""
    os.makedirs(save_txt_dir, exist_ok=True)
    time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    safe_metric_name = metric_name.replace(os.sep, '_')
    return os.path.join(save_txt_dir, f'{time_str}.{safe_metric_name}.txt')


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

    input_paths = get_input_paths(args.target, args.target_txt)
    ref_paths = None
    if args.ref is not None or args.ref_txt is not None:
        ref_paths = get_input_paths(args.ref, args.ref_txt)

    if metric_mode == 'FR':
        assert ref_paths is not None, 'Please specify --ref or --ref_txt for FR metric.'
        assert len(input_paths) == len(ref_paths), (
            f'Number of target images ({len(input_paths)}) and reference images '
            f'({len(ref_paths)}) must be the same.'
        )

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
        txt_f.write(f'time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
        txt_f.write('\n')

    avg_score = 0
    test_img_num = len(input_paths)
    if 'fid' not in metric_name:
        pbar = tqdm(total=test_img_num, unit='image')
        for idx, img_path in enumerate(input_paths):
            img_name = os.path.basename(img_path)
            if metric_mode == 'FR':
                ref_img_path = ref_paths[idx]
            else:
                ref_img_path = None

            start_time = time()
            score = iqa_model(img_path, ref_img_path).cpu().item()
            end_time = time()
            elapsed_time = end_time - start_time
            avg_score += score
            pbar.update(1)
            pbar.set_description(f'{metric_name} of {img_name}: {score}')
            pbar.write(
                f'{metric_name} of {img_name}: {score}\tTime: {elapsed_time:.2f}s'
            )
            if sfwriter is not None:
                sfwriter.writerow([img_name, score])
            if txt_f is not None:
                if ref_img_path is None:
                    txt_f.write(f'{img_path}\t{score}\t{elapsed_time:.4f}s\n')
                else:
                    txt_f.write(
                        f'{img_path}\t{ref_img_path}\t{score}\t{elapsed_time:.4f}s\n'
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

    msg = f'Average {metric_name} score of {args.target or args.target_txt} with {test_img_num} images is: {avg_score}'
    print(msg)

    if txt_f is not None:
        txt_f.write('\n')
        txt_f.write(msg + '\n')
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

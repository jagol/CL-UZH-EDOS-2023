import argparse
import json
import os
import re
from typing import List

from get_loggers import get_logger


CHECKPOINT_REGEX = re.compile(r'checkpoint-(\d+)')


cp_logger = get_logger('checkpoint')


def get_checkpoints(path_out_dir: str) -> List[int]:
    step_nums = []
    for item in os.listdir(path_out_dir):
        match = re.match(CHECKPOINT_REGEX, item)
        if match:
            step_num = int(match.groups()[0])
            step_nums.append(step_num)
    return sorted(step_nums)


def get_best_checkpoint(path_out_dir: str, checkpoints: List[int], metric='eval_f1-macro') -> int:
    checkpoint_eval = {}  # step_num: score
    f_train_state = os.path.join(path_out_dir, f'checkpoint-{checkpoints[-1]}', 'trainer_state.json')
    with open(f_train_state) as fin:
        contents = json.load(fin)
        for item in contents['log_history']:
            if 'eval_f1-macro' in item:
                checkpoint_eval[item['step']] = item[metric]
    return max(checkpoint_eval, key=checkpoint_eval.get)


def delete_non_best_checkpoints(path_out_dir: str, checkpoints: List[int], best_checkpoint: int) -> None:
    assert best_checkpoint in checkpoints
    for checkpoint in checkpoints:
        if checkpoint != best_checkpoint:
            path_checkpoint = os.path.join(path_out_dir, f'checkpoint-{checkpoint}')
            cp_logger.info(f'Delete checkpoint: {path_checkpoint}')
            os.system(f'rm -r {path_checkpoint}')


def construct_best_checkpoint_path(path_out_dir: str, best_checkpoint: int) -> str:
    return os.path.join(path_out_dir, f'checkpoint-{best_checkpoint}')


def main(args: argparse.Namespace) -> None:
    checkpoints = get_checkpoints(args.path_out_dir)
    best_checkpoint = get_best_checkpoint(args.path_out_dir, checkpoints)
    cp_logger.info(f'Checkpoints available: {checkpoints}')
    cp_logger.info(f'Best checkpoint: {best_checkpoint}, delete other checkpoints.')
    delete_non_best_checkpoints(args.path_out_dir, checkpoints, best_checkpoint)
    cp_logger.info('Other checkpoints deleted.')
    # path_best_checkpoint = construct_best_checkpoint_path(args.path_out_dir, best_checkpoint)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path_out_dir', help='Path to output directory containing the checkpoints.')
    cmd_args = parser.parse_args()
    main(cmd_args)

import argparse
from collections import defaultdict
import json
from typing import Tuple, Dict

import pandas as pd
from sklearn.utils import resample, shuffle

import compute_corpus_stats
from get_loggers import get_logger

RANDOM_STATE = 42
ba_logger = get_logger('balance')


def load_dataset(path: str) -> pd.DataFrame:
    dataset_dict = defaultdict(list)
    with open(path) as fin:
        for line in fin:
            d_line = json.loads(line)
            for key in d_line:
                dataset_dict[key].append(d_line[key])
    return pd.DataFrame(dataset_dict)


def write_dataset_to_file(df: pd.DataFrame, path_out: str) -> None:
    with open(path_out, 'w') as fout:
        for index, row in df.iterrows():
            fout.write(json.dumps(dict(**row), ensure_ascii=False) + '\n')


def balance_task_by_label_value(df: pd.DataFrame, label_type: str, method: str, n: int) -> pd.DataFrame:
    df_class = df[df['label_type'] == label_type]
    df_0 = df_class[df_class['label_value'] == 0]
    df_1 = df_class[df_class['label_value'] == 1]
    if len(df_0) == len(df_1):
        ba_logger.info('Dataset already balanced. Returning unchanged.')
        return df
    return METHODS[method](df, label_type, n=n)


def get_min_maj_class(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_0 = df[df['label_value'] == 0]
    df_1 = df[df['label_value'] == 1]
    if len(df_0) == len(df_1):
        raise Exception('Classes are already balanced.')
    elif len(df_0) < len(df_1):
        df_min = df_0
        df_maj = df_1
    else:
        df_min = df_1
        df_maj = df_0
    return df_min, df_maj


def get_df_class(df: pd.DataFrame, label_type: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Get a dataframe with only rows of the specified label_type."""
    df_rel = df[df['label_type'] == label_type]
    df_other = df[df['label_type'] != label_type]
    return df_rel, df_other


def upsample_minority_class(df: pd.DataFrame, label_type: str) -> pd.DataFrame:
    df_class, df_other = get_df_class(df, label_type)
    df_min, df_maj = get_min_maj_class(df_class)
    df_min_upsampled = resample(df_min, random_state=RANDOM_STATE, n_samples=int(df_maj['id'].count()), replace=True)
    df_final = pd.concat([df_maj, df_min_upsampled, df_other])
    return df_final


def downsample_majority_class(df: pd.DataFrame, label_type: str) -> pd.DataFrame:
    df_class, df_other = get_df_class(df, label_type)
    df_min, df_maj = get_min_maj_class(df_class)
    df_maj_downsampled = resample(df_maj, random_state=RANDOM_STATE, n_samples=int(df_min['id'].count()), replace=True)
    df_final = pd.concat([df_min, df_maj_downsampled, df_other])
    return df_final


def down_up_sample_to_center(df: pd.DataFrame, label_type: str) -> pd.DataFrame:
    """Compute the average between min and maj class. Downsample maj and upsample min to that average."""
    df_class, df_other = get_df_class(df, label_type)
    df_min, df_maj = get_min_maj_class(df_class)
    avg_size = int((len(df_min) + len(df_maj)) / 2)
    df_maj_downsampled = resample(df_maj, random_state=RANDOM_STATE, n_samples=avg_size, replace=True)
    df_min_upsampled = resample(df_min, random_state=RANDOM_STATE, n_samples=avg_size, replace=True)
    df_final = pd.concat([df_min_upsampled, df_maj_downsampled, df_other])
    return df_final


def upsample_to_40_60(df: pd.DataFrame, label_type: str) -> pd.DataFrame:
    """Upsample the minority class such that a ratio of 40(minority class):60(majority class) is reached."""
    df_class, df_other = get_df_class(df, label_type)
    df_min, df_maj = get_min_maj_class(df_class)
    target_size_min = int(round((len(df_maj) / 6) * 4))
    if len(df_min) > target_size_min:
        ba_logger.info(f'Warning:Minority class already is more than 40%. Min class size: {len(df_min)}, '
                       f'40% of maj: {target_size_min}')
    df_min_upsampled = resample(df_min, random_state=RANDOM_STATE, n_samples=target_size_min, replace=True)
    df_final = pd.concat([df_min_upsampled, df_maj, df_other])
    return df_final


def upsample_to_30_70(df: pd.DataFrame, label_type: str) -> pd.DataFrame:
    """Upsample the minority class such that a ratio of 30(minority class):70(majority class) is reached."""
    df_class, df_other = get_df_class(df, label_type)
    df_min, df_maj = get_min_maj_class(df_class)
    target_size_min = int(round((len(df_maj) / 7) * 3))
    if len(df_min) > target_size_min:
        ba_logger.info(f'Warning:Minority class already is more than 30%. Min class size: '
                       f'{len(df_min)}, 30% of maj: {target_size_min}')
    df_min_upsampled = resample(df_min, random_state=RANDOM_STATE, n_samples=target_size_min, replace=True)
    df_final = pd.concat([df_min_upsampled, df_maj, df_other])
    return df_final


def upsample_max_300_possible_30_70_downsample(df: pd.DataFrame, label_type: str) -> pd.DataFrame:
    """Upsample positive class to up to 300% to reach a 30(1):70(0) ratio.

    If that is not enough to reach a 30:70 ratio, then downsample the 0-class to reach the ratio.
    """
    df_class, df_other = get_df_class(df, label_type)
    df_pos = df_class[df_class['label_value'] == 1]
    df_neg = df_class[df_class['label_value'] == 0]
    # upsample positive class to max 300%
    if len(df_pos) >= ((len(df_neg) / 7) * 3):  # if positive class makes up more than 30%
        ba_logger.info('Positive class already makes up more or equal to 30% of examples. Returning corpus unchanged.')
        return df
    if len(df_pos) * 3 > ((len(df_neg) / 7) * 3):  # if 300% upsampled positive class makes up more than 30%
        target_size_df_pos = int(round((len(df_neg) / 7) * 3))
        df_pos_upsampled = resample(df_pos, random_state=RANDOM_STATE, n_samples=target_size_df_pos, replace=True)
        return pd.concat([df_pos_upsampled, df_neg, df_other])
    else:  # if 300% upsampled positive class still makes up less than 30%
        df_pos_upsampled = resample(df_pos, random_state=RANDOM_STATE, n_samples=len(df_pos) * 3, replace=True)
        # downsample negative class to reach a 30:70 ratio
        target_size_df_neg = int(round((len(df_pos_upsampled) / 3) * 7))
        df_neg_downsampled = resample(df_neg, random_state=RANDOM_STATE, n_samples=target_size_df_neg, replace=True)
        return pd.concat([df_pos_upsampled, df_neg_downsampled, df_other])


def downsample_to_40_60(df: pd.DataFrame, label_type: str) -> pd.DataFrame:
    """Downsample the majority class such that a ratio of 40(minority class):60(majority class) is reached."""
    df_class, df_other = get_df_class(df, label_type)
    df_min, df_maj = get_min_maj_class(df_class)
    target_size_maj = int(round((len(df_min) / 4) * 6))
    if len(df_maj) <= target_size_maj:
        ba_logger.info(f'Warning:Majority class already is less than 60%. Maj class size: {len(df_maj)}, '
                       f'Target maj class size: {target_size_maj}')
    df_maj_downsampled = resample(df_maj, random_state=RANDOM_STATE, n_samples=target_size_maj, replace=True)
    df_final = pd.concat([df_min, df_maj_downsampled, df_other])
    return df_final


def downsample_to_30_70(df: pd.DataFrame, label_type: str) -> pd.DataFrame:
    """Downsample the majority class such that a ratio of 30(minority class):70(majority class) is reached."""
    df_class, df_other = get_df_class(df, label_type)
    df_min, df_maj = get_min_maj_class(df_class)
    target_size_maj = int(round((len(df_min) / 3) * 7))
    if len(df_maj) <= target_size_maj:
        ba_logger.info(f'Warning:Majority class already is less than 70%. Maj class size: {len(df_maj)}, '
                       f'Target maj class size: {target_size_maj}')
    df_maj_downsampled = resample(df_maj, random_state=RANDOM_STATE, n_samples=target_size_maj, replace=True)
    df_final = pd.concat([df_min, df_maj_downsampled, df_other])
    return df_final


def downsample_to_20_80(df: pd.DataFrame, label_type: str) -> pd.DataFrame:
    """Downsample the majority class such that a ratio of 20(minority class):80(majority class) is reached."""
    df_class, df_other = get_df_class(df, label_type)
    df_min, df_maj = get_min_maj_class(df_class)
    target_size_maj = int(round((len(df_min) / 2) * 8))
    if len(df_maj) <= target_size_maj:
        ba_logger.info(f'Warning:Majority class already is less than 80%. Maj class size: {len(df_maj)}, '
                       f'Target maj class size: {target_size_maj}')
    df_maj_downsampled = resample(df_maj, random_state=RANDOM_STATE, n_samples=target_size_maj, replace=True)
    df_final = pd.concat([df_min, df_maj_downsampled, df_other])
    return df_final


def upsample_to_n_perc(df: pd.DataFrame, label_type: str, n: int) -> pd.DataFrame:
    """Upsample all classes that make up below n% of the data to n%.

    Upsampling to n% refers to n% of the original data. If there 
    are multiple classes that are upsampled the rising overall 
    amount of data will reduce the final percentage to below n%. 
    """
    total_num = len(df)
    minimum_num = int(round((total_num / 100) * n, 0))
    label_values = df['label_value'].unique().tolist()
    upsampled_dfs = []
    for label_value in label_values:
        cur_df = df[df['label_value'] == label_value]
        if len(cur_df) < minimum_num:
            cur_df = resample(cur_df, random_state=RANDOM_STATE, n_samples=minimum_num, replace=True)
        upsampled_dfs.append(cur_df)
    df_up = pd.concat(upsampled_dfs)
    df_up_shuffled = df_up.sample(frac=1, random_state=42).reset_index(drop=True)
    return df_up_shuffled


def divide_by_corpus(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    corpus_names = df['source'].unique()
    dict_of_dfs = {}
    for cname in corpus_names:
        dict_of_dfs[cname] = df[df['source'] == cname]
    return dict_of_dfs


def balance_by_label_value(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """Balance class distributions for each task per corpus.

    Skip corpora that are not in args.corpora.
    """
    dict_of_dfs = divide_by_corpus(df)
    balanced_dfs = []
    for corpus_name in dict_of_dfs:
        if corpus_name in args.corpora:
            balanced_df = dict_of_dfs[corpus_name]
            ba_logger.info(f'Processing: {corpus_name}')
            for label_type in args.label_types:
                if label_type in balanced_df.label_type.values:
                    ba_logger.info(f'Processing label: {label_type}')
                    balanced_df = balance_task_by_label_value(balanced_df, label_type, args.method, n=args.target_percent)
            balanced_dfs.append(balanced_df)
        else:
            ba_logger.info(f'Skipping processing and appending unchanged: {corpus_name}')
            balanced_dfs.append(dict_of_dfs[corpus_name])
    return pd.concat(balanced_dfs)


# def balance_by_label_type(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
#     """Balance the distribution of label-types (tasks) for each corpus.

#     Skip corpora that are not in args.corpora.
#     """
#     dict_of_dfs = divide_by_corpus(df)
#     balanced_dfs = []
#     for corpus_name in dict_of_dfs:
#         if corpus_name in args.corpora:
#             balanced_df = dict_of_dfs[corpus_name]
#             ba_logger.info(f'Processing: {corpus_name}')
#             balanced_df = balance_corpora_by_label_type(balanced_df, args.min_HOF_ratio)
#             balanced_dfs.append(balanced_df)
#         else:
#             ba_logger.info(f'Skipping processing and appending unchanged: {corpus_name}')
#             balanced_dfs.append(dict_of_dfs[corpus_name])
#     return pd.concat(balanced_dfs)


def main(args: argparse.Namespace) -> None:
    df = load_dataset(args.input)
    if args.attribute == 'label_value':
        ba_logger.info('Balance by label-value.')
        df_out = balance_by_label_value(df, args)
    # elif args.attribute == 'label_type':
    #     ba_logger.info('Balance by label-value.')
    #     df_out = balance_by_label_type(df, args)
    else:
        raise Exception(f'Encountered unexpected balancing attribute: {args.attribute}.')
    df_final_shuffled = shuffle(df_out, random_state=RANDOM_STATE)
    ba_logger.info(f'Balanced dataset written to: {args.output}')
    write_dataset_to_file(df_final_shuffled, args.output)
    ba_logger.info('Computing statistics of resulting dataset.')
    compute_corpus_stats.compute_corpus_stats(args.output)


METHODS = {
    'upsample_to_n_perc': upsample_to_n_perc,
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='Path to input file.')
    parser.add_argument('-o', '--output', help='Path to output file.')
    parser.add_argument('-m', '--method', choices=list(METHODS.keys()),
                        help='Choose the balancing method to apply.')
    parser.add_argument('-n', '--target_percent', type=int, help='Num percent to upsample to.')
    parser.add_argument('-c', '--corpora', nargs='+', help='List of corpora to balance.')
    parser.add_argument('-a', '--attribute', choices=['label_value', 'label_type'], help='Choose which attribute to balance.')
    parser.add_argument('-l', '--label_types', nargs='+', help='Specify which label types should be balanced.')
    cmd_args = parser.parse_args()
    main(cmd_args)

import argparse
import csv
from collections import defaultdict
import re
from typing import Dict

from mappings import GLOBAL_LABEL_MAPPING
from to_label_desc_format import strip_numbering


def compute_perc(counts: defaultdict[str, int], del_none: bool = False) -> Dict[str, float]:
    perc = {}
    if del_none:
        del counts['none']
    total_num = sum(counts.values())
    for label, count in counts.items():
        perc[label] = count / total_num
    return perc


def create_label_str(counts: defaultdict[str, int], perc: Dict[str, float], label: str):
    proc_label = re.sub(r'&', r'and', label)
    return f"{proc_label} & {counts[label]} & {100*perc[label]:.1f}"


def create_label_list(counts: defaultdict[str, int], perc: Dict[str, float], label: str):
    return [label, counts[label], 100*perc[label]]


def main(args: argparse.Namespace) -> None:
    label_sexist_counts = defaultdict(int)
    label_category_counts = defaultdict(int)
    label_vector_counts = defaultdict(int)
    with open(args.input) as fin:
        reader = csv.DictReader(fin)
        for row in reader:
            label_sexist_counts[row['label_sexist']] += 1
            label_category_counts[row['label_category']] += 1
            label_vector_counts[row['label_vector']] += 1
    label_sexist_perc = compute_perc(label_sexist_counts)
    label_category_perc = compute_perc(label_category_counts, del_none=True)
    label_vector_perc = compute_perc(label_vector_counts, del_none=True)
    
    label_lists = {}
    for label in label_sexist_counts:
        label_lists[label] = create_label_list(label_sexist_counts, label_sexist_perc, label)
    for label in label_category_counts:
        label_lists[label] = create_label_list(label_category_counts, label_category_perc, label)
    for label in label_vector_counts:
        label_lists[label] = create_label_list(label_vector_counts, label_vector_perc, label)
    # convert to latex table
    # header = r"label A & \# & \% & label B & \# & \% & label C & \# & \% \\"
    # empty_label_str = '& & &'
    # hline = r'\hline'
    # hdashline = r'\hdashline'
    # rows = [
    #     f"{hline} \\\\",
    #     f"{label_str['sexist']} & {label_str['1. threats, plans to harm and incitement']} & {label_str['1.1 threats of harm']} \\\\",
    #     f"{empty_label_str} {empty_label_str} {label_str['1.2 incitement and encouragement of harm']} \\\\",
    #     f"{hdashline}",
    #     f"{empty_label_str} {label_str['2. derogation']} & {label_str['2.1 descriptive attacks']} \\\\",
    #     f"{empty_label_str} {empty_label_str} {label_str['2.2 aggressive and emotive attacks']} \\\\",
    #     f"{empty_label_str} {empty_label_str} {label_str['2.3 dehumanising attacks & overt sexual objectification']} \\\\",
    #     f"{hdashline}",
    # ]
    # print(header)
    # for row in rows:
    #     print(row)
    rows = [
        ['label A', '#', '%', 'label B', '#', '%', 'label C', '#', '%'],
        label_lists['sexist'] + label_lists['1. threats, plans to harm and incitement'] + label_lists['1.1 threats of harm'],
        6*['-'] + label_lists['1.2 incitement and encouragement of harm'],
        3*['-'] + label_lists['2. derogation'] + label_lists['2.1 descriptive attacks'],
        6*['-'] + label_lists['2.2 aggressive and emotive attacks'],
        6*['-'] + label_lists['2.3 dehumanising attacks & overt sexual objectification'],
        3*['-'] + label_lists['3. animosity'] + label_lists['3.1 casual use of gendered slurs, profanities, and insults'],
        6*['-'] + label_lists['3.2 immutable gender differences and gender stereotypes'],
        6*['-'] + label_lists['3.3 backhanded gendered compliments'],
        6*['-'] + label_lists['3.4 condescending explanations or unwelcome advice'],
        3*['-'] + label_lists['4. prejudiced discussions'] + label_lists['4.1 supporting mistreatment of individual women'],
        6*['-'] + label_lists['4.2 supporting systemic discrimination against women as a group'],
        label_lists['not sexist'] + 6*['-'],
    ]
    with open(args.output, 'w') as fout:
        writer = csv.writer(fout)
        for row in rows:
            writer.writerow(row)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='Path to input file in json format as produced by "compute_corpus_stats.py."')
    parser.add_argument('-o', '--output', help='Path to output csv-file.')
    # parser.add_argument('-c', '--corpus', help='Corpus name.')
    cmd_args = parser.parse_args()
    main(cmd_args)

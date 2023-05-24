import argparse
import csv


def main(args: argparse.Namespace) -> None:
    with open(args.texts) as ftexts, open(args.labels) as flabels, open(args.output, 'w') as fout:
        reader_t = csv.DictReader(ftexts)
        reader_l = csv.DictReader(flabels)
        writer = csv.DictWriter(fout, fieldnames=['rewire_id', 'text', args.name_label])
        writer.writeheader()
        for row_t, row_l in zip(reader_t, reader_l):
            assert row_t['rewire_id'] == row_l['rewire_id']
            writer.writerow({
                'rewire_id': row_t['rewire_id'], 
                'text': row_t['text'],
                args.name_label: row_l['label']
            })


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--texts', help='Path to csv-files containing ids and texts.')
    parser.add_argument('-l', '--labels', help='Path to csv-files containing ids and labels.')
    parser.add_argument('-o', '--output', help='Path to csv output file.')
    parser.add_argument('-n', '--name_label', help='How the output label will be called')
    cmd_args = parser.parse_args()
    main(cmd_args)

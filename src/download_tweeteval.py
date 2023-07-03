import datasets
import pandas as pd


datasets = {
    'hate': datasets.load_dataset('tweet_eval', 'hate'),
    'offensive': datasets.load_dataset('tweet_eval', 'offensive'),
    'irony': datasets.load_dataset('tweet_eval', 'irony'),
    'sentiment': datasets.load_dataset('tweet_eval', 'sentiment'),
    'emotion': datasets.load_dataset('tweet_eval', 'emotion'),
    'stance_abortion': datasets.load_dataset('tweet_eval', 'stance_abortion'),
    'stance_feminist': datasets.load_dataset('tweet_eval', 'stance_feminist')
}


splits = ['train', 'validation', 'test']
for split in splits:
    all_datasets = None
    for label_type, dataset in datasets.items():
        train_set = dataset[split].to_pandas()
        train_set['label_type'] = label_type
        if all_datasets is None:
            all_datasets = train_set
        else:
            all_datasets = pd.concat([all_datasets, train_set])

    all_datasets.to_csv(f'CL-UZH-EDOS-2023/data/TWE/TWE_{split}.csv')

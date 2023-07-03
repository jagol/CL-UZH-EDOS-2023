import datasets 
import pandas as pd

dataset = datasets.load_dataset('ucberkeley-dlab/measuring-hate-speech', 'binary')   
df = dataset['train'].to_pandas()

def infer_women_target(row) -> str:
    return row.target_gender_transgender_women or row.target_gender_women

# df['target_women'] = df.apply(infer_women_target)
df['target_women'] = df['target_gender_transgender_women'] + df['target_gender_women']

df_new = df[['comment_id', 'text', 'hatespeech', 'sentiment', 'respect', 'insult', 'humiliate', 'status', 'dehumanize', 'violence', 'genocide', 'target_gender', 'target_women']]

df_new.to_csv('CL-UZH-EDOS-2023/data/MHS/mhs_relevant_columns.csv')

from har_utils import *
import numpy as np
import pandas as pd

spacing = 250

def extract(outfile, df, labels):
    new_df = pd.DataFrame()
    for col in df:
        start = 0
        end = spacing
        if col == 'timestamp' or col == 'label':
            continue
        means = []
        stdevs = []
        skews = []
        kert = []
        labels_new = []
        while end < len(labels):
            means.append(index_avg(df[col], start, end))
            stdevs.append(index_stdev(df[col], start, end))
            skews.append(index_skew(df[col], start, end))
            kert.append(index_kertosis(df[col], start, end))
            labels_new.append(index_mode(df['label'], start, end))
            start += spacing
            end += spacing
        new_df[col + "_mean"] = means
        new_df[col + "_stdev"] = stdevs
        new_df[col + "_skew"] = skews
        new_df[col + "_kertosis"] = kert
    new_df['label'] = labels_new
    new_df.to_csv(outfile, index=False)        
        
    
def main():
    filename = "dataset/1.csv"
    df = pd.read_csv(filename)
    labels = df['label']
    extract("out.csv", df, labels)
        
if __name__ == "__main__":
    main()

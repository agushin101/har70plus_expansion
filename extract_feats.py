from har_utils import *
import numpy as np
import pandas as pd

spacing = 250

def extract_mean(filename):
    df = pd.read_csv(filename)
    labels = df['label']
    new_df = pd.DataFrame()
    
    for col in df:
        start = 0
        end = spacing
        if col == 'timestamp' or col == 'label':
            continue
        means = []
        labels_new = []
        while end < len(labels):
            means.append(index_avg(df[col], start, end))
            labels_new.append(index_mode(df['label'], start, end))
            start += spacing
            end += spacing
        new_df[col] = means
    new_df['label'] = labels_new
    new_df.to_csv('means.csv', index=False)
                
        
    
def main():
    i = 1
    filename = "dataset/" + str(i) + ".csv"
    extract_mean(filename)
        
if __name__ == "__main__":
    main()

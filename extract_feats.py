from har_utils import *
import numpy as np
import pandas as pd

spacing = 250

def process_grav(df):
    new_df = pd.DataFrame()
    for col in df:
        start = 0
        end = spacing
        
        means = []
        stdevs = []
        medians = []
        variations = []
        first_quart = []
        third_quart = []
        minimums = []
        maximums = []
        
        while end < len(df):
            means.append(index_avg(df[col], start, end))
            stdevs.append(index_stdev(df[col], start, end))
            variations.append(index_variation(df[col], start, end))

            minimum, first, med, third, maximum = index_quantiles(df[col], start, end)

            minimums.append(minimum)
            first_quart.append(first)
            medians.append(med)
            third_quart.append(third)
            maximums.append(maximum)
            
            start += spacing
            end += spacing
            
        new_df[col + "_mean"] = means
        new_df[col + "_median"] = medians
        new_df[col + "_stdev"] = stdevs
        new_df[col + "_variation"] = variations
        new_df[col + "_25th_percent"] = first_quart
        new_df[col + "_75th_percent"] = third_quart
        new_df[col + "_minimum"] = minimums
        new_df[col + "_maximum"] = maximums
        
    return new_df

def process_mov(df):
    return None

def process_labels(df):
    return None
    
def main():
    for i in range(1, 19):
        grav = pd.read_csv("processing/grav_mov/" + str(i) + "_grav.csv")
        mov = pd.read_csv("processing/grav_mov/" + str(i) + "_mov.csv")
        labels = pd.read_csv("processing/labels/" + str(i) + "_y.csv")

        grav_df = process_grav(grav)

        grav_df.to_csv("processing/" + str(i) + ".csv", index=False)

        
        
if __name__ == "__main__":
    main()

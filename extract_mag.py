#Extracts the magnitudes to compute the 8 signals suggested by Logacjov et al

from har_utils import *
import numpy as np
import pandas as pd


def main():
    sample_freq = 50 #50 Hz
    for i in range(1, 19):
        filename = "dataset/" + str(i) + ".csv"
        df = pd.read_csv(filename)
        new_df = pd.DataFrame()

        back_mag = np.sqrt((df['back_x'] * df['back_x']) + (df['back_y'] * df['back_y']) + (df['back_z'] * df['back_z']))
        thigh_mag = np.sqrt((df['thigh_x'] * df['thigh_x']) + (df['thigh_y'] * df['thigh_y']) + (df['thigh_z'] * df['thigh_z']))
        
        new_df['back_x'] = df['back_x']
        new_df['back_y'] = df['back_y']
        new_df['back_z'] = df['back_z']
        new_df['back_mag'] = back_mag

        new_df['thigh_x'] = df['thigh_x']
        new_df['thigh_y'] = df['thigh_y']
        new_df['thigh_z'] = df['thigh_z']
        new_df['thigh_mag'] = thigh_mag
        
        new_df['label'] = df['label']
        new_df.to_csv("processing/magnitude/" + str(i) + "_m.csv", index=False)

if __name__ == "__main__":
    main()

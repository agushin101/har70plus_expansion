from har_utils import *
import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq


def main():
    sample_freq = 50 #50 Hz
    for i in range(1, 19):
        filename = "dataset/" + str(i) + ".csv"
        df = pd.read_csv(filename)
        labels = df['label']
        new_df = pd.DataFrame()

        for col in df:
            start = 0
            end = len(labels)
            if col == 'timestamp' or col == 'label':
                continue
        
            grav, mov = extract_gravity_movement(df[col], start, end)
            fy = fft(mov.values)
            fx = fftfreq(len(fy), 1/sample_freq)
            new_df[col + "_grav"] = grav
            new_df[col + "_mov"] = mov
            new_df[col + "_freq_mag"] = fx
            new_df[col + "_freq_pow"] = fy
        new_df['label'] = labels
        new_df.to_csv("transformed/" + str(i) + "_split.csv", index=False)

if __name__ == "__main__":
    main()

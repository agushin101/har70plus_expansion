from har_utils import *
import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq


def main():
    sample_freq = 50 #50 Hz
    for i in range(1, 19):
        filename = "processing/magnitude/" + str(i) + "_m.csv"
        df = pd.read_csv(filename)
        new_df_mov = pd.DataFrame()
        new_df_grav = pd.DataFrame()

        for col in df:
            start = 0
            end = len(df)
        
            grav, mov = extract_gravity_movement(df[col], start, end)
            
            fy = fft(mov.values)
            fy = (2 * np.abs(fy)) / len(fy)
            fx = fftfreq(len(fy), 1/sample_freq)
            
            new_df_grav[col + "_grav"] = grav
            new_df_mov[col + "_mov"] = mov
            new_df_mov[col + "_freq_mag"] = fx
            new_df_mov[col + "_freq_pow"] = fy
        new_df_mov.to_csv("processing/grav_mov/" + str(i) + "_mov.csv", index=False)
        new_df_grav.to_csv("processing/grav_mov/" + str(i) + "_grav.csv", index=False)

if __name__ == "__main__":
    main()

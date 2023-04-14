from har_utils import *
import numpy as np
import pandas as pd

spacing = 250

def process_grav(df):
    new_df = pd.DataFrame()
    for col in df:
        start = 0
        end = spacing

        #Features to be extracted
        
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
    new_df = pd.DataFrame()
    for col in df:
        start = 0
        end = spacing

        #Features to be extracted

        skew = []
        kurtosis = []
        sig_en = []

        while end < len(df):
            skew.append(index_skew(df[col], start, end))
            kurtosis.append(index_kurtosis(df[col], start, end))
            sig_en.append(signal_energy(df[col], start, end))

            start += spacing
            end += spacing

        new_df[col + "_skew"] = skew
        new_df[col + "_kurtosis"] = kurtosis
        new_df[col + "_sig_en"] = sig_en

    return new_df

def process_fft(df):
    new_df = pd.DataFrame()
    prefixes = ['back_x', 'back_y', 'back_z', 'back_mag', 'thigh_x', 'thigh_y', 'thigh_z', 'thigh_mag']

    for prefix in prefixes:
        start = 0
        end = spacing

        #Features to be extracted

        fmag_mean = []
        fmag_stdev = []
        dom_freq = []
        dom_freq_mag = []
        sig_power = []

        while end < len(df):
            freq = prefix + "_freq_mag"
            mag = prefix + "_freq_pow"

            fmag_mean.append(index_avg(df[mag], start, end))
            fmag_stdev.append(index_stdev(df[mag], start, end))

            dom, dom_mag = extract_dom(df[freq], df[mag], start, end)
            dom_freq.append(dom)
            dom_freq_mag.append(dom_mag)

            sig_power.append(signal_power(df[mag], start, end))

            start += spacing
            end += spacing

        new_df[prefix + "_fmag_mean"] = fmag_mean
        new_df[prefix + "_fmag_stdev"] = fmag_stdev
        new_df[prefix + "_dom_freq"] = dom_freq
        new_df[prefix + "_dom_freq_mag"] = dom_freq_mag
        new_df[prefix + "_sig_power"] = sig_power
    
    return new_df

def process_labels(df):
    new_df = pd.DataFrame()
    
    start = 0
    end = spacing
    modes = []

    while end < len(df):
        modes.append(index_mode(df['label'], start, end))
        start += spacing
        end += spacing

    new_df['label'] = modes
    
    return new_df
    
def main():
    for i in range(1, 19):
        grav = pd.read_csv("processing/grav_mov/" + str(i) + "_grav.csv")
        mov = pd.read_csv("processing/grav_mov/" + str(i) + "_mov.csv")
        fft = pd.read_csv("processing/grav_mov/" + str(i) + "_fft.csv")
        labels = pd.read_csv("processing/labels/" + str(i) + "_y.csv")

        grav_df = process_grav(grav)
        mov_df = process_mov(mov)
        fft_df = process_fft(fft)
        final_labels = process_labels(labels)

        concat = pd.concat([grav_df, mov_df, fft_df], axis=1)

        concat.to_csv("processing/features/" + str(i) + ".csv", index=False)
        final_labels.to_csv("processing/final_labels/" + str(i) + "yf.csv", index=False)
        
        
if __name__ == "__main__":
    main()

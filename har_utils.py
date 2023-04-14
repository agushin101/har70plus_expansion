import numpy as np
import scipy.stats as stats
import scipy.signal as sig

###
# Statistics
###

#Compute various statistics on specified subarrays

def index_avg(arr, start, end):
    return np.mean(arr[start:end])

def index_stdev(arr, start, end):
    return np.std(arr[start:end])

def index_skew(arr, start, end):
    return stats.skew(arr[start:end])

def index_kurtosis(arr, start, end):
    return stats.kurtosis(arr[start:end])

def index_variation(arr, start, end):
    return stats.variation(arr[start:end])

def index_mode(arr, start, end):
    return stats.mode(arr[start:end], axis=None).mode[0]

def index_variation(arr, start, end):
    return stats.variation(arr[start:end])

def index_quantiles(arr, start, end):
    sub = arr[start:end]
    minimum = np.amin(sub)
    first = np.quantile(sub, .25)
    med = np.quantile(sub, .5)
    third = np.quantile(sub, .75)
    maximum = np.amax(sub)
    return minimum, first, med, third, maximum
    

###
# Signal Processing
###

#Extract the gravity and movement components, as defined by
#Logacjov et al

def extract_gravity_movement(arr, start, end):
    subarr = arr[start:end]
    filt = sig.butter(4, 1, btype='lowpass', analog=True, output='sos')
    grav = sig.sosfilt(filt, subarr)
    mov = np.subtract(subarr, grav)

    return grav, mov

def signal_energy(arr, start, end):
    sub = arr[start:end]
    return (np.sum(sub**2)) / len(sub)

def signal_power(arr, start, end):
    sub = arr[start:end]
    power_density = (sub**2) / (len(sub)**2)
    return np.sum(power_density)

def extract_dom(freqs, mags, start, end):
    freqs = freqs[start:end]
    mags = mags[start:end]

    dom = np.argmax(mags)

    return freqs.iloc[dom], mags.iloc[dom]

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

def index_kertosis(arr, start, end):
    return stats.kurtosis(arr[start:end])

def index_variation(arr, start, end):
    return stats.variation(arr[start:end])

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



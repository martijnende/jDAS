from scipy.signal import tukey, butter, filtfilt, sosfiltfilt
import scipy.fft  # Scipy's FFT package is faster than Numpy's


def _butter_bandpass(lowcut, highcut, fs, order, mode):
    """Compute the numerator and denominator of a butterworth filter
    
    Parameters
    ----------
    
    lowcut : float
        Lower passband frequency (sames units as `fs`). Set to `-1` for a lowpass filter
    highcut : float
        Upper passband frequency (same units as `fs`). Set to `-1` for a highpass filter
    fs : float
        Sampling frequency
    order : int
        Filter order. Note that this order is doubled due to the forward and backward pass
    mode : str, default "sos"
        Type of filter design. Using b/a coefficients ("ba") is faster, but less stable than second-order sections ("sos") for higher orders
        
    Returns
    -------
    
    if mode is "ba":
    
    b : `numpy.array`
        Array of denominator coefficients
    a : `numpy.array`
        Array of numerator coefficients    
        
    if mode is "sos":
    
    sos : `numpy.array`
        Second-order sections representation
    
    """
    
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    
    if low < 0:
        Wn = high
        btype = "lowpass"
    elif high < 0:
        Wn = low
        btype = "highpass"
    else:
        Wn = [low, high]
        btype = "bandpass"
        
    if mode == "ba":
        b, a = butter(order, Wn, btype=btype, output="ba")
        return b, a
    
    if mode == "sos":
        sos = butter(order, Wn, btype=btype, output="sos")
        return sos
    
    return False


def taper_filter(arr, fmin, fmax, samp, order=2, mode="sos"):
    """Apply a taper and a butterworth filter to the input data
    
    Filter the data in a given frequency band using a 4th-order butterworth filter. The filter is applied forward and backward in time to prevent phase shifts.
    
    To avoid boundary effects, a Tukey taper is first applied to the data.
    
    Parameters
    ----------
    
    arr : `numpy.array`
        Input data. The filter will be applied to the last axis
    fmin : float
        Lower passband frequency (sames units as `samp_DAS`). Set to `-1` for a lowpass filter
    fmax : float
        Upper passband frequency (same units as `samp_DAS`). Set to `-1` for a highpass filter
    samp : float
        Sampling frequency
    order : int, default 2
        Filter order. Note that this order is doubled due to the forward and backward pass
    mode : str, default "sos"
        Type of filter design. Using b/a coefficients ("ba") is faster, but less stable than second-order sections ("sos") for higher orders
    
    Returns
    -------
    
    arr_filt : `numpy.array`
        Filtered data
    
    """
    
    if mode not in ("ba", "sos"):
        print(f"Filter type {mode} not recognised")
        print("Valid options: {'ba', 'sos'}")
        return False
    
    window_time = tukey(arr.shape[-1], 0.1)
    arr_wind = arr * window_time
    
    if mode == "ba":
        b, a = _butter_bandpass(fmin, fmax, samp, order, mode)
        arr_filt = filtfilt(b, a, arr_wind, axis=-1)
    
    if mode == "sos":
        sos = _butter_bandpass(fmin, fmax, samp, order, mode)
        arr_filt = sosfiltfilt(sos, arr_wind, axis=-1)
    
    return arr_filt
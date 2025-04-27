#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 15:41:24 2020

@author: dong

This module provides functions for Hilbert transform analysis of signals.
Useful for extracting instantaneous frequency and amplitude information.
"""
import numpy as np
import math
from scipy.signal import hilbert


def FAhilbert(imfs, dt):
    """
    Performs Hilbert transformation on IMFs (Intrinsic Mode Functions).
    
    Parameters:
    -----------
    imfs : numpy.ndarray
        Array of Intrinsic Mode Functions with shape (n_imfs, signal_length)
    dt : float
        Time step between samples
        
    Returns:
    --------
    tuple
        f : numpy.ndarray
            Instantaneous frequencies for each IMF
        a : numpy.ndarray
            Instantaneous amplitudes for each IMF
    """
    n_imfs = imfs.shape[0]
    f = []
    a = []
    
    for i in range(n_imfs - 1):
        # Get the current IMF
        inst_imf = imfs[i, :]
        
        # Calculate amplitude and phase using Hilbert transform
        inst_amp, phase = hilb(inst_imf, unwrap=True)
        
        # Calculate instantaneous frequency from phase
        inst_freq = (1/dt) * np.diff(phase) / (2 * math.pi)
        
        # Append last value to match original length
        inst_freq = np.insert(inst_freq, len(inst_freq), inst_freq[-1])
        inst_amp = np.insert(inst_amp, len(inst_amp), inst_amp[-1])
        
        f.append(inst_freq)
        a.append(inst_amp)
        
    return np.asarray(f).T, np.asarray(a).T


def hilb(s, unwrap=False):
    """
    Performs Hilbert transformation on a signal.
    
    Parameters:
    -----------
    s : numpy.ndarray
        Input signal
    unwrap : bool
        If True, unwraps the phase to make it continuous
        If False, phase is in range [-π, π)
        
    Returns:
    --------
    tuple
        amp : numpy.ndarray
            Instantaneous amplitude of signal
        phase : numpy.ndarray
            Instantaneous phase of signal
    """
    # Apply Hilbert transform
    H = hilbert(s)
    
    # Calculate amplitude and phase
    amp = np.abs(H)
    phase = np.arctan2(H.imag, H.real)
    
    # Unwrap phase if requested
    if unwrap:
        phase = np.unwrap(phase)

    return amp, phase










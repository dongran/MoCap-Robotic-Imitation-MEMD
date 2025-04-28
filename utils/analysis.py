#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 2023

@author: dongran

Analysis utilities for MoCap-Robotic-Imitation-MEMD project
"""

import numpy as np
from scipy.fftpack import fft, ifft
from utils.CurvatureNumeric import calc_curvature_range_kutta, calc_curvature_2_derivative
from utils.data_processing import min_max


def load_motion_data(file_path):
    """
    Load motion data from CSV file
    
    Parameters:
    -----------
    file_path : str
        Path to CSV file containing motion data
        
    Returns:
    --------
    tuple
        motors: List of motor names
        data: Motion data as numpy array
        time: Time values
        dt: Time increment
        l: Original CSV data as list
    """
    import csv
    with open(file_path) as f:
        reader = csv.reader(f)
        l = [row for row in reader]

    motors = l[0][1:]
    data = np.array(l[1:], dtype=np.float64)[:,1:]
    time = np.array(l[1:], dtype=np.float64)[:,0]
    dt = time[1] - time[0]
    
    return motors, data, time, dt, l


def calculate_hilbert_spectrum(imf, dt):
    """
    Calculate instantaneous frequencies and amplitudes using Hilbert transform
    
    Parameters:
    -----------
    imf : numpy.ndarray
        Intrinsic mode functions from MEMD
    dt : float
        Time increment
        
    Returns:
    --------
    tuple
        freq: Frequency data
        amp: Amplitude data
        allfreq: Frequency data for all motors
        allamp: Amplitude data for all motors
    """
    import utils.ht as hs
    
    allfreq = []
    allamp = []
    
    for i in range(imf.shape[1]):
        freq_i, amp_i = hs.FAhilbert(imf[:,i], dt)
        allfreq.append(freq_i)
        allamp.append(amp_i)

    allfreq = np.array(allfreq)
    allamp = np.array(allamp)

    freq = np.mean(allfreq, axis=0)
    amp = np.linalg.norm(allamp, ord=2, axis=0)
    
    return freq, amp, allfreq, allamp


def evaluate_imf_performance(imf, data, t, dt):
    """
    Evaluate performance metrics of various IMF combinations
    
    Parameters:
    -----------
    imf : numpy.ndarray
        Intrinsic mode functions from MEMD
    data : numpy.ndarray
        Original motion data
    t : numpy.ndarray
        Time values
    dt : float
        Time increment
        
    Returns:
    --------
    tuple
        Various performance metrics
    """
    from utils.MEMD_all import avgF, avgA
    
    # Calculate average frequencies and amplitudes
    avgfreq = avgF(imf, dt)
    avgamp = avgA(imf, dt)
    freqSD = np.std(avgfreq, axis=1)
    ampSD = np.std(avgamp, axis=1)
    allavgfreq = np.mean(avgF(imf, dt), axis=1)
    allavgamp = np.mean(avgA(imf, dt), axis=1)
    
    # Initialize lists to store metrics
    sd = []
    mse = []
    maxspeed = []
    maxaccelerate = []
    maxcurvature = []
    avgspeed = []
    avgaccelerate = []
    avgcurvature = []
    
    # Evaluate metrics for each IMF combination
    for i in range(imf.shape[0]-1):
        tmpdata = np.sum(imf[i:], axis=0)   
        sd.append(np.std(tmpdata, axis=1))
        mse.append(np.mean((data.T - tmpdata) ** 2, axis=1))
        
        # Calculate maximum speed and acceleration for each motor
        max_speeds = []
        max_accels = []
        for j in range(tmpdata.shape[0]):
            max_speeds.append(np.max(np.abs(np.diff(tmpdata[j]))))
            max_accels.append(np.max(np.abs(np.diff(np.diff(tmpdata[j])))))
        
        maxspeed.append(max_speeds)
        maxaccelerate.append(max_accels)
        
        curvature_range_kutta = []
        for j in range(tmpdata.shape[0]):
            curvature_range_kutta.append(calc_curvature_range_kutta(t, tmpdata[j]))  
            
        max_curvs = []
        avg_curvs = []
        for j in range(tmpdata.shape[0]):
            max_curvs.append(np.max(np.abs(curvature_range_kutta[j])))
            avg_curvs.append(np.mean(np.abs(curvature_range_kutta[j])))
            
        maxcurvature.append(max_curvs)
        avgcurvature.append(avg_curvs)
        
        avgspeed.append(np.mean(np.abs(np.diff(tmpdata))))
        avgaccelerate.append(np.mean(np.abs(np.diff(np.diff(tmpdata)))))
    
    # Add metrics for the trend (last IMF)
    sd.append(np.std(imf[-1], axis=1))
    mse.append(np.mean((data.T - imf[-1]) ** 2, axis=1))
    
    # Calculate maximum speed and acceleration for each motor
    max_speeds = []
    max_accels = []
    for j in range(imf[-1].shape[0]):
        max_speeds.append(np.max(np.abs(np.diff(imf[-1][j]))))
        max_accels.append(np.max(np.abs(np.diff(np.diff(imf[-1][j])))))
    
    maxspeed.append(max_speeds)
    maxaccelerate.append(max_accels)
    
    avgspeed.append(np.mean(np.abs(np.diff(imf[-1]))))
    avgaccelerate.append(np.mean(np.abs(np.diff(np.diff(imf[-1])))))
    
    curvature_range_kutta = []
    for j in range(imf[-1].shape[0]):
        curvature_range_kutta.append(calc_curvature_range_kutta(t, imf[-1][j]))  
        
    max_curvs = []
    avg_curvs = []
    for j in range(imf[-1].shape[0]):
        max_curvs.append(np.max(np.abs(curvature_range_kutta[j])))
        avg_curvs.append(np.mean(np.abs(curvature_range_kutta[j])))
        
    maxcurvature.append(max_curvs)
    avgcurvature.append(avg_curvs)
    
    # Convert lists to arrays
    sd = np.array(sd)
    mse = np.array(mse)
    maxspeed = np.array(maxspeed)
    maxaccelerate = np.array(maxaccelerate)
    avgspeed = np.array(avgspeed)
    avgaccelerate = np.array(avgaccelerate)
    maxcurvature = np.array(maxcurvature)
    avgcurvature = np.array(avgcurvature)
    
    # Calculate mean and sum metrics
    meansd = np.mean(sd, axis=1)
    meanmse = np.mean(mse, axis=1)
    sumsd = np.sum(sd, axis=1)
    summse = np.sum(mse, axis=1)
    
    return (avgfreq, allavgfreq, sd, mse, maxspeed, maxaccelerate, maxcurvature, 
            avgspeed, avgaccelerate, avgcurvature, meansd, meanmse, sumsd, summse)


def perform_fourier_transform_analysis(avgfreq, data, t, dt, nIMFs, sd):
    """
    Perform Fourier Transform (FT) analysis for comparison with MEMD
    
    Parameters:
    -----------
    avgfreq : numpy.ndarray
        Average frequency data from MEMD
    data : numpy.ndarray
        Original motion data
    t : numpy.ndarray
        Time values
    dt : float
        Time increment
    nIMFs : int
        Number of IMFs
    sd : numpy.ndarray
        Standard deviation metrics
        
    Returns:
    --------
    tuple
        FT performance metrics
    """
    np.random.seed(0)
    avgfreqFT = np.zeros([avgfreq.shape[0] + 1, avgfreq.shape[1]]) 
    avgfreqFT[1:] = avgfreq  # Cut same frequency components as HHT

    ftdata = np.zeros([nIMFs, sd.shape[1], data.shape[0]])
    
    # Apply Fourier Transform with different cutoff frequencies
    for i in range(nIMFs):
        for j in range(sd.shape[1]):
            fc = avgfreqFT[i,j]
            N = data.shape[0]
            f = data[:,j]
            meanData = np.mean(f)
            f = f - meanData

            F = fft(f)
            F_abs = np.abs(F)
            F_abs_amp = F_abs / N * 2
            F_abs_amp[0] = F_abs_amp[0] / 2
            
            fq = np.linspace(0, 1.0/dt, N)
            
            F_ifft = ifft(F)
            F_ifft_real = F_ifft.real
            
            F2 = np.copy(F)
            if fc != 0:
                F2[(fq >= fc)] = 0
            
            F2_abs = np.abs(F2)
            F2_abs_amp = F2_abs / N * 2
            F2_abs_amp[0] = F2_abs_amp[0] / 2
            
            F2_ifft = np.fft.ifft(F2)
            F2_ifft_real = F2_ifft.real * 2
            
            tmpft = F2_ifft_real + meanData
            ftdata[i,j] = tmpft
    
    # Calculate FT performance metrics
    sdFT = []
    mseFT = []
    maxspeedFT = []
    maxaccelerateFT = []
    maxcurvatureFT = []
    avgspeedFT = []
    avgaccelerateFT = []
    avgcurvatureFT = []
    
    for i in range(ftdata.shape[0]):
        tmpdata = ftdata[i]
        sdFT.append(np.std(ftdata[i], axis=1))
        mseFT.append(np.mean((data.T - ftdata[i]) ** 2, axis=1))
        
        # Calculate maximum speed and acceleration for each motor
        max_speeds = []
        max_accels = []
        for j in range(tmpdata.shape[0]):
            max_speeds.append(np.max(np.abs(np.diff(tmpdata[j]))))
            max_accels.append(np.max(np.abs(np.diff(np.diff(tmpdata[j])))))
        
        maxspeedFT.append(max_speeds)
        maxaccelerateFT.append(max_accels)

        curvature_range_kutta = []
        for j in range(tmpdata.shape[0]):
            curvature_range_kutta.append(calc_curvature_2_derivative(t, tmpdata[j]))  
            
        max_curvs = []
        avg_curvs = []
        for j in range(tmpdata.shape[0]):
            max_curvs.append(np.max(np.abs(curvature_range_kutta[j])))
            avg_curvs.append(np.mean(np.abs(curvature_range_kutta[j])))
            
        maxcurvatureFT.append(max_curvs)
        avgcurvatureFT.append(avg_curvs)
        
        avgspeedFT.append(np.mean(np.abs(np.diff(tmpdata))))
        avgaccelerateFT.append(np.mean(np.abs(np.diff(np.diff(tmpdata)))))
    
    # Convert lists to arrays
    mseFT = np.array(mseFT)
    sdFT = np.array(sdFT)
    maxspeedFT = np.array(maxspeedFT)
    maxaccelerateFT = np.array(maxaccelerateFT)
    avgspeedFT = np.array(avgspeedFT)
    avgaccelerateFT = np.array(avgaccelerateFT)
    maxcurvatureFT = np.array(maxcurvatureFT)
    avgcurvatureFT = np.array(avgcurvatureFT)
    
    # Calculate mean and sum metrics
    meansdFT = np.mean(sdFT, axis=1)
    meanmseFT = np.mean(mseFT, axis=1)
    sumsdFT = np.sum(sdFT, axis=1)
    summseFT = np.sum(mseFT, axis=1)
    
    return (ftdata, sdFT, mseFT, maxspeedFT, maxaccelerateFT, avgspeedFT, avgaccelerateFT,
            maxcurvatureFT, avgcurvatureFT, meansdFT, meanmseFT, sumsdFT, summseFT)


def optimize_robot_motion(imf, maxspeedFT, mse, alpha):
    """
    Optimize robot motion using a combination of speed and accuracy metrics
    
    Parameters:
    -----------
    imf : numpy.ndarray
        Intrinsic mode functions from MEMD
    maxspeedFT : numpy.ndarray
        Maximum speed metrics for FT
    mse : numpy.ndarray
        MSE metrics for MEMD
    alpha : float
        Weighting parameter for optimization
        
    Returns:
    --------
    list
        Optimal IMF indices for each motor
    """
    phi = maxspeedFT[:-1]
    sigma = mse[:-1]
    beta = 1 - alpha
    
    imfmotorlist = []
    
    for i in range(imf.shape[1]):
        testp = min_max(phi[:,i])
        tests = min_max(sigma[:,i])
        
        indca = alpha * testp + beta * tests
        
        imfmotorlist.append(np.argmin(indca) + 1)
    
    return imfmotorlist


def generate_output_csv(output_path, l, t, data, slow, imfmotorlist, imf, ftdata):
    """
    Generate CSV output files for the original and optimized motions
    
    Parameters:
    -----------
    output_path : str
        Path to output directory
    l : list
        Original CSV data as list
    t : numpy.ndarray
        Time values
    data : numpy.ndarray
        Original motion data
    slow : float
        Slow parameter
    imfmotorlist : list
        Optimal IMF indices for each motor
    imf : numpy.ndarray
        Intrinsic mode functions from MEMD
    ftdata : numpy.ndarray
        Fourier transform data
    """
    import csv
    import os
    
    # Original motion
    out = np.deg2rad(data)
    
    listout = []
    listout.append(l[0])
    
    for i in range(out.shape[0]):
        listout.append([t[i] * slow + 3] + list(out[i]))
    
    with open(os.path.join(output_path, 'out_org.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerows(listout)
    
    # MEMD optimized motion
    out = np.zeros([imf.shape[2], imf.shape[1]])
    for i, item in enumerate(imfmotorlist):
        out[:,i] = np.sum(imf[item:,i], axis=0).T
    
    out = np.deg2rad(out)
    
    listout = []
    listout.append(l[0])
    
    for i in range(out.shape[0]):
        listout.append([t[i] * slow + 3] + list(out[i]))
    
    with open(os.path.join(output_path, 'out_hhtAgr.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerows(listout)
    
    # FT optimized motion
    out = np.zeros([ftdata.shape[2], ftdata.shape[1]])
    for i, item in enumerate(imfmotorlist):
        out[:,i] = ftdata[item - 1, i].T
    
    out = np.deg2rad(out)
    
    listout = []
    listout.append(l[0])
    
    for i in range(out.shape[0]):
        listout.append([t[i] * slow + 3] + list(out[i]))
    
    with open(os.path.join(output_path, 'out_hhtFT.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerows(listout) 
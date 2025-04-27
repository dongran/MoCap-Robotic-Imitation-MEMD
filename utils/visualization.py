#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 2023

@author: dongran

Visualization utilities for MoCap-Robotic-Imitation-MEMD project
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_original_motion(t, data, motors, important_motors):
    """
    Plot original motion data
    
    Parameters:
    -----------
    t : numpy.ndarray
        Time values
    data : numpy.ndarray
        Motion data
    motors : list
        Names of motors
    important_motors : list
        List of important motors to highlight
    """
    plt.figure(figsize=(15,5), dpi=150)
    plt.rcParams["font.family"] = "Times New Roman" 
    plt.rcParams["font.size"] = 20
    plt.rcParams['axes.linewidth'] = 3
    plt.rcParams['legend.fontsize'] = 20

    plt.plot(t, data[:,12], label="RHipYawPitch", linewidth=3)   
    for i in range(data.shape[1]):
        if motors[i][1:] in important_motors:
            plt.plot(t, data[:,i], label=motors[i][1:], linewidth=3)

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    plt.title("Original motion capture data")
    plt.xlabel("Time [$s$]")
    plt.ylabel("Angle [$deg$]")
    plt.xlim(t[0], t[-1])
    plt.savefig("orignal.pdf", bbox_inches='tight')
    plt.show()


def plot_decomposed_motion(t, imf, motors, important_motors):
    """
    Plot MEMD decomposed motion
    
    Parameters:
    -----------
    t : numpy.ndarray
        Time values
    imf : numpy.ndarray
        Intrinsic mode functions from MEMD
    motors : list
        Names of motors
    important_motors : list
        List of important motors to highlight
    """
    nIMFs = imf.shape[0]
    plt.figure(figsize=(18,25), dpi=150)
    lineint = 2
    
    for n in range(nIMFs - 1):
        plt.subplot(nIMFs+1, 3, n+1)
        
        plt.plot(t, imf[n,12,:], label="RHipYawPitch", linewidth=lineint)   
        for i in range(imf.shape[1]):
            if motors[i][1:] in important_motors:
                plt.plot(t, imf[n,i], label=motors[i][1:], linewidth=lineint)
        
        plt.xlabel("Time [$s$]")
        plt.ylabel("Angle [$deg$]")
        plt.xlim(t[0], t[-1])
        plt.title("IMF$_{%i}$" %(n+1))
        
    plt.subplot(nIMFs+1, 3, nIMFs)
    plt.plot(t, imf[nIMFs-1,12,:], label="RHipYawPitch", linewidth=lineint)   
    for i in range(imf.shape[1]):
        if motors[i][1:] in important_motors:
            plt.plot(t, imf[nIMFs-1,i,:], label=motors[i][1:], linewidth=lineint)

    plt.xlabel("Time [$s$]")
    plt.ylabel("Angle [$deg$]")
    plt.xlim(t[0], t[-1])
    plt.title("Trend")
    plt.tight_layout()
    plt.savefig("MEMDdecomposed.pdf", bbox_inches='tight')
    plt.show()


def plot_motion_spectrum(t, freq, amp):
    """
    Plot motion spectrum
    
    Parameters:
    -----------
    t : numpy.ndarray
        Time values
    freq : numpy.ndarray
        Frequency data
    amp : numpy.ndarray
        Amplitude data
    """
    n = freq.shape[0]
    m = amp.shape[1]
    
    t2 = np.zeros((n, m))
    for i in range(m):
        t2[:,i] = t

    plt.figure(dpi=200, figsize=(15,9))
    plt.rcParams['axes.linewidth'] = 2
    plt.rcParams["font.family"] = "Times New Roman" 
    plt.rcParams["font.size"] = 40

    plt.scatter(t2, freq, s=100, c=np.log(amp[:-1]), cmap='jet')
    ax = plt.gca()
    ax.set_facecolor([0.0,0.0,0.5])
    plt.ylim(0, 50)
    plt.xlim(0, t[-1])
    plt.xlabel('Time [sec]')
    plt.ylabel('Frequency [Hz]', labelpad=20) 
    plt.colorbar(label="Amplitude [deg] (log scale)")
    plt.show()

    # Plot detailed spectrums for different IMF ranges
    plot_detailed_spectrums(t, t2, freq, amp)


def plot_detailed_spectrums(t, t2, freq, amp):
    """
    Plot detailed motion spectrums for different IMF ranges
    
    Parameters:
    -----------
    t : numpy.ndarray
        Time values
    t2 : numpy.ndarray
        Time values matrix for plotting
    freq : numpy.ndarray
        Frequency data
    amp : numpy.ndarray
        Amplitude data
    """
    numplot = 20
    amplog = np.log(amp[:-1])
    maxv = np.max(amplog)
    minv = np.min(amplog)

    plt.rcParams["font.size"] = 25
    plt.rcParams['axes.linewidth'] = 2
    plt.figure(dpi=200, figsize=(10,15))

    # IMF 1-2
    plt.subplot(3, 1, 1)
    plt.scatter(t2[:,:2], freq[:,:2], s=numplot, c=amplog[:,:2], cmap='jet')
    ax = plt.gca()
    ax.set_facecolor([0.0,0.0,0.5])
    plt.ylim(0, 50)
    plt.xlim(0, t[-1])
    plt.clim(minv, maxv)
    plt.title("IMF$_{1-2}$")
    plt.xlabel('Time [$s$]')
    plt.ylabel('Frequency [Hz]')
    plt.colorbar(label="Amplitude [$deg$] (log scale)")

    # IMF 3-5
    plt.subplot(3, 1, 2)
    plt.scatter(t2[:,2:5], freq[:,2:5], s=numplot, c=amplog[:,2:5], cmap='jet')
    ax = plt.gca()
    ax.set_facecolor([0.0,0.0,0.5])
    plt.ylim(0, 20)
    plt.xlim(0, t[-1])
    plt.clim(minv, maxv)
    plt.title("IMF$_{3-5}$")
    plt.xlabel('Time [$s$]')
    plt.ylabel('Frequency [Hz]')
    plt.colorbar(label="Amplitude [$deg$] (log scale)")

    # IMF 6-8
    plt.subplot(3, 1, 3)
    plt.scatter(t2[:,5:], freq[:,5:], s=numplot, c=amplog[:,5:], cmap='jet')
    ax = plt.gca()
    ax.set_facecolor([0.0,0.0,0.5])
    plt.ylim(0, 3)
    plt.xlim(0, t[-1])
    plt.clim(minv, maxv)
    plt.title("IMF$_{6-8}$")
    plt.xlabel('Time [$s$]')
    plt.ylabel('Frequency [Hz]')
    plt.colorbar(label="Amplitude [$deg$] (log scale)")

    plt.tight_layout()
    plt.savefig("spectrum.pdf", bbox_inches='tight')
    plt.show()


def plot_comparison_metrics(allavgfreq, maxspeed, maxspeedFT, meanmse, meanmseFT, dt):
    """
    Plot comparison of MEMD and FT metrics
    
    Parameters:
    -----------
    allavgfreq : numpy.ndarray
        Average frequency data
    maxspeed : numpy.ndarray
        Maximum speed metrics for MEMD
    maxspeedFT : numpy.ndarray
        Maximum speed metrics for FT
    meanmse : numpy.ndarray
        Mean MSE for MEMD
    meanmseFT : numpy.ndarray
        Mean MSE for FT
    dt : float
        Time increment
    """
    # Create x-axis values and labels
    x = list(range(0, 7))
    values = []
    for i in range(1, len(allavgfreq)):
        values.append(str(round(allavgfreq[i], 2)))

    # Set up plot parameters
    plt.rcParams["font.size"] = 40
    plt.rc('legend', fontsize=40)
    plt.rcParams['axes.linewidth'] = 3

    # Create figure
    plt.figure(dpi=200, figsize=(15,15))
    
    # Maximum speed plot
    plt.subplot(2, 1, 1)
    plt.plot(maxspeed[1:-1] / dt, marker="o", markersize=15, linestyle="-", linewidth=5, label='MEMD')
    plt.plot(maxspeedFT[1:-1] / dt, marker="^", markersize=15, linestyle="-", linewidth=5, label='FT')
    plt.xticks(x, values)
    plt.title("Maximum motor speed")
    plt.xlabel("Low-pass filter [Hz]")
    plt.ylabel("Angle velocity [deg/sec]")
    plt.legend()

    # MSE plot
    plt.subplot(2, 1, 2)
    plt.plot(meanmse[1:-1], marker="o", markersize=15, linestyle="-", linewidth=5, label='MEMD')
    plt.plot(meanmseFT[1:-1], marker="^", markersize=15, linestyle="-", linewidth=5, label='FT')
    plt.xticks(x, values)
    plt.xlabel("Low-pass filter [Hz]")
    plt.ylabel("Angle [deg]")
    plt.title("Average MSE")
    plt.legend()
    
    plt.tight_layout()
    plt.show()


def plot_motor_metrics(avgcurvature, avgcurvatureFT, mse, mseFT, motors, important_motors, allavgfreq):
    """
    Plot metrics for individual motors
    
    Parameters:
    -----------
    avgcurvature : numpy.ndarray
        Average curvature metrics for MEMD
    avgcurvatureFT : numpy.ndarray
        Average curvature metrics for FT
    mse : numpy.ndarray
        MSE metrics for MEMD
    mseFT : numpy.ndarray
        MSE metrics for FT
    motors : list
        Names of motors
    important_motors : list
        List of important motors to highlight
    allavgfreq : numpy.ndarray
        Average frequency data
    """
    # Create x-axis values and labels
    x = list(range(0, 7))
    values = []
    for i in range(1, len(allavgfreq)):
        values.append(str(round(allavgfreq[i], 2)))
    
    # Set up plot parameters for curvature plots
    linw = 3
    msize = 15
    plt.rcParams["font.size"] = 22
    plt.rc('legend', fontsize=22)
    plt.rcParams['axes.linewidth'] = 3
    plt.figure(dpi=200, figsize=(25,10))
    
    # Plot curvature for each important motor
    count = 1
    for j in range(avgcurvature.shape[1]):
        motor = j
        
        if motors[j][1:] in important_motors:
            plt.subplot(3, 4, count)
            plt.plot(avgcurvature[1:-1, motor], marker="o", markersize=msize, linestyle="-", linewidth=linw, label='MEMD')
            plt.plot(avgcurvatureFT[1:-1, motor], marker="^", markersize=msize, linestyle="-", linewidth=linw, label='FT')
            plt.xticks(x, values)
            plt.title(motors[j])
            plt.xlabel("Low-pass filter [Hz]")
            plt.ylabel("Curvature [$rad/m$]")
            plt.legend()
            count += 1

    # Plot average curvature for all motors
    plt.subplot(3, 4, count)
    plt.plot(np.mean(avgcurvature[1:-1], axis=1), marker="o", markersize=msize, linestyle="-", linewidth=linw, label='MEMD')
    plt.plot(np.mean(avgcurvatureFT[1:-1], axis=1), marker="^", markersize=msize, linestyle="-", linewidth=linw, label='FT')
    plt.xticks(x, values)
    plt.title("Average all motors")
    plt.xlabel("Low-pass filter [Hz]")
    plt.ylabel("Curvature [$rad/m$]")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("evaljointC.pdf", bbox_inches='tight')
    plt.show()
    
    # Set up plot parameters for MSE plots
    plt.figure(dpi=200, figsize=(25,10))
    
    # Plot MSE for each important motor
    count = 1
    for j in range(mse.shape[1]):
        motor = j
        
        if motors[j][1:] in important_motors:
            plt.subplot(3, 4, count)
            plt.plot(mse[1:-1, motor], marker="o", markersize=msize, linestyle="-", linewidth=linw, label='MEMD')
            plt.plot(mseFT[1:-1, motor], marker="^", markersize=msize, linestyle="-", linewidth=linw, label='FT')
            plt.xticks(x, values)
            plt.xlabel("Low-pass filter [Hz]")
            plt.ylabel("MSE [$deg$]")
            plt.title(motors[j])
            plt.legend()
            count += 1
    
    # Plot average MSE for all motors
    plt.subplot(3, 4, count)
    plt.plot(np.mean(mse[1:-1], axis=1), marker="o", markersize=msize, linestyle="-", linewidth=linw, label='MEMD')
    plt.plot(np.mean(mseFT[1:-1], axis=1), marker="^", markersize=msize, linestyle="-", linewidth=linw, label='FT')
    plt.xticks(x, values)
    plt.xlabel("Low-pass filter [Hz]")
    plt.ylabel("MSE [$deg$]")
    plt.title("Average all motors")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("evaljointM.pdf", bbox_inches='tight')
    plt.show() 
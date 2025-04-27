#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 08:35:16 2022

@author: randong

Motion optimization using Multivariate Empirical Mode Decomposition (MEMD)
"""

import argparse
import numpy as np

# Import local modules
from utils.MEMD_all import memd
from utils.visualization import (
    plot_original_motion, plot_decomposed_motion, plot_motion_spectrum,
    plot_comparison_metrics, plot_motor_metrics
)
from utils.analysis import (
    load_motion_data, calculate_hilbert_spectrum, evaluate_imf_performance,
    perform_fourier_transform_analysis, optimize_robot_motion, generate_output_csv
)


def parse_arguments():
    """
    Parse command line arguments
    
    Returns:
    --------
    argparse.Namespace
        Parsed arguments
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process MEMD options.')
    parser.add_argument('--input_csv', type=str, required=True, help='Path to the input CSV file')
    parser.add_argument('--output_opt', type=str, required=True, help='Path to the output directory')
    parser.add_argument('--slow', type=float, default=1.0, help='Slow parameter (default: 1.0)')
    parser.add_argument('--alpha', type=float, default=0.5, help='Alpha parameter (default: 0.5)')

    return parser.parse_args()


def main():
    """
    Main function to execute the MEMD optimization process
    """
    # Parse command line arguments
    args = parse_arguments()
    
    # Load motion data
    motors, data, time, dt, l = load_motion_data(args.input_csv)
    
    # Select first 400 frames and convert to degrees
    data = np.rad2deg(data[:400])
    t = np.linspace(0, data.shape[0] * dt, data.shape[0])
    
    # Define important motors to focus on
    important_motors = ["HeadYaw", "HeadPitch", "LShoulderPitch", "LShoulderRoll", 
                       "RShoulderPitch", "RShoulderRoll", "RHipYawPitch", "RHipRoll", 
                       "RHipPitch", "LHipYawPitch", "LHipRoll", "LHipPitch"]
    
    # Plot original motion
    plot_original_motion(t, data, motors, important_motors)
    
    # Apply MEMD to decompose the motion
    imf = memd(data)
    nIMFs = imf.shape[0]
    
    # Plot decomposed motion
    plot_decomposed_motion(t, imf, motors, important_motors)
    
    # Calculate Hilbert transform to get frequency and amplitude
    freq, amp, allfreq, allamp = calculate_hilbert_spectrum(imf, dt)
    
    # Plot motion spectrum
    plot_motion_spectrum(t, freq, amp)
    
    # Evaluate IMF performance
    (avgfreq, allavgfreq, sd, mse, maxspeed, maxaccelerate, maxcurvature, 
     avgspeed, avgaccelerate, avgcurvature, meansd, meanmse, sumsd, 
     summse) = evaluate_imf_performance(imf, data, t, dt)
    
    # Perform Fourier Transform analysis for comparison
    (ftdata, sdFT, mseFT, maxspeedFT, maxaccelerateFT, avgspeedFT, 
     avgaccelerateFT, maxcurvatureFT, avgcurvatureFT, meansdFT, meanmseFT, 
     sumsdFT, summseFT) = perform_fourier_transform_analysis(avgfreq, data, t, dt, nIMFs, sd)
    
    # Plot comparison metrics
    #plot_comparison_metrics(allavgfreq, maxspeed, maxspeedFT, meanmse, meanmseFT, dt)
    
    # Plot metrics for individual motors
    plot_motor_metrics(avgcurvature, avgcurvatureFT, mse, mseFT, motors, important_motors, allavgfreq)
    
    # Optimize robot motion
    imfmotorlist = optimize_robot_motion(imf, maxspeedFT, mse, args.alpha)
    
    # Generate output CSV files
    generate_output_csv(args.output_opt, l, t, data, args.slow, imfmotorlist, imf, ftdata)


if __name__ == "__main__":
    main()

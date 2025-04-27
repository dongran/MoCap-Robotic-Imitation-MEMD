#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 18:03:45 2022

@author: randong

This module provides various methods for calculating curvature of 2D data points.
Each function takes X and Y coordinates and returns curvature values.
"""

import math
import numpy as np


def make_angles_continuous(angles):
    """
    Make an array of angles continuous by removing jumps of 2π.
    
    Parameters:
    -----------
    angles : numpy.ndarray
        Array of angles in radians
        
    Returns:
    --------
    numpy.ndarray
        Array of continuous angles
    """
    angles = np.array(angles)
    for i in range(len(angles)-1):
        d_angle = angles[i+1] - angles[i]
        if d_angle >= np.pi:
            angles[i+1:] -= 2.0 * np.pi
        elif d_angle <= -np.pi:
            angles[i+1:] += 2.0 * np.pi
    return angles


def calc_curvature_range_kutta(x, y):
    """
    Calculate curvature using Range-Kutta method.
    
    Parameters:
    -----------
    x : numpy.ndarray
        X coordinates of points
    y : numpy.ndarray
        Y coordinates of points
        
    Returns:
    --------
    numpy.ndarray
        Curvature values for each point
    """
    # Calculate distances between consecutive points
    dists = np.array([np.hypot(dx, dy) for dx, dy in zip(np.diff(x), np.diff(y))])
    
    # Initialize curvature array with zeros at endpoints
    curvatures = [0.0, 0.0]
    
    # Calculate curvature for internal points
    for i in np.arange(2, len(x)-1):
        dx = (x[i+1] - x[i])/dists[i]
        dy = (y[i+1] - y[i])/dists[i]
        ddx = (x[i-2] - x[i-1] - x[i] + x[i+1])/(2*dists[i]**2)
        ddy = (y[i-2] - y[i-1] - y[i] + y[i+1])/(2*dists[i]**2)
        
        # Curvature formula: κ = (y''x' - x''y')/((x'^2 + y'^2)^(3/2))
        curvature = (ddy * dx - ddx * dy) / ((dx ** 2 + dy ** 2) ** 1.5)
        curvatures.append(curvature)
    
    # Add zero at the end
    curvatures.append(0.0)
    return curvatures


def calc_curvature_2_derivative(x, y):
    """
    Calculate curvature using second derivatives.
    This method uses weighted central differences for first and second derivatives.
    
    Parameters:
    -----------
    x : numpy.ndarray
        X coordinates of points
    y : numpy.ndarray
        Y coordinates of points
        
    Returns:
    --------
    numpy.ndarray
        Curvature values for each point
    """
    # Initialize curvature array with zero for the first point
    curvatures = [0.0]
    
    # Calculate curvature for internal points
    for i in np.arange(1, len(x)-1):
        # First differences
        dxn = x[i] - x[i - 1]
        dxp = x[i + 1] - x[i]
        dyn = y[i] - y[i - 1]
        dyp = y[i + 1] - y[i]
        
        # Calculate distances
        dn = np.hypot(dxn, dyn)
        dp = np.hypot(dxp, dyp)
        
        # Weighted first derivatives
        dx = 1.0 / (dn + dp) * (dp / dn * dxn + dn / dp * dxp)
        dy = 1.0 / (dn + dp) * (dp / dn * dyn + dn / dp * dyp)
        
        # Second derivatives
        ddx = 2.0 / (dn + dp) * (dxp / dp - dxn / dn)
        ddy = 2.0 / (dn + dp) * (dyp / dp - dyn / dn)
        
        # Curvature formula: κ = (y''x' - x''y')/((x'^2 + y'^2)^(3/2))
        curvature = (ddy * dx - ddx * dy) / ((dx ** 2 + dy ** 2) ** 1.5)
        curvatures.append(curvature)
    
    return curvatures


def calc_curvature_with_yaw_diff(x, y, yaw):
    """
    Calculate curvature using difference in yaw angles.
    
    Parameters:
    -----------
    x : numpy.ndarray
        X coordinates of points
    y : numpy.ndarray
        Y coordinates of points
    yaw : numpy.ndarray
        Yaw angles in radians
        
    Returns:
    --------
    numpy.ndarray
        Curvature values for each point
    """
    # Calculate distances between consecutive points
    dists = np.array([np.hypot(dx, dy) for dx, dy in zip(np.diff(x), np.diff(y))])
    
    # Make yaw angles continuous and calculate differences
    d_yaw = np.diff(make_angles_continuous(yaw))
    
    # Curvature is the change in angle divided by distance
    curvatures = d_yaw / dists
    
    # Add zero at the end to match input length
    curvatures = np.concatenate([curvatures, [0.0]])

    return curvatures


def calc_curvature_circle_fitting(x, y, npo=1):
    """
    Calculate curvature using circle fitting method.
    
    Parameters:
    -----------
    x : numpy.ndarray
        X coordinates of points
    y : numpy.ndarray
        Y coordinates of points
    npo : int
        Number of points on each side to use for fitting circles
        e.g., npo=1 uses 3 points (current + 1 on each side)
        npo=2 uses 5 points (current + 2 on each side)
        
    Returns:
    --------
    list
        Curvature values for each point
    """
    cv = []
    n_data = len(x)

    for i in range(n_data):
        # Calculate indices for points to use in circle fitting
        lind = max(0, i - npo)
        hind = min(n_data, i + npo + 1)

        xs = x[lind:hind]
        ys = y[lind:hind]
        
        # Fit circle to points
        (cxe, cye, re) = CircleFitting(xs, ys)

        if len(xs) >= 3:
            # Determine sign of curvature 
            c_index = int((len(xs) - 1) / 2.0)
            
            # Cross product to determine sign
            sign = (xs[0] - xs[c_index]) * (ys[-1] - ys[c_index]) - (
                    ys[0] - ys[c_index]) * (xs[-1] - xs[c_index])

            # Check if points form a straight line
            a = np.array([xs[0] - xs[c_index], ys[0] - ys[c_index]])
            b = np.array([xs[-1] - xs[c_index], ys[-1] - ys[c_index]])
            
            # Calculate angle between vectors
            dot_product = np.dot(a, b)
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            
            # Avoid division by zero
            if norm_a * norm_b == 0:
                theta = 0
            else:
                theta = math.degrees(math.acos(
                    dot_product / (norm_a * norm_b)))

            if theta == 180.0:
                cv.append(0.0)  # straight line has zero curvature
            elif sign > 0:
                cv.append(1.0 / -re)  # negative curvature
            else:
                cv.append(1.0 / re)  # positive curvature
        else:
            cv.append(0.0)  # not enough points for circle fitting

    return cv


def CircleFitting(x, y):
    """
    Fit a circle to a set of points using least squares method.
    
    Parameters:
    -----------
    x : numpy.ndarray
        X coordinates of points
    y : numpy.ndarray
        Y coordinates of points
        
    Returns:
    --------
    tuple
        (cx, cy, r) where (cx, cy) is the center of the circle and r is the radius
    """
    # Calculate sums for the linear system
    sumx = sum(x)
    sumy = sum(y)
    sumx2 = sum([ix ** 2 for ix in x])
    sumy2 = sum([iy ** 2 for iy in y])
    sumxy = sum([ix * iy for (ix, iy) in zip(x, y)])

    # Matrix F for the linear system F * T = G
    F = np.array([[sumx2, sumxy, sumx],
                  [sumxy, sumy2, sumy],
                  [sumx, sumy, len(x)]])

    # Vector G for the linear system
    G = np.array([[-sum([ix ** 3 + ix * iy ** 2 for (ix, iy) in zip(x, y)])],
                  [-sum([ix ** 2 * iy + iy ** 3 for (ix, iy) in zip(x, y)])],
                  [-sum([ix ** 2 + iy ** 2 for (ix, iy) in zip(x, y)])]])

    # Solve the linear system F * T = G
    try:
        T = np.linalg.inv(F).dot(G)
    except np.linalg.LinAlgError:
        # Singular matrix - cannot fit a circle
        return 0, 0, float("inf")

    # Calculate center coordinates
    cxe = float(T[0] / -2)
    cye = float(T[1] / -2)

    # Calculate radius
    try:
        re = math.sqrt(cxe ** 2 + cye ** 2 - T[2])
    except:
        # Cannot calculate radius
        return cxe, cye, float("inf")
        
    return cxe, cye, re

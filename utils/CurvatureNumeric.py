#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 18:03:45 2022

@author: randong
"""

import math
import numpy as np


def make_angles_continuous(angles):
    angles = np.array(angles)
    for i in range(len(angles)-1):
        d_angle = angles[i+1] - angles[i]
        if d_angle >= np.pi:
            angles[i+1:] -= 2.0 * np.pi
        elif d_angle <= -np.pi:
            angles[i+1:] += 2.0 * np.pi
    return angles


def calc_curvature_range_kutta(x, y):
    dists = np.array([np.hypot(dx, dy) for dx, dy in zip(np.diff(x), np.diff(y))])
    curvatures = [0.0, 0.0]
    for i in np.arange(2, len(x)-1):
        dx = (x[i+1] - x[i])/dists[i]
        dy = (y[i+1] - y[i])/dists[i]
        ddx = (x[i-2] - x[i-1] - x[i] + x[i+1])/(2*dists[i]**2)
        ddy = (y[i-2] - y[i-1] - y[i] + y[i+1])/(2*dists[i]**2)
        curvature = (ddy * dx - ddx * dy) / ((dx ** 2 + dy ** 2) ** 1.5)
        curvatures.append(curvature)
    curvatures.append(0.0)
    return curvatures


def calc_curvature_2_derivative(x, y):

    curvatures = [0.0]
    for i in np.arange(1, len(x)-1):
        dxn = x[i] - x[i - 1]
        dxp = x[i + 1] - x[i]
        dyn = y[i] - y[i - 1]
        dyp = y[i + 1] - y[i]
        dn = np.hypot(dxn, dyn)
        dp = np.hypot(dxp, dyp)
        dx = 1.0 / (dn + dp) * (dp / dn * dxn + dn / dp * dxp)
        ddx = 2.0 / (dn + dp) * (dxp / dp - dxn / dn)
        dy = 1.0 / (dn + dp) * (dp / dn * dyn + dn / dp * dyp)
        ddy = 2.0 / (dn + dp) * (dyp / dp - dyn / dn)
        curvature = (ddy * dx - ddx * dy) / ((dx ** 2 + dy ** 2) ** 1.5)
        curvatures.append(curvature)
    return curvatures


def calc_curvature_with_yaw_diff(x, y, yaw):
    dists = np.array([np.hypot(dx, dy) for dx, dy in zip(np.diff(x), np.diff(y))])
    d_yaw = np.diff(make_angles_continuous(yaw))
    curvatures = d_yaw / dists
    curvatures = np.concatenate([curvatures, [0.0]])

    return curvatures


def calc_curvature_circle_fitting(x, y, npo=1):
    """
    Calc curvature
    x,y: x-y position list
    npo: the number of points using Calculation curvature
    ex) npo=1: using 3 point
        npo=2: using 5 point
        npo=3: using 7 point
    """

    cv = []
    n_data = len(x)

    for i in range(n_data):
        lind = i - npo
        hind = i + npo + 1

        if lind < 0:
            lind = 0
        if hind >= n_data:
            hind = n_data

        xs = x[lind:hind]
        ys = y[lind:hind]
        (cxe, cye, re) = CircleFitting(xs, ys)

        if len(xs) >= 3:
            # sign evaluation
            c_index = int((len(xs) - 1) / 2.0)
            sign = (xs[0] - xs[c_index]) * (ys[-1] - ys[c_index]) - (
                    ys[0] - ys[c_index]) * (xs[-1] - xs[c_index])

            # check straight line
            a = np.array([xs[0] - xs[c_index], ys[0] - ys[c_index]])
            b = np.array([xs[-1] - xs[c_index], ys[-1] - ys[c_index]])
            theta = math.degrees(math.acos(
                np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))))

            if theta == 180.0:
                cv.append(0.0)  # straight line
            elif sign > 0:
                cv.append(1.0 / -re)
            else:
                cv.append(1.0 / re)
        else:
            cv.append(0.0)

    return cv


def CircleFitting(x, y):
    """Circle Fitting with least squared
        input: point x-y positions  
        output  cxe x center position
                cye y center position
                re  radius of circle 
    """

    sumx = sum(x)
    sumy = sum(y)
    sumx2 = sum([ix ** 2 for ix in x])
    sumy2 = sum([iy ** 2 for iy in y])
    sumxy = sum([ix * iy for (ix, iy) in zip(x, y)])

    F = np.array([[sumx2, sumxy, sumx],
                  [sumxy, sumy2, sumy],
                  [sumx, sumy, len(x)]])

    G = np.array([[-sum([ix ** 3 + ix * iy ** 2 for (ix, iy) in zip(x, y)])],
                  [-sum([ix ** 2 * iy + iy ** 3 for (ix, iy) in zip(x, y)])],
                  [-sum([ix ** 2 + iy ** 2 for (ix, iy) in zip(x, y)])]])

    try:
        T = np.linalg.inv(F).dot(G)
    except np.linalg.LinAlgError:
        return 0, 0, float("inf")

    cxe = float(T[0] / -2)
    cye = float(T[1] / -2)

    try:
        re = math.sqrt(cxe ** 2 + cye ** 2 - T[2])
    except np.linalg.LinAlgError:
        return cxe, cye, float("inf")
    return cxe, cye, re

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 2023

@author: dongran

Utility functions for data processing in MoCap-Robotic-Imitation-MEMD
"""

import numpy as np


def min_max(x, axis=None):
    """
    Normalize data to [0, 1] range
    
    Parameters:
    -----------
    x : numpy.ndarray
        Input data to normalize
    axis : int or None
        Axis along which to normalize
        
    Returns:
    --------
    numpy.ndarray
        Normalized data in range [0, 1]
    """
    min_val = x.min(axis=axis, keepdims=True)
    max_val = x.max(axis=axis, keepdims=True)
    result = (x - min_val) / (max_val - min_val)
    return result 
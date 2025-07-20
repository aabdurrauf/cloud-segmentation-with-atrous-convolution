# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 00:19:35 2024

@author: ammar
"""

import numpy as np

smooth = 0.0000001

def compute_intersection(x, y):
    # total = 0
    # for ix in range(len(x)):
    #     total += (x[ix] * y[ix])

    # print(total)
    # return total
    return np.matmul(x, y)

def compute_union(x, y):
    total = 0
    for ix in range(len(x)):
        if x[ix] == 1 or y[ix] == 1:
            total += 1

    return total

def iou_func(y_true, y_pred):
    """
    this function give the same result as the iou_score
    """
    mask_flatten = np.array(y_true).flatten()
    pred_flatten = np.array(y_pred).flatten()
    
    intersection = compute_intersection(mask_flatten, pred_flatten)
    print(f'intersection: {intersection}')
    union = compute_union(mask_flatten, pred_flatten)
    print(f'union: {union}')
    
    return (intersection + smooth)/(union + smooth)


def true_positive(x, y):
    return np.matmul(x, y)


def false_negative(x, y):
    fn = 0
    for ix in range(len(x)):
        if x[ix] == 1 and y[ix] == 0:
            fn += 1
    return fn

def false_positive(x, y):
    fp = 0
    for ix in range(len(x)):
        if x[ix] == 0 and y[ix] == 1:
            fp += 1
    return fp

def true_negative(x, y):
    tn = 0
    for ix in range(len(x)):
        if x[ix] == 0 and y[ix] == 0:
            tn += 1
    return tn

def iou_denominator(x, y):
    tp = np.matmul(x, y)
    fn = false_negative(x, y)
    fp = false_positive(x, y)
    
    return tp + fn + fp

def iou_score(y_true, y_pred):
    """
    this function give the same result as the iou_func
    """
    true_flatten = np.array(y_true).flatten()
    pred_flatten = np.array(y_pred).flatten()
    
    numerator = true_positive(true_flatten, pred_flatten)
    denominator = iou_denominator(true_flatten, pred_flatten)
    
    return (numerator + smooth)/(denominator + smooth)

def precision_score(y_true, y_pred):
    true_flatten = np.array(y_true).flatten()
    pred_flatten = np.array(y_pred).flatten()
    
    numerator = true_positive(true_flatten, pred_flatten)
    denominator = true_positive(true_flatten, pred_flatten) + false_positive(true_flatten, pred_flatten)
    
    return (numerator + smooth)/(denominator + smooth)

def recall_score(y_true, y_pred):
    true_flatten = np.array(y_true).flatten()
    pred_flatten = np.array(y_pred).flatten()
    
    numerator = true_positive(true_flatten, pred_flatten)
    denominator = true_positive(true_flatten, pred_flatten) + false_negative(true_flatten, pred_flatten)
    
    return (numerator + smooth)/(denominator + smooth)

def specificity_score(y_true, y_pred):
    true_flatten = np.array(y_true).flatten()
    pred_flatten = np.array(y_pred).flatten()
    
    numerator = true_negative(true_flatten, pred_flatten)
    denominator = true_negative(true_flatten, pred_flatten) + false_positive(true_flatten, pred_flatten)
    
    return (numerator + smooth)/(denominator + smooth)

def accuracy_score(y_true, y_pred):
    true_flatten = np.array(y_true).flatten()
    pred_flatten = np.array(y_pred).flatten()
    
    numerator = true_positive(true_flatten, pred_flatten) + true_negative(true_flatten, pred_flatten)
    denominator = numerator + false_negative(true_flatten, pred_flatten) + false_positive(true_flatten, pred_flatten)
    
    return (numerator + smooth)/(denominator + smooth)
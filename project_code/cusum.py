import numpy as np
import pandas as pd


#CUSUM Algorithm 
def cusum(signal, target_mean, threshold = 5, k = 0.5):
    ''''
    CUSUM algorithm monitors change detection
    
    Inputs:
        signal: N x 1 time-series data of sensor readings
        target_mean: scalar value of the baseline healthy value that we are monitoring deviations from 
        threhold: scalar sensitivity level (when to triger change alert)
        k: scalra slack value (helps ignore small random fluctiations)

    Output:
        change_inds: 2 x N  np array to store the indices (time steps) where significant positive and negative changes are detected
    '''

    s_pos = s_neg = 0
    change_inds_pos = []
    change_inds_neg = []

    for i, x in enumerate(signal):                               #i tracks the cycle number, x loops each value in the signal
        s_pos = max(0, s_pos + (x - target_mean - k))            #if x is higher than mean it adds to s_pos, if value goes back to normal s_pos resets to zero
        if s_pos > threshold:                                    #if accumulation exceeds threshold --> flag it as a change point & record index
            change_inds_pos.append(i)
            s_pos = 0 

        # s_neg = min(0, s_neg + (x - target_mean + k))
        # if abs(s_neg) > threshold:                                   
        #     change_inds_neg.append(i)
        #     s_neg = 0 

    return change_inds_pos #, change_inds_neg
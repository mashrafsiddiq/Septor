import numpy as np
from scipy import stats
import math

def get_stats(actual, predicted):
    correlation, p_val = stats.pearsonr(actual, predicted)
    mse = np.square(np.subtract(actual,predicted)).mean() 
    rmse = math.sqrt(mse)
    return correlation, rmse, p_val
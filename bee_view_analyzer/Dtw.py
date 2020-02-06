import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy

import tslearn.metrics
import tslearn.preprocessing

import dtwalign
import dtaidistance
import dtaidistance.dtw_visualisation

import bee_view_analyzer.Correlation as Correlation

def _dtw_path(series_1, series_2, window_size):
    if window_size == -1:
        return tslearn.metrics.dtw_path(series_1, series_2)
    else:
        return tslearn.metrics.dtw_path(series_1, series_2, "sakoe_chiba", window_size)

def distance(series_1, series_2, window_size = -1, mode = 0):
    """
    Calculate the Euclidean distance between aligned time series.

    window_size:   constraint maximum warping
    mode:          uses different libraries
    """

    if mode == 0:
        series_1 = np.array(series_1, dtype=np.double)
        series_2 = np.array(series_2, dtype=np.double)
        
        if window_size == -1:
            return dtaidistance.dtw.distance_fast(series_1, series_2)
        else:
            return dtaidistance.dtw.distance_fast(series_1, series_2, window = window_size)
    else:
        if window_size == -1:
            return tslearn.metrics.dtw(series_1, series_2)
        else:
            return tslearn.metrics.dtw(series_1, series_2, "sakoe_chiba", window_size)
    
def show_corr_table(time_series):
    data = []
    min_length = min([len(series) for series in time_series])
    
    for i in range(len(time_series)):
        row = []        
        
        for j in range(len(time_series)):
            p = Correlation._pearson(time_series[i][ : min_length], time_series[j][ : min_length])
            row.append(p)

        data.append(row)
        
    col_names = ["Round {}".format(i) for i in range(1, len(time_series) + 1)]
        
    return pd.DataFrame(data, columns = col_names, index = col_names)

def scale(time_series):
    scaled_series = []
    
    for series in time_series:
        scaled = tslearn.preprocessing.TimeSeriesScalerMinMax(min=0., max=1.).fit_transform(series).flatten()
        scaled_series.append(scaled)
        
    return scaled_series

def resample(time_series):
    resampled_series = []
    
    min_length = min([len(series) for series in time_series])
    
    for series in time_series:
        resampled = tslearn.preprocessing.TimeSeriesResampler(sz = min_length).fit_transform(series).flatten()
        resampled_series.append(resampled)

    return resampled_series

def show_dtw_path(series_1, series_2, window_size = -1, mode = 0):
    if mode == 0:
        path, sim = _dtw_path(series_1, series_2, window_size)

        size = max(len(series_1), len(series_2))
        matrix_path = np.zeros((size, size), dtype = np.float)
        for i, j in path:
            matrix_path[i, j] = 1

        for i in range(size):
            matrix_path[i, i] = 0.3

        plt.figure(figsize=(10, 10))
        plt.imshow(matrix_path, cmap="gray_r")
    elif mode == 1:
        if window_size > -1:
            res = dtwalign.dtw(series_1, series_2, dist="euclidean",step_pattern='symmetric2',
                         window_type='sakoechiba', window_size=window_size)
        else:
            res = dtwalign.dtw(series_1, series_2, dist="euclidean",step_pattern='symmetric2')
            
        res.plot_path()
    elif mode == 2:
        d, paths = dtaidistance.dtw.warping_paths(series_1, series_2, window=20)
        best_path = dtaidistance.dtw.best_path(paths)
        dtaidistance.dtw_visualisation.plot_warpingpaths(series_1, series_2, paths, best_path)
    else:
        if window_size > -1:
            d, paths = dtaidistance.dtw.warping_paths(series_1, series_2, window=window_size)
        else:
            d, paths = dtaidistance.dtw.warping_paths(series_1, series_2)
            
        best_path = dtaidistance.dtw.best_path(paths)
        
        return series_1, series_2, paths, best_path
    
def warp(base_series, warp_series, window_size = -1):
    warped = []    
    path, sim = _dtw_path(base_series, warp_series, window_size)

    current_base_frame = -1

    for base_frame, warp_frame in path:
        if current_base_frame == base_frame:
            continue
        
        current_base_frame = base_frame
        warped.append(warp_series[warp_frame])
    
    return warped

def plot_warped_series(base_series, warp_series, one_plot = True, window_size = -1):
    warped_series = warp(base_series, warp_series, window_size)
    
    plt.figure(figsize=(15, 5))
    plt.plot(base_series)
    
    if not one_plot:
        plt.figure(figsize=(15, 5))
    
    plt.plot(warped_series)
    
def correlate_warped_series(base_series, warp_series, window_size = -1):
    warped_series = warp(base_series, warp_series, window_size)
    
    return Correlation._pearson(base_series, warped_series)

def show_warped_corr_table(time_series, window_size = -1):
    data = []
    
    for i in range(len(time_series)):
        row = []
        
        for j in range(len(time_series)):
            p = correlate_warped_series(time_series[i], time_series[j], window_size)
            row.append(p)

        data.append(row)
        
    row_names = ["Round {}".format(i) for i in range(1, len(time_series) + 1)]
    col_names = ["Warped round {}".format(i) for i in range(1, len(time_series) + 1)]
        
    return pd.DataFrame(data, columns = col_names, index = row_names)

def get_warp_divergence(base_series, warp_series, window_size = -1):
    path, sim = _dtw_path(base_series, warp_series, window_size)
    
    mean_divergence = 0
    max_divergence = 0
    
    for base_frame, warp_frame in path:
        divergence = abs(base_frame - warp_frame)
        
        mean_divergence += divergence
        max_divergence = max(max_divergence, divergence)
        
    mean_divergence = mean_divergence / len(path)
    
    return mean_divergence, max_divergence

def show_warped_divergence_table(time_series, mean = True, window_size = -1):
    data = []
    
    for i in range(len(time_series)):
        row = []
        
        for j in range(len(time_series)):
            mean_divergence, max_divergence = get_warp_divergence(time_series[i], time_series[j], window_size)
            row.append(mean_divergence if mean else max_divergence)

        data.append(row)
        
    row_names = ["Round {}".format(i) for i in range(1, len(time_series) + 1)]
    col_names = ["Warped round {}".format(i) for i in range(1, len(time_series) + 1)]
        
    return pd.DataFrame(data, columns = col_names, index = row_names)

def znorm(time_series):
    normed_series = []
    
    for series in time_series:
        normed = scipy.stats.zscore(series)
        normed_series.append(normed)
        
    return normed_series

def show_warping(series_1, series_2, window_size = -1, fig_width = 20, fig_height = 10, save_path = None):
    plt.rcParams['figure.figsize'] = (fig_width, fig_height)

    if window_size == -1:
        path = dtaidistance.dtw.warping_path(series_1, series_2)
    else:
        path = dtaidistance.dtw.warping_path(series_1, series_2, window = window_size)
    
    if save_path is None:
        return dtaidistance.dtw_visualisation.plot_warping(series_1, series_2, path)
    else:
        dtaidistance.dtw_visualisation.plot_warping(series_1, series_2, path, save_path)

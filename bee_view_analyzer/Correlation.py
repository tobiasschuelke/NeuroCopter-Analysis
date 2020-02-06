import scipy.stats
import math
import torch
from multiprocessing import Pool

import bee_view_analyzer.Utils as Utils

"""
Calculates pearson correlation with pytorch (same as scipy.stats.pearsonr for numpy).
For 1D tensor version see: https://github.com/pytorch/pytorch/issues/1254

Arguments
------
x : 2D torch.Tensor
y : 2D torch.Tensor

Returns
-------
r_val : float
    pearsonr correlation coefficient between x and y
    
Scipy docs ref:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html
    
Scipy code ref:
    https://github.com/scipy/scipy/blob/v0.19.0/scipy/stats/stats.py#L2975-L3033
"""
def _torch_pearsonr(x, y):
    mx = torch.mean(x, 1).unsqueeze(1)
    my = torch.mean(y, 1).unsqueeze(1)
    
    xm = x.sub(mx)
    ym = y.sub(my)
    
    r_num = torch.sum(xm * ym, 1)
    r_den = torch.norm(xm, 2, dim=1) * torch.norm(ym, 2, dim=1)
    
    r_val = r_num / r_den
    
    return r_val

"""
Correlates same y vector for all vectors in x.

Arguments
------
x : 2D torch.Tensor
y : 1D torch.Tensor
"""
def _torch_pearsonr_2D_1D(x, y):
    mx = torch.mean(x, 1).unsqueeze(1)
    my = torch.mean(y)
    
    xm = x.sub(mx)
    ym = y.sub(my)
    
    r_num = torch.sum(xm * ym, 1)
    r_den = torch.norm(xm, 2, dim=1) * torch.norm(ym, 2)
    
    r_val = r_num / r_den
    
    return r_val

"""
Arguments
------
x : 2D torch.Tensor
y : 2D torch.Tensor
"""
def _torch_pearsonr_2D_2D(x, y):
    mx = torch.mean(x, 1).unsqueeze(1)
    my = torch.mean(y, 1).unsqueeze(1)
    
    xm = x.sub(mx)
    ym = y.sub(my)
    
    r_num = torch.sum(xm * ym, 1)
    r_den = torch.norm(xm, 2, dim=1) * torch.norm(ym, 2, dim=1)
    
    r_val = r_num / r_den
    
    return r_val

"""

Arguments
------
x : 3D torch.Tensor
y : 2D torch.Tensor
"""
def _torch_pearsonr_3D_2D(x, y):
    mx = torch.mean(x, 2).unsqueeze(2)
    my = torch.mean(y, 1).unsqueeze(1)
    
    xm = x.sub(mx)
    ym = y.sub(my)
    
    r_num = torch.sum(xm * ym, 2)
    r_den = torch.norm(xm, 2, dim=2) * torch.norm(ym, 2, dim=1)
    
    r_val = r_num / r_den
    
    return r_val

"""

Arguments
------
x : 3D torch.Tensor
y : 3D torch.Tensor
"""
def _torch_pearsonr_3D_3D(x, y):
    mx = torch.mean(x, 2).unsqueeze(2)
    my = torch.mean(y, 2).unsqueeze(2)
    
    xm = x.sub(mx)
    ym = y.sub(my)
    
    r_num = torch.sum(xm * ym, 2)
    r_den = torch.norm(xm, 2, dim=2) * torch.norm(ym, 2, dim=2)
    
    r_val = r_num / r_den
    
    return r_val

def _pearson(a, b):
    p = scipy.stats.pearsonr(a, b)[0]
            
    if math.isnan(p):
        p = 0
        
    return p

class Correlation:
    def __init__(self, index, layer, filter_num, row, col):
        self.index = index
        self.layer = layer
        self.filter_num = filter_num
        self.row = row
        self.col = col
        
    def calculate_pearson(self, activation_series, neuro_series, window_size = -1):
        if window_size > 0:
            self.pearson = []
            
            for i in range(0, len(neuro_series) - window_size, window_size):
                p = _pearson(activation_series[i : i + window_size], neuro_series[i : i + window_size])
                self.pearson.append(p)
                    
        else:
            self.pearson = _pearson(activation_series, neuro_series)
        
    def calculate_spearman(self, activation_series, neuro_series, window_size = -1):
        if window_size > 0:
            self.spearman = []
            
            for i in range(0, len(neuro_series) - window_size, window_size):                
                s = scipy.stats.spearmanr(activation_series[i : i + window_size], neuro_series[i : i + window_size])[0]
                self.spearman.append(s)
                    
        else:
            self.spearman = scipy.stats.spearmanr(activation_series, neuro_series)[0]

    def set_pearson(self, pearson):
        self.pearson = pearson

    def set_spearman(self, spearman):
        self.spearman = spearman

    def get_infos(self):
        if type(self.pearson) is not list:
            return "layer {}, activation {}, row {}, column {}, pearson {}, spearman {}".format(self.layer, self.filter_num, self.row, self.col, self.pearson, self.spearman)
        else:
            return "layer {}, activation {}, row {}, column {}".format(self.layer, self.filter_num, self.row, self.col)
    
    def get_activation_series(self, activations):
        if self.row > -1 and self.col > -1:
            return activations[self.layer][self.filter_num][self.row][self.col]
        else:
            return activations[self.layer][self.filter_num]
    
import time

current_milli_time = lambda: int(round(time.time() * 1000))

def _count_entries(activations):
    count = 0

    for layer in range(len(activations)):
        for filter_num in range(len(activations[layer])):
        
            # check if time series constists of sum of activations
            if activations[layer][filter_num].ndim == 1:
                count += 1
            else:
                # rows * columns
                count += len(activations[layer][filter_num]) * len(activations[layer][filter_num][0])
                
    return count
   
def correlate(activations, neuro_data, start_frame = -1, end_frame = -1, window_size = -1, pearson = True, spearman = True):        
    
    def get_windowed_parts(window_size, series):
        parts = []
        for i in range(0, len(series) - window_size + 1, window_size):
            parts.append(series[i : i + window_size])
            
        return parts

    def append_corr(time_series, layer, filter_num, row = -1, col = -1):
        nonlocal corrs, index, neuro_data
                            
        if end_frame != -1 and end_frame != -1:
            time_series = time_series[start_frame : end_frame]
        
        corr = Correlation(index, layer, filter_num, row, col)

        if torch.cuda.is_available():            
            if pearson:
                time_series_pearson = time_series
                if window_size > 0:
                    time_series_pearson = get_windowed_parts(window_size, time_series)

                all_activation_series[index] = torch.tensor(time_series_pearson, dtype = torch.float64)
            
            if spearman:                
                corr.calculate_spearman(time_series, neuro_data, window_size)
            
        else:
            if pearson:
                corr.calculate_pearson(time_series, neuro_data, window_size)
        
            if spearman:
                corr.calculate_spearman(time_series, neuro_data, window_size)

        corrs.append(corr)
        index += 1

    corrs = []
    index = 0
    
    if start_frame != -1 and end_frame != -1:
        neuro_data = neuro_data[start_frame : end_frame]
    
    neuro_data_torch = neuro_data
    
    if torch.cuda.is_available():
        entry_count = _count_entries(activations)
        time_series_length = len(neuro_data)         
        
        if window_size < 1:
            all_activation_series = torch.empty((entry_count, time_series_length), dtype = torch.float64)
        else:
            windows = time_series_length // window_size
            all_activation_series = torch.empty((entry_count, windows, window_size), dtype = torch.float64)
            
            print("entry_count: ", entry_count, ", windows: ", windows, ", window_size: ", window_size)

            neuro_data_torch = get_windowed_parts(window_size, neuro_data)
        
        neuro_data_torch = torch.tensor(neuro_data_torch, dtype = torch.float64).cuda()
    
    for layer in range(len(activations)):
        for filter_num in range(len(activations[layer])):
            
            # check if time series constists of sum of activations
            if activations[layer][filter_num].ndim == 1:
                time_series = activations[layer][filter_num]
                append_corr(time_series, layer, filter_num)
                continue
            
            # each activation of a filter is a time series
            for row in range(len(activations[layer][filter_num])):
                for col in range(len(activations[layer][filter_num][row])):
                    time_series = activations[layer][filter_num][row][col]                    
                    append_corr(time_series, layer, filter_num, row, col)

    if torch.cuda.is_available():
        part_size = 4096
        
        for r in range((len(corrs) // part_size) + 1):            
            x = all_activation_series[r * part_size : (r + 1) * part_size].cuda()
               
            if window_size < 0:
                r_values = _torch_pearsonr_2D_1D(x, neuro_data_torch).cpu()
            else:
                r_values = _torch_pearsonr_3D_2D(x, neuro_data_torch).cpu()
            
            for i in range(r * part_size, min((r + 1) * part_size, len(corrs)), 1):
                corrs[i].set_pearson(r_values[i % part_size])

    return corrs

def correlate_series(series_1, series_2, pearson = True, start_frame = -1, end_frame = -1):
    series_1 = series_1 if start_frame < 0 and end_frame < 0 else series_1[start_frame : end_frame]
    series_2 = series_2 if start_frame < 0 and end_frame < 0 else series_2[start_frame : end_frame]
    
    if pearson:
        return _pearson(series_1, series_2)
    else:
        return scipy.stats.spearmanr(series_1, series_2)

# Cross correlate an already windowed correlated flight round with all possible sequences of same activation with same length
# windowed_corr_round: windowed correlation of a flight round 
def cross_correlate(activations, neuro_data, windowed_corr_round, window_size):
    if torch.cuda.is_available():
        return _cross_correlate_torch(activations, neuro_data, windowed_corr_round, window_size)
    
    activation_series = windowed_corr_round.get_activation_series(activations)

    all_windowed_corrs = []
    round_corrs = windowed_corr_round.pearson
    
    sequence_corrs = []
    for x in range(0, window_size):
        complete_sequence = []
        for i in range(x, len(activation_series) - window_size, window_size):
            p = _pearson(activation_series[i : i + window_size], neuro_data[i : i + window_size])
            complete_sequence.append(p)
        sequence_corrs.append(complete_sequence)
    
    for x in range(0, len(activation_series) - (window_size * len(round_corrs))):
        start = x % window_size
        sequence = sequence_corrs[start][(x // window_size) : ((x // window_size) + len(round_corrs))]
        all_windowed_corrs.append(sequence)
        
    cross_corrs = []
    for windowed_corr in all_windowed_corrs:        
        corr = Correlation(-1, -1, -1, -1, -1)
        corr.calculate_pearson(round_corrs, windowed_corr)
        cross_corrs.append(corr)
        
    return cross_corrs

def _cross_correlate_torch(activations, neuro_data, windowed_corr_round, window_size):
    activation_series = windowed_corr_round.get_activation_series(activations)    
    entries = ((len(activation_series) - window_size) // window_size) + 1
    
    sequence_activations = torch.empty((window_size, entries, window_size), dtype = torch.float64)
    sequence_neuro = torch.empty((window_size, entries, window_size), dtype = torch.float64)
    empty = [-1 for i in range(window_size)]
    
    for x in range(0, window_size):
        complete_sequence_activations = []
        complete_sequence_neuro = []
        for i in range(x, len(activation_series) - window_size, window_size):
            complete_sequence_activations.append(activation_series[i : i + window_size])
            complete_sequence_neuro.append(neuro_data[i : i + window_size])

        for i in range(len(complete_sequence_activations), entries, 1):
            complete_sequence_activations.append(empty)
            complete_sequence_neuro.append(empty)

        sequence_activations[x] = torch.tensor(complete_sequence_activations, dtype = torch.float64)
        sequence_neuro[x] = torch.tensor(complete_sequence_neuro, dtype = torch.float64)
    
    sequence_corrs = _torch_pearsonr_3D_3D(sequence_activations, sequence_neuro)
    
    round_corrs = windowed_corr_round.pearson
    entries = len(activation_series) - (window_size * len(round_corrs))
    all_windowed_corrs = torch.empty((entries, len(round_corrs)), dtype = torch.float64)
    
    for x in range(entries):
        start = x % window_size
        sequence = sequence_corrs[start][(x // window_size) : ((x // window_size) + len(round_corrs))]
        all_windowed_corrs[x] = sequence
        
    pearson_tensor = _torch_pearsonr_2D_1D(all_windowed_corrs, round_corrs)
    
    cross_corrs = []
    for pearson in pearson_tensor:        
        corr = Correlation(-1, -1, -1, -1, -1)
        corr.set_pearson(pearson)
        cross_corrs.append(corr)
    
    return cross_corrs
    
# sliding window correlation: correlate a part of one time series with every part of the same length
#                             of the same time series
def swc(time_series, start_frame, end_frame):
    window_length = end_frame - start_frame
    entry_count = len(time_series) - window_length
	
    time_series_window = torch.tensor(time_series[start_frame : end_frame], dtype = torch.float64)
    time_series_windows = torch.empty((entry_count, window_length), dtype = torch.float64)

    for i in range(entry_count):
        series = torch.tensor(time_series[i : i + window_length])
        time_series_windows[i] = series
        
    auto_corrs = _torch_pearsonr_2D_1D(time_series_windows, time_series_window)
	
    return auto_corrs

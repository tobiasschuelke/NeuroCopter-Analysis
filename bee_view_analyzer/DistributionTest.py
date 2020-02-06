import sys
import scipy.stats
import torch
from multiprocessing import Pool

import bee_view_analyzer.Utils as Utils
import bee_view_analyzer.Correlation as Correlation

def _mannwhitneyu(a, b, show_errors = False):    
    try:
        mw_statistic, mw_p = scipy.stats.mannwhitneyu(a, b)
    except ValueError as e:
        if show_errors:        
            print(e)
        
        mw_p = 1
        mw_statistic = 0
        
    return mw_statistic, mw_p

class DistributionTest:
    def __init__(self, layer, filter_num, row, col, ks_statistic, ks_p, mw_statistic, mw_p):
        self.layer = layer
        self.filter_num = filter_num
        self.row = row
        self.col = col
        self.ks_statistic = ks_statistic
        self.ks_p = ks_p
        self.mw_statistic = mw_statistic
        self.mw_p = mw_p
        
    def set_ks(self, p, statistic = -1):
        self.ks_p = p
        self.ks_statistic = statistic
        
    def set_mw(self, p, statistic = -1):
        self.mw_p = p
        self.mw_statistic = statistic
        
    def get_infos(self):
        return "layer {}, activation {}, row {}, column {}, ks_statistic {}, ks_p {}, mw_statistic {}, mw_p {}".format(self.layer, self.filter_num, self.row, self.col, self.ks_statistic, self.ks_p, self.mw_statistic, self.mw_p)

def test_distributions(windowed_corrs, activations, neuro_data, window_size, start_frames, exclude_start_frame_num):
    distribution_results = []
    
    # needed if correlations were calculated on gpu:
    if len(windowed_corrs) != 18:
        tmp = [[] for layer_num in range(18)]
        
        for corr in windowed_corrs:
            tmp[corr.layer].append(corr)
    
        windowed_corrs = tmp
    
    for layer_num in range(len(activations)):
        if Utils.is_windows():
            distr = _calc_distributions(-1, layer_num, windowed_corrs, activations, neuro_data, window_size, start_frames, exclude_start_frame_num)
        else:
            with Pool(8) as p:
                options = []
                for pid in range(8):
                    options.append((pid, layer_num, windowed_corrs, activations, neuro_data, window_size, start_frames, exclude_start_frame_num))
                    
                distr = p.starmap(_calc_distributions, options)
                distr = [d for process in distr for d in process]
                
        distribution_results.extend(distr)
        
        sys.stdout.write("\rLayer finished : {}".format(layer_num))
        
    return distribution_results

def _calc_distributions(pid, layer_num, windowed_corrs, activations, neuro_data, window_size, start_frames, exclude_start_frame_num):
    delta = 100
    
    results = []
    progress = 0
      
    corrs = windowed_corrs[layer_num]
    
    if pid > -1:
        size = len(windowed_corrs[layer_num]) // 8
        corrs = windowed_corrs[layer_num][size * pid : size * (pid + 1)]
        
    for corr in corrs:        
        cross_corrs = Correlation.cross_correlate(activations, neuro_data, corr, window_size)
        cross_corrs = [c.pearson for c in cross_corrs]
        
        signal_distribution = []
        null_hyptothesis_distribution = cross_corrs[ : start_frames[0] - delta]
            
        for i in range(len(start_frames)):
            # Do NOT include first round in signal distribution!
            if i != exclude_start_frame_num:
                signal_distribution += cross_corrs[start_frames[i] - delta : start_frames[i] + delta]
                    
            end = start_frames[i + 1] - delta if i + 1 < len(start_frames) else len(cross_corrs)
            null_hyptothesis_distribution += cross_corrs[start_frames[i] + delta : end]

        ks_statistic, ks_p = scipy.stats.ks_2samp(signal_distribution, null_hyptothesis_distribution)
        mw_statistic, mw_p = _mannwhitneyu(signal_distribution, null_hyptothesis_distribution)

        result = DistributionTest(layer_num, corr.filter_num, corr.row, corr.col, ks_statistic, ks_p, mw_statistic, mw_p)
        results.append(result)
    
    return results

import math
import matplotlib.pyplot as plt
import numpy as np
from operator import attrgetter

import bee_view_analyzer.Correlation as Correlation
import bee_view_analyzer.Visualizer as Visualizer
import bee_view_analyzer.DistributionTest as DistributionTest
import bee_view_analyzer.Utils as Utils

def global_hist(activations, neuro_data, norm = False):
    corrs = Correlation.correlate(activations, neuro_data)
    Visualizer.hist3d(corrs, bins = 45, pearson = True, norm = norm)
    
def round_hists(activations, neuro_data, start_frames, end_frames):
    if len(start_frames) != len(end_frames):
        raise ValueError("Arrays with start and end frame numbers have not the same length!")
    
    for round_num in range(len(start_frames)):
        corrs = Correlation.correlate(activations, neuro_data, start_frames[round_num], end_frames[round_num])
        corrs = [c.pearson for c in corrs if not math.isnan(c.pearson) and c.pearson != 0]
        
        plt.hist(corrs, bins=45, label = "round {}".format(round_num))
        
    plt.legend(loc='upper right')
    plt.show()
    
def get_distributions(activations, neuro_data, window_size, start_frame_search_pattern, end_frame_search_pattern,
                      start_frames_correlation, exclude_start_frame_num, save_pattern_path = None):
    
    corr_windowed = Correlation.correlate(activations, neuro_data, start_frame_search_pattern, end_frame_search_pattern,
                                          window_size, spearman = False)

    distribution_tests = DistributionTest.test_distributions(corr_windowed, activations, neuro_data, window_size,
                                                             start_frames_correlation, exclude_start_frame_num)
        
    if save_pattern_path != None:
        Utils.save_object(save_pattern_path, distribution_tests, True)
        
    return distribution_tests

def print_best_distributions(distributions, activations, neuro_data, show_best_pattern = True, highlight = None):
    ks_p_sorted = sorted(distributions, key=attrgetter('ks_p'), reverse=False)

    print("KS test:")
    for i in range(10):
        ks = ks_p_sorted[i]
        print("Layer {}, activation {}, ks_p {}".format(ks.layer, ks.filter_num, ks.ks_p))
      
    if show_best_pattern:
        plot_distr_test_result(ks_p_sorted[0], activations, neuro_data, -1, -1, highlight)
        plot_distr_test_result(ks_p_sorted[0], activations, neuro_data, 200, 1, highlight)
    
    print()
    print("mannwhitneyu test:")
    
    mw_p_sorted = sorted(distributions, key=attrgetter('mw_p'), reverse=False)

    for i in range(10):
        mw = mw_p_sorted[i]
        print("Layer {}, activation {}, mw_p {}".format(mw.layer, mw.filter_num, mw.mw_p))
        
    if show_best_pattern:
        plot_distr_test_result(mw_p_sorted[0], activations, neuro_data, -1, -1, highlight)
        plot_distr_test_result(mw_p_sorted[0], activations, neuro_data, 200, 1, highlight)
        
def plot_distr_test_result(distrTest, activations, neuro_data, smooth_points, smoothness, highlight = None):
    activation_series = activations[distrTest.layer][distrTest.filter_num]
    activation_series = activation_series - np.min(activation_series)
    neuro_data_norm = (neuro_data / np.max(neuro_data)) * np.max(activation_series)
    
    Visualizer.plot([neuro_data_norm, activation_series], labels = ["neuro spikes", "activations"], highlight_regions = highlight,
         smooth_points = smooth_points, smoothness = smoothness, show_unsmoothed_points = False)

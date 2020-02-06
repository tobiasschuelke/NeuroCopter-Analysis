import torch
import math
import numpy as np
from enum import Enum
import sys

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
from scipy import interpolate
from mpl_toolkits.axes_grid1 import make_axes_locatable
from operator import attrgetter
from matplotlib.gridspec import GridSpec

import bee_view_analyzer.Utils as Utils

def show_frame(model_loader, frame_num):
    for i_batch, (xb, yb) in enumerate(model_loader.data_loader):    
        xbd = xb.data
        
        if ((i_batch + 1) * model_loader.batch_size) > frame_num:
            break
            
    img = xbd[frame_num % model_loader.batch_size].data.numpy()
    img = np.transpose(img, (1, 2, 0))
    plt.imshow(img)
    plt.show()

# Show all activations in one frame
def show_all_activations(model_loader, frame_num):
    for i_batch, (xb, yb) in enumerate(model_loader.data_loader):    
        xbd = xb.data
        
        if (i_batch + 1) > frame_num:
            break
        
    with torch.no_grad():
        model_loader.model.eval()
        
        xr = xbd.to(model_loader.device)

        for ii, layer in enumerate(model_loader.model.encoder):
            xr = layer(xr)
            
            fig, axes = plt.subplots(xr.shape[1] // 8, 8, figsize=(12, 3 * xr.shape[1] // 8))

            for i in range(xr.shape[1]):
                r, c = divmod(i, 8)
                
                if xr.shape[1] // 8 > 1:
                    ax = axes[r, c]
                else:
                    ax = axes[c]
                    
                ax.imshow(xr[frame_num % model_loader.batch_size, i].cpu().data.numpy())
                ax.axis('off')

            fig.tight_layout()
            fig.suptitle('Layer {}'.format(ii))
            plt.show
            
            #plt.savefig("activations layer {}.png".format(ii))
      
# Save activations of layers 4, 8 and 12 in all frames
def save_activations_in_all_frames(model_loader, save_path):
    for i_batch, (xb, yb) in enumerate(model_loader.data_loader):    
        xbd = xb.data

        with torch.no_grad():
            model_loader.model.eval()

            xr = xbd.to(model_loader.device)
            
            for ii, layer in enumerate(model_loader.model.encoder):
                xr = layer(xr)
                
                if ii == 4 or ii == 8 or ii == 12:
                    layer_path = Utils.get_path([save_path, "activation_layer_" + str(ii)], '')
                    
                    fig = plt.figure(frameon=False, figsize=(12, 3 * xr.shape[1] // 8))
                    ax = plt.Axes(fig, [0., 0., 1., 1.])
                    fig.add_axes(ax)
                    
                    for img_num in range(batch_size):
                        img = xr[img_num, 2].cpu().data.numpy()
                        
                        ax.clear()
                        ax.set_axis_off()
                        ax.imshow(img)
                        fig.savefig(layer_path + str((i_batch * model_loader.batch_size) + img_num) + ".png", bbox_inches='tight', pad_inches=0)
                        
                    plt.close(fig)
                
                if ii == 12:
                    break
                    
        sys.stdout.write("\rBatch: {}".format(i_batch))

def show_all_correlation(corrs, pearson = True, threshold = -1, save_path = ""):
    current_layer = 0
    current_activation = 0
    current_row_number = 0
    current_row = []
    current_image = []
    layer_images = []
    
    layer_arrays = []

    max_corr = 0
    min_corr = 0
    if pearson:
        max_corr = max(corrs, key=attrgetter('pearson')).pearson
        min_corr = min(corrs, key=attrgetter('pearson')).pearson
    else:
        # remove nans since max of spearman correlation list with nan returns nan
        corrs_without_nan = [c.spearman for c in corrs if not math.isnan(c.spearman)]
        max_corr = max(corrs_without_nan)
        min_corr = min(corrs_without_nan)
    
    for corr in corrs:
        if current_row_number != corr.row:        
            current_image.append(np.array(current_row))

            current_row_number = corr.row
            current_row.clear()

        if current_activation != corr.filter_num:        
            img = np.array(current_image)
            layer_images.append(img)

            current_activation = corr.filter_num
            current_image.clear()

        if current_layer != corr.layer:        
            image_count = len(layer_images)        
            fig, axes = plt.subplots(image_count // 8, 8, figsize=(12, 3 * image_count // 8))

            last_ax = None
            for i in range(image_count):
                r, c = divmod(i, 8)

                if image_count // 8 > 1:
                    ax = axes[r, c]
                else:
                    ax = axes[c]

                im = ax.imshow(layer_images[i], vmin = min_corr, vmax = max_corr)
                ax.axis('off')

                last_ax = ax

            fig.tight_layout()
            fig.suptitle('Layer {}'.format(current_layer))

            if current_layer == 0:
                plt.colorbar(im, ax=axes.ravel().tolist())                
                
            if save_path != "":
                Utils.add_fig_to_image(fig)
            else:
                plt.show

            current_layer = corr.layer
            layer_images.clear()

        corr_value = corr.pearson if pearson else corr.spearman
        
        if threshold > -1:
            corr_value = 1 if abs(corr_value) > threshold else 0
        
        current_row.append(corr_value)
        
    if save_path != "":
        Utils.save_combined_image(save_path)

def plot(time_series, activation_time_series = [], start_frame = -1, end_frame = -1, labels = None,
         highlight_regions = None, smoothness = -1, smooth_points = -1, show_unsmoothed_points = True):
    
    if type(time_series) is list:
        time_series_list = time_series
    else:
        time_series_list = []
        time_series_list.append(time_series)
    
    def plot_smooth(series, smooth_points, smoothness, show_unsmoothed_points, color, label):
        x = range(len(series))
        y = series
        
        if show_unsmoothed_points:
            plt.plot(x, y, marker = 'o', markersize = 1, linestyle='')
        
        x_int = np.linspace(x[0], x[-1], smooth_points if smooth_points > 0 else len(series))
        tck = interpolate.splrep(x, y, k = 3, s = smoothness)
        y_int = interpolate.splev(x_int, tck, der = 0)
        plt.plot(x_int, y_int, linestyle = '-', linewidth = 2, label=label)
    
    plt.figure(figsize=(25,8))
    
    i = 0
    for series in time_series_list:
        time_series = series
        
        if type(series) is torch.Tensor:
            time_series = time_series.numpy()
    
        time_series = time_series if start_frame == -1 and end_frame == -1 else time_series[start_frame : end_frame]
    
        if smoothness > 0:        
            plot_smooth(time_series, smooth_points, smoothness, show_unsmoothed_points, "red", labels[i] if labels != None else None)
        else:
            plt.plot(time_series, label = labels[i] if labels != None else None)

        i += 1
        
    if labels != None:
        plt.legend(loc='upper right')
    
    if highlight_regions != None:
        for i in range(0, len(highlight_regions), 2):
            plt.axvspan(highlight_regions[i], highlight_regions[i + 1], color='green', alpha=0.2)
    
    plt.show()
    
# Same function as plot() above but plots to an axis
def plot_ax(ax, time_series, labels = None, highlight_regions = None, smoothness = -1, x_values = None, show_unsmoothed_points = False, linewidth = 2, color=None):

    if type(time_series) is list:
        time_series_list = time_series
    else:
        time_series_list = []
        time_series_list.append(time_series)

    def plot_smooth(x_scaled_values, series, smoothness, label):
        x = range(len(series))
        y = series
        
        if show_unsmoothed_points:
            x_unsmoothed = x if x_scaled_values is None else x_scaled_values
            plt.plot(x_unsmoothed, y, marker = 'o', markersize = 0.3, linestyle='', label = "SWC")
        
        x_int = np.linspace(x[0], x[-1], len(series))
        tck = interpolate.splrep(x, y, k = 3, s = smoothness)
        y_int = interpolate.splev(x_int, tck, der = 0)
        
        x = x_int if x_scaled_values is None else x_scaled_values
        ax.plot(x, y_int, linestyle = '-', linewidth = linewidth, label=label, color=color)
        
        # instead of splev, rolling mean can be used (produces similar results):
        '''
        y = pd.Series(series).rolling(5).mean()
        x = np.arange(len(series)) if x_scaled_values is None else x_scaled_values
        ax.plot(x, y, linestyle = '-', linewidth = linewidth, label=label)
        '''

    i = 0
    for series_num in range(len(time_series_list)):
        time_series = time_series_list[series_num]
        x = x_values[series_num] if x_values is not None else None

        if type(time_series) is torch.Tensor:
            time_series = time_series.numpy()

        if smoothness > 0:        
            plot_smooth(x, time_series, smoothness, labels[i] if labels != None else None)
        else:
            x_series = x if x is not None else np.linspace(x[0], x[-1], len(series))
            ax.plot(x_series, time_series, label = labels[i] if labels != None else None)

        i += 1

    if labels != None:
        ax.legend(loc='upper right', fontsize=FONT_SIZE_LABEL)

    if highlight_regions != None:
        for i in range(0, len(highlight_regions), 2):
            ax.axvspan(highlight_regions[i], highlight_regions[i + 1], color='green', alpha=0.2)
    
def plot_graphs(graphs, labels = None):
    plt.figure(figsize=(15,8))
    
    for i in range(len(graphs)):
        if labels is None:
            plt.plot(graphs[i])
        else:
            plt.plot(graphs[i], label = labels[i])
        
    plt.legend(loc='upper right')
    plt.show()
    
def show_correlation(corr, show_frame, model_loader, model_children):    
    img_num = show_frame % model_loader.batch_size
    
    for i_batch, (xb, yb) in enumerate(model_loader.data_loader):
        if (i_batch * model_loader.batch_size) + model_loader.batch_size < show_frame:
            xbd = xb.data
            xr = xbd.to(model_loader.device)
            
            fig = plt.figure()
            plt.subplot(1,2,1)
            
            img = xr[img_num].cpu().data.numpy()
            img = np.transpose(img, (1, 2, 0))
            plt.imshow(img)
            
            with torch.no_grad():
                model_children.eval()

                for ii, layer in enumerate(model_children):
                    xr = layer(xr)
                
                    if ii == corr.layer:
                        img = xr[img_num, corr.filter_num].cpu().data.numpy()
                        
                        ax_activation = plt.subplot(1,2,2)                        
                        
                        rect = patches.Rectangle((corr.col - 1, corr.row - 1), 2, 2, linewidth=8, edgecolor='r', facecolor='none')
                        ax_activation.add_patch(rect)
                        
                        suptitle = 'Frame {}'.format(show_frame);
                        if hasattr(corr, 'layer') and hasattr(corr, 'filter_num'):
                            suptitle += ', layer {}, filter {}'.format(corr.layer, corr.filter_num)
                        if hasattr(corr, 'pearson') and not isinstance(corr.pearson, list) and not isinstance(corr.pearson, torch.Tensor):
                            suptitle += ', pearson {}'.format(corr.pearson)
                        if hasattr(corr, 'spearman') and not isinstance(corr.spearman, list):
                            suptitle += ', spearman {}'.format(corr.spearman)
                        
                        plt.suptitle(suptitle)
                            
                        im = plt.imshow(img)
                        
                        plt.colorbar(im, ax=ax_activation)  
                        
                        break
            
            break
            
    return fig

# adapted from https://matplotlib.org/2.1.1/gallery/mplot3d/hist3d.html
def hist3d(corrs, bins = 20, pearson = True, title = None, show = True, norm = True):
    all_corrs = []
    
    if pearson:
        all_corrs = [c.pearson for c in corrs if not math.isnan(c.pearson)]
    else:
        all_corrs = [c.spearman for c in corrs if not math.isnan(c.spearman)]
    
    layers = max([c.layer for c in corrs]) + 1
    
    x_min = min(all_corrs)
    x_max = max(all_corrs)
    y_min = 0
    y_max = layers
    
    x = []
    y = []
    sums = [0 for i in range(layers)]
    for corr in corrs:
        r = corr.pearson if pearson else corr.spearman
        sums[corr.layer] += 1
        
        x.append(r)
        y.append(corr.layer)

    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111, projection='3d')
    hist, xedges, yedges = np.histogram2d(x, y, bins = [bins, layers], range=[[x_min, x_max], [y_min, y_max]])
    
    if norm:
        for layer in range(layers):
            for bin_num in range(len(hist)):
                hist[bin_num, layer] /= sums[layer]
    
    # Construct arrays for the anchor positions of the 16 bars.
    # Note: np.meshgrid gives arrays in (ny, nx) so we use 'F' to flatten xpos,
    # ypos in column-major order. For numpy >= 1.7, we could instead call meshgrid
    # with indexing='ij'.
    xpos, ypos = np.meshgrid(xedges[:-1], yedges[:-1])
    xpos = xpos.flatten('F')
    ypos = ypos.flatten('F')
    zpos = np.zeros_like(xpos)
    
    # Construct arrays with the dimensions for the 16 bars.
    dx = 0.01 * np.ones_like(zpos)
    dy = dx.copy()
    dz = hist.flatten()
    
    color_values = []
    for y_pos in ypos:
        color = plt.cm.jet(y_pos / float(layers))
        color_values.append(color)
    
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=color_values, zsort='average')

    ax.set_xlabel('pearson' if pearson else 'spearman', fontsize = 20)
    ax.set_ylabel('layer', fontsize = 20)
    
    zlabel_normed = "normed " if norm else ""
    ax.set_zlabel(zlabel_normed + 'amount of correlation values', fontsize = 20)
    
    if title is not None:
        ax.set_title(title, fontsize = 40)
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig

def map_frame(model_loader, frame_num, frame_coords, camera_positions, mask_overlay = None, fig_width = 8, fig_height = 8):
    frame_coord = frame_coords[frame_num]
    camera_position = camera_positions[frame_num]
    
    for i_batch, (xb, yb) in enumerate(model_loader.data_loader):    
        xbd = xb.data
        
        if ((i_batch + 1) * model_loader.batch_size) > frame_num:
            break
            
    img = xbd[frame_num % model_loader.batch_size].data.numpy()
    img = np.transpose(img, (1, 2, 0))
    
    mask = np.zeros((img.shape[0], img.shape[1], img.shape[2]))
    
    delta_x = 187 # distance to leftmost x-coordinate from origin on field
    delta_y = 297 # distance to topmost y-coordinate from origin on field
    
    img_map = np.zeros((355, 258, 3))
    
    for coord in frame_coord:
        x = int(delta_x + np.around(coord[0]))
        y = int(delta_y - np.around(coord[2]))
        
        # divide by 2 since bee images are scaled down for autoencoder
        bee_x = min(img.shape[1] - 1, int(coord[3] // 2))
        bee_y = min(img.shape[0] - 1, int(coord[4] // 2))
        
        # mark red pixel
        mask[bee_y][bee_x] = np.array([1.0, 0.0, 0.0])
        
        if y < 0 or x < 0 or y >= img_map.shape[0] or x >= img_map.shape[1]:
            continue
        
        mask[bee_y][bee_x] = np.array([0.0, 0.0, 1.0])
        
        if mask_overlay is not None and np.array_equal(mask_overlay[bee_y][bee_x], np.array([255, 255, 255, 255])):
            continue
        
        # mark green pixel
        mask[bee_y][bee_x] = np.array([0.0, 1.0, 0.0])
        
        img_map[y, x] = img[bee_y][bee_x]
    
    # camera position
    img_map[delta_y - int(camera_position[2]), delta_x + int(camera_position[0])] = np.array((1.0, 0.0, 0.0))

    fig = plt.figure(figsize = (fig_width, fig_height), facecolor='white')
    fig.subplots_adjust(bottom=0, top=1, left = 0.07, right = 0.95, wspace=0.1)
    
    rows = 2
    cols = 3
    gs = GridSpec(rows, cols, figure=fig)

    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(img)
    ax.axis('off')
    
    if mask_overlay is not None:
        ax = fig.add_subplot(gs[1, 0])
        ax.imshow(img)
        ax.imshow(mask, alpha = 0.6)
        ax.axis('off')
    
    ax = fig.add_subplot(gs[ : , 1 : ])
    ax.imshow(img_map)
    ax.axis('off')
    
    return fig
    
    
MAP_FUNCTION = Enum('MapFunctions', 'mean threshold min max std var percentile5 percentile95 median count')

def map_frames(map_function, model_loader, start_frame, end_frame, frame_coords, camera_positions = None, mask = None):
    delta_x = 187 # distance to leftmost x-coordinate from origin on field
    delta_y = 297 # distance to topmost y-coordinate from origin on field
    
    if map_function == MAP_FUNCTION.mean:
        img_map = np.zeros((end_frame - start_frame, 355, 258, 3))
    elif map_function == MAP_FUNCTION.threshold:
        img_map = np.zeros((355, 258, 3))
    else:
        raise NotImplementedError('Map function not implemented!')
    
    for i_batch, (xb, yb) in enumerate(model_loader.data_loader):    
        xbd = xb.data

        if ((i_batch + 1) * model_loader.batch_size) < start_frame:
            continue
            
        if (i_batch * model_loader.batch_size) > end_frame:
            break

        start = 0
        if i_batch * model_loader.batch_size < start_frame:
            start = start_frame % model_loader.batch_size
        
        end = model_loader.batch_size
        if (i_batch + 1) * model_loader.batch_size - 1 > end_frame:
            end = end_frame % model_loader.batch_size
            
        for frame_num in range(start, end, 1):        
            img = xbd[frame_num % model_loader.batch_size].data.numpy()
            img = np.transpose(img, (1, 2, 0))
            
            frame_index = i_batch * model_loader.batch_size + frame_num
            
            for coord in frame_coords[frame_index]:
                x = int(delta_x + np.around(coord[0]))
                y = int(delta_y - np.around(coord[2]))
                
                # divide by 2 since bee images are scaled down for autoencoder
                bee_x = min(img.shape[1] - 1, int(coord[3] // 2))
                bee_y = min(img.shape[0] - 1, int(coord[4] // 2))
                                                    
                if mask is not None and np.array_equal(mask[bee_y][bee_x], np.array([255, 255, 255, 255])):
                    continue
                    
                if map_function == MAP_FUNCTION.mean:
                    if y < 0 or x < 0 or y >= img_map.shape[1] or x >= img_map.shape[2]:
                        continue
                    
                    img_map[frame_index - start_frame, y, x] = img[bee_y][bee_x]
                else:
                    if y < 0 or x < 0 or y >= img_map.shape[0] or x >= img_map.shape[1]:
                        continue
                    
                    if np.all(np.greater(img[bee_y][bee_x], np.array([0.3, 0.3, 0.3]))):
                        img_map[y, x] = img[bee_y][bee_x]
    
    if map_function == MAP_FUNCTION.mean:
        # mean of all colors per pixel (excludes (0, 0, 0))
        img_map = np.true_divide(img_map.sum(0),(img_map!=0).sum(0))
    
    # camera positions
    if camera_positions is not None:
        for frame_num in range(start_frame, end_frame):
            y = delta_y - int(camera_positions[frame_num][2])
            x = delta_x + int(camera_positions[frame_num][0])
            
            if y < 0 or x < 0 or y >= img_map.shape[0] or x >= img_map.shape[1]:
                        continue
            
            img_map[y, x] = np.array((1.0, 0.0, 0.0))
    
    fig = figure(figsize = (4, 4))
    fig.subplots_adjust(bottom=0, top=1, left = 0, right = 1)
    
    ax = fig.add_subplot(111)
    ax.imshow(img_map)
    ax.axis('off')
    
    return fig, img_map
    
def map_activation_map(map_function, layer_num, activation_map_num, model_loader, start_frame, end_frame, frame_coords, fig_size, show_colorbar = True, colorbar_label_size = 2, camera_positions = None, mask = None):
    delta_x = 187 # distance to leftmost x-coordinate from origin on field
    delta_y = 297 # distance to topmost y-coordinate from origin on field
    
    if map_function == MAP_FUNCTION.mean:
        img_map = np.zeros((end_frame - start_frame, 355, 258))
    else:
        img_map = np.zeros((355, 258))
    
    for i_batch, (xb, yb) in enumerate(model_loader.data_loader):    
        xbd = xb.data

        if ((i_batch + 1) * model_loader.batch_size) < start_frame:
            continue
            
        if (i_batch * model_loader.batch_size) > end_frame:
            break

        start = 0
        if i_batch * model_loader.batch_size < start_frame:
            start = start_frame % model_loader.batch_size
        
        end = model_loader.batch_size
        if (i_batch + 1) * model_loader.batch_size - 1 > end_frame:
            end = end_frame % model_loader.batch_size
        
        with torch.no_grad():
            model_loader.model.eval()
        
            xr = xbd.to(model_loader.device)

            for ii, layer in enumerate(model_loader.model.encoder):
                xr = layer(xr)
                
                if ii < layer_num:
                    continue
                    
                if ii > layer_num:
                    break
        
                for frame_num in range(start, end):        
                    img = xr[frame_num % model_loader.batch_size, activation_map_num].cpu().data.numpy()

                    frame_index = i_batch * model_loader.batch_size + frame_num

                    for coord in frame_coords[frame_index]:
                        x = int(delta_x + np.around(coord[0]))
                        y = int(delta_y - np.around(coord[2]))

                        # divide by 2 since bee images are scaled down for autoencoder
                        bee_x = min(img.shape[1] - 1, int(coord[3] // 2))
                        bee_y = min(img.shape[0] - 1, int(coord[4] // 2))

                        if mask is not None and np.array_equal(mask[bee_y][bee_x], np.array([255, 255, 255, 255])):
                            continue

                        if map_function == MAP_FUNCTION.mean:
                            if y < 0 or x < 0 or y >= img_map.shape[1] or x >= img_map.shape[2]:
                                continue

                            img_map[frame_index - start_frame, y, x] = img[bee_y][bee_x]
                        else:
                            if y < 0 or x < 0 or y >= img_map.shape[0] or x >= img_map.shape[1]:
                                continue

                            if img[bee_y][bee_x] > 0.3:
                                img_map[y, x] = img[bee_y][bee_x]
    
    if map_function == MAP_FUNCTION.mean:
        # mean of all colors per pixel (excludes (0, 0, 0))
        img_map = np.true_divide(img_map.sum(0),(img_map!=0).sum(0))
    
    overlay = np.full((355, 258), np.nan)
    
    # camera positions
    if camera_positions is not None:
        for frame_num in range(start_frame, end_frame):
            y = delta_y - int(camera_positions[frame_num][2])
            x = delta_x + int(camera_positions[frame_num][0])
            
            if y < 0 or x < 0 or y >= img_map.shape[0] or x >= img_map.shape[1]:
                        continue
            
            overlay[y, x] = np.array((1.0))
            
    width = fig_size * 0.5 * 0.9
    fig = plt.figure(figsize = (width, width / 0.8))
    fig.subplots_adjust(bottom=0, top=1, left = 0, right = 1)
    
    current_cmap = matplotlib.cm.get_cmap()
    current_cmap.set_bad(alpha=0)
    
    ax = fig.add_subplot(111)
    im = ax.imshow(img_map, vmin=-0.6, vmax=0.4)
    ax.axis('off')
    
    ax.imshow(overlay, cmap = plt.cm.hsv)
    
    fig_colorbar = None
    
    if show_colorbar:
        fig_colorbar = plt.figure(figsize = (fig_size, 0.6))
        fig_colorbar.subplots_adjust(bottom=0.7, top=0.95, left = 0.2, right = 0.8)
        
        cbar_ax = fig_colorbar.add_subplot(111)
        fig_colorbar.colorbar(im, cax=cbar_ax, orientation='horizontal')
        
        cbar_ax.tick_params(labelsize = colorbar_label_size)
        
    return fig, fig_colorbar
    	
    
def map_swc(field_map, start_frame, start_frame_corr_part, end_frame_corr_part, correlations, camera_positions, show_colorbar = True):
    delta_x = 187 # distance to leftmost x-coordinate from origin on field
    delta_y = 297 # distance to topmost y-coordinate from origin on field
    
    fig = figure(figsize = (30,30))
    plt.imshow(field_map / 4)
    
    corr_overlay = np.full((field_map.shape[0], field_map.shape[1]), np.nan)
    highlight_overlay = np.full((field_map.shape[0], field_map.shape[1]), np.nan)
    
    highlight_overlay[0,0] = 0
    
    for frame_num in range(start_frame, start_frame + len(correlations)):
        y = delta_y - int(camera_positions[frame_num][2])
        x = delta_x + int(camera_positions[frame_num][0])
        
        corr_overlay[y, x] = correlations[frame_num - start_frame]
        
        if frame_num == start_frame_corr_part or frame_num == end_frame_corr_part:
            highlight_overlay[y - 2 : y + 2, x - 2 : x + 2] = 1
        
    current_cmap = matplotlib.cm.get_cmap()
    current_cmap.set_bad(alpha=0)    
    
    plt.imshow(highlight_overlay, cmap=plt.cm.gray)
    
    im = plt.imshow(corr_overlay, vmin = -0.3, vmax = 0.4)
    
    if show_colorbar:
        plt.colorbar(im)
    
    return fig

def get_spike_rate_map_matrix(spike_rate, start_frames, end_frames, frame_coords, leave_out_frames = -1):
    delta_x = 187 # distance to leftmost x-coordinate from origin on field
    delta_y = 297 # distance to topmost y-coordinate from origin on field
    
    frame_count = 0
    for i in range(len(start_frames)):
        frame_count += end_frames[i] - start_frames[i]
        
    map_matrix = np.full((frame_count, 355, 258), np.nan)
    
    for i in range(len(start_frames)):
        start_frame = start_frames[i]
        end_frame = end_frames[i]
        
        for frame_num in range(start_frame, end_frame):
            if leave_out_frames > 0 and frame_num % leave_out_frames != 0:
                continue
            
            for coord in frame_coords[frame_num]:
                x = int(delta_x + np.around(coord[0]))
                y = int(delta_y - np.around(coord[2]))

                if y < 0 or x < 0 or y >= map_matrix.shape[1] or x >= map_matrix.shape[2]:
                    continue

                map_matrix[frame_num - start_frame, y, x] = spike_rate[frame_num]
                
    return map_matrix

def get_spike_rate_map(map_matrix, map_function, start_frames, end_frames, camera_positions = None, show_colorbar = True, title = None, ax = None, use_log = False, dummy = False, text_size = 48):
    delta_x = 187 # distance to leftmost x-coordinate from origin on field
    delta_y = 297 # distance to topmost y-coordinate from origin on field
    
    if dummy:
        img_map = map_matrix
    elif map_function == MAP_FUNCTION.threshold:
        img_map = np.full((355, 258), np.nan)
        
        for i in range(len(map_matrix)):
            for y in range(len(map_matrix[i])):
                for x in range(len(map_matrix[i, y])):
                    if not np.isnan(map_matrix[i, y, x]):
                        img_map[y, x] = map_matrix[i, y, x]
    
    elif map_function == MAP_FUNCTION.mean:
        img_map = np.nanmean(map_matrix, axis = 0)        
    elif map_function == MAP_FUNCTION.min:
        img_map = np.nanmin(map_matrix, axis = 0)
    elif map_function == MAP_FUNCTION.max:
        img_map = np.nanmax(map_matrix, axis = 0)
    elif map_function == MAP_FUNCTION.std:
        img_map = np.nanstd(map_matrix, axis = 0)
    elif map_function == MAP_FUNCTION.var:
        img_map = np.nanvar(map_matrix, axis = 0)
    elif map_function == MAP_FUNCTION.percentile5:
        img_map = np.nanpercentile(map_matrix, 5, axis = 0)
    elif map_function == MAP_FUNCTION.percentile95:
        img_map = np.nanpercentile(map_matrix, 95, axis = 0)
    elif map_function == MAP_FUNCTION.median:
        img_map = np.nanpercentile(map_matrix, 50, axis = 0)
    elif map_function == MAP_FUNCTION.count:
        # count non-nan elements
        img_map = np.count_nonzero(~np.isnan(map_matrix), axis = 0)
    else:
        raise NotImplementedError('Map function not implemented!')
    
    if use_log:
        img_map = np.log1p(img_map)
    
    u = np.nanmean(img_map)
    std = np.nanstd(img_map)
    
    fig = figure(figsize = (15,15)) if ax is None else None
    
    im = None
    if ax:
        im = ax.imshow(img_map, vmin = max(0, u - 2 * std), vmax = u + 2 * std)
    else:
        im = plt.imshow(img_map, vmin = max(0, u - 2 * std), vmax = u + 2 * std)
    
    # camera positions
    if camera_positions is not None:
        cam_overlay = np.full((img_map.shape[0], img_map.shape[1], 4), [0,0,0,0], dtype = np.uint8)
        
        current_cmap = matplotlib.cm.get_cmap()
        current_cmap.set_bad(alpha=0)  
        
        for i in range(len(start_frames)):
            start_frame = start_frames[i]
            end_frame = end_frames[i]
            
            for frame_num in range(start_frame, end_frame):
                y = delta_y - int(camera_positions[frame_num][2])
                x = delta_x + int(camera_positions[frame_num][0])

                if y < 0 or x < 0 or y >= img_map.shape[0] or x >= img_map.shape[1]:
                            continue

                cam_overlay[y, x] = np.array([255, 0, 0, 255])

            if ax:
                ax.imshow(cam_overlay)
            else:
                plt.imshow(cam_overlay)
    
    if title:
        if ax:
            ax.set_title(title, fontsize = text_size, y = 1.03)
        else:
            plt.suptitle(title, fontsize = text_size, y = 1.03)
    
    if ax:
        ax.axis('off')
    
    if show_colorbar:
        if ax:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.1)
            cbar = plt.colorbar(im, cax=cax, orientation='vertical')
        else:        
            cbar = colorbar(im)
            
        cbar.ax.tick_params(labelsize = text_size)
        
    return fig, img_map, im
    
def colorbar(mappable):
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    return fig.colorbar(mappable, cax=cax)

def map_spike_rate(map_function, spike_rate, start_frames, end_frames, frame_coords, camera_positions = None, leave_out_frames = -1, show_colorbar = True, title = None):
    map_matrix = get_spike_rate_map_matrix(spike_rate, start_frames, end_frames, frame_coords, leave_out_frames)
    return get_spike_rate_map(map_matrix, map_function, start_frames, end_frames, camera_positions, show_colorbar, title)

def get_mean_map(maps, show_colorbar = True, title = "Mean"):
    mean_map = np.nanmean(maps, 0)

    u = np.nanmean(mean_map)
    std = np.nanstd(mean_map)

    fig = figure(figsize = (15,15))
    im = plt.imshow(mean_map, vmin = u - 2 * std, vmax = u + 2 * std)
    
    if title:
        plt.suptitle(title, fontsize=48)
    
    if show_colorbar:
        plt.colorbar(im)
    
    return fig
    
def show_camera_movement(field_map, start_frame, end_frame, camera_positions):
    delta_x = 187 # distance to leftmost x-coordinate from origin on field
    delta_y = 297 # distance to topmost y-coordinate from origin on field
    
    fig = figure(figsize = (30,30))
    #plt.imshow(field_map / 4)
    
    cam_overlay = np.full((field_map.shape[0], field_map.shape[1], 4), [0,0,0,0], np.uint8)
    
    for frame_num in range(start_frame, end_frame):
        y = delta_y - int(camera_positions[frame_num][2])
        x = delta_x + int(camera_positions[frame_num][0])
        
        if (frame_num % 100) == 0 or (frame_num - 1) % 100 == 0 or (frame_num + 1) % 100 == 0:
            cam_overlay[y, x] = [255, 0, 0, 255]
        elif (frame_num % 50) == 0 or (frame_num - 1) % 50 == 0 or (frame_num + 1) % 50 == 0:
            cam_overlay[y, x] = [0, 0, 255, 255]
        else:
            cam_overlay[y, x] = [0, 255, 0, 255]
    
    plt.imshow(cam_overlay)
    plt.show()

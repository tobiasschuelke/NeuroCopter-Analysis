import torch
import csv
import numpy as np
from enum import Enum
import geopy.distance

from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

ACTIVATION_FUNCTION = Enum('ActivationFunction', 'sum_abs max_abs mean_abs sum max mean')
_ABS_FUNCTIONS = [ACTIVATION_FUNCTION.sum_abs, ACTIVATION_FUNCTION.max_abs, ACTIVATION_FUNCTION.mean_abs]

def load_training_data(img_height, img_width, path_training, valid_size, batch_size):
    t = transforms.Compose(
        [transforms.Resize((img_height, img_width), interpolation=5), 
         transforms.ToTensor()]
    )

    data = ImageFolder(path_training, transform=t)
	
    num_train = int(len(data))
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    
    train_loader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=4, sampler=train_sampler, drop_last=True)
    valid_loader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=4, sampler=valid_sampler, drop_last=True)
    
    return train_loader, valid_loader

def load_neuro_data(model_loader, csv_file_path, spike_type_row, start_frame = 0, start_frames = None, end_frames = None):
    neuro_data = []
    i = 0
    
    if model_loader is not None:
        image_count = model_loader.get_image_count()
    
    with open(csv_file_path) as csvfile:
        neuro_reader = csv.reader(csvfile)
    
        row_num = -1
        for row in neuro_reader:
            row_num += 1
            
            if row_num < start_frame:
                continue
            
            if model_loader is not None and row_num - start_frame >= image_count:
                break
            
            if start_frames is not None and end_frames is not None:
                if row_num - start_frame  < start_frames[i]:
                    continue
                    
                if row_num - start_frame == end_frames[i]:
                    i += 1
                    if i == len(start_frames):
                        break
                    
                    continue
            
            neuro_data.append(int(row[spike_type_row]))
            
    return np.array(neuro_data)

def load_camera_positions(csv_file_path, voltage_column, start_frame = 0):
    radar_coords = (52.459798, 13.298177) # from google maps (lat, long)
    camera_positions = []
    
    def convert_coords(coords):
        """ convert the gps coordinates to the coordinates used by beeview """
        
        dist_y = geopy.distance.geodesic(radar_coords, (coords[0], radar_coords[1])).m
        dist_x = geopy.distance.geodesic(radar_coords, (radar_coords[0], coords[1])).m
    
        if radar_coords[1] > coords[1]:
            dist_x *= (-1)
        
        if radar_coords[0] > coords[0]:
            dist_y *= (-1)
    
        return (dist_x, dist_y)
    
    with open(csv_file_path) as csvfile:
        neuroReader = csv.reader(csvfile)
        row_num = -1       
        
        for row in neuroReader:
            row_num += 1
        
            if row_num < start_frame:
                continue
            
            # IMU_ATTI columns:
            lat = row[voltage_column + 16]
            long = row[voltage_column + 15]
            
            relative_height = float(row[voltage_column + 54]) if row[voltage_column + 54] else 0
            
            if lat != "" and long != "":
                coords = [float(lat), float(long)]
            else:
                coords = radar_coords
                
            x, y, = convert_coords(coords)
            
            position = [x, relative_height, y]
            camera_positions.append(position)
            
    return camera_positions
            
def load_motor_speeds(csv_file_path, voltage_column, start_frame):
    flight_speeds = [[], [], [], []] # motor 1 - 4
    flight_accel = [[], [], []]      # x, y, z
    
    with open(csv_file_path) as csvfile:
        reader = csv.reader(csvfile)        
    
        row_num = -1
        for row in reader:
            row_num += 1
            
            if row_num < start_frame:
                continue
                
            flight_speeds[0].append(int(row[voltage_column + 100]))
            flight_speeds[1].append(int(row[voltage_column + 101]))
            flight_speeds[2].append(int(row[voltage_column + 102]))
            flight_speeds[3].append(int(row[voltage_column + 103]))
            
            flight_accel[0].append(float(row[voltage_column + 21]))
            flight_accel[1].append(float(row[voltage_column + 22]))
            flight_accel[2].append(float(row[voltage_column + 23]))
    
    return np.array(flight_speeds), np.array(flight_accel)
            
# pytorch batch: [img-num][activation][row][column]
# transform to:  [layer][activation][row][column][time series/frame-num]
def load_activations(model_loader, start_frame = -1, end_frame = -1):
    activations = [None for i in range(len(model_loader.model.encoder))]
    
    for layer_num in range(len(model_loader.model.encoder)):
        _set_activations(model_loader, activations, None, start_frame, end_frame, layer_num)
        
    return activations
    
# pytorch batch: [img-num][activation][row][column]
# transform to:  [activation][row][column][time series/frame-num]
def load_activations_of_layer(layer_num, model_loader, start_frame = -1, end_frame = -1):
    activations = [None for i in range(len(model_loader.model.encoder))]
    
    _set_activations(model_loader, activations, None, start_frame, end_frame, layer_num)
        
    return activations[layer_num]

# pytorch batch: [img-num][activation][row][column]
# transform to:  [layer][activation][time series/frame-num]
def load_filter_activations(model_loader, activation_function, start_frame = -1, end_frame = -1):
    activations = [None for i in range(len(model_loader.model.encoder))]
    
    _set_activations(model_loader, activations, activation_function, start_frame, end_frame, layer_num = -1)
    
    return activations
    
def _set_activations(model_loader, activations, activation_function, start_frame, end_frame, layer_num):
    start = start_frame
    end = end_frame
    
    if type(start_frame) == list:
        i = 0
        start = start_frame[i]
        end = end_frame[i]
    
    for i_batch, (xb, yb) in enumerate(model_loader.data_loader):
        if type(start_frame) == list and i == len(start_frame):
            break
        
        xbd = xb.data
        
        first_frame_of_batch = i_batch * model_loader.batch_size
        last_frame_of_batch = (i_batch * model_loader.batch_size) + model_loader.batch_size - 1
        
        if start > -1:
            if last_frame_of_batch < start:
                continue
            
            if first_frame_of_batch < start:
                xbd = xbd[start % model_loader.batch_size : ]

        if end > -1:
            if first_frame_of_batch > end:
                if type(start_frame) == list:
                    i += 1
                    if i == len(start_frame):
                        break
                    
                    start = start_frame[i]
                    end = end_frame[i]
                    
                    if (i_batch * model_loader.batch_size) < start:
                        xbd = xbd[start % model_loader.batch_size : ]
                    else:
                        continue
                else:
                    break
                
            if last_frame_of_batch > end:
                xbd_prime = xbd[ : end % model_loader.batch_size]
                
                if type(start_frame) == list:
                    i += 1
                    
                    if i < len(start_frame):
                        start = start_frame[i]
                        end = end_frame[i]
                    
                        if start < last_frame_of_batch:
                            xbd = torch.cat((xbd_prime, xbd[start % model_loader.batch_size : ]))
                    
                xbd = xbd_prime
        
        with torch.no_grad():
            model_loader.model.eval()

            xr = xbd.to(model_loader.device)

            for ii, layer in enumerate(model_loader.model.encoder):
                xr = layer(xr)
                
                if layer_num > -1:
                    if ii > layer_num:
                        break
                    elif ii != layer_num:
                        continue
                
                xrPrime = xr.clone()
                
                if activation_function is not None:
                    if activation_function in _ABS_FUNCTIONS:
                        xrPrime = torch.abs(xrPrime)
                                           
                    if activation_function == ACTIVATION_FUNCTION.sum_abs or activation_function == ACTIVATION_FUNCTION.sum:
                        xrPrime = torch.sum(xrPrime, (2, 3))
                        
                    elif activation_function == ACTIVATION_FUNCTION.max_abs or activation_function == ACTIVATION_FUNCTION.max:
                        xrPrime = torch.max(xrPrime, 3)[0]
                        xrPrime = torch.max(xrPrime, 2)[0]
                        
                    elif activation_function == ACTIVATION_FUNCTION.mean_abs or activation_function == ACTIVATION_FUNCTION.mean:
                        xrPrime = torch.mean(xrPrime, 3)
                        xrPrime = torch.mean(xrPrime, 2)
                     
                dim1 = 4 if activation_function == None else 2
                dim2 = 3 if activation_function == None else 1
                
                xrPrime = torch.unsqueeze(xrPrime, dim=dim1)
                xrPrime = torch.cat(([xrPrime[i] for i in range(len(xrPrime))]), dim=dim2)
                        
                if activations[ii] is None:
                    activations[ii] = xrPrime.cpu().data.numpy()
                else:
                    activations[ii] = np.concatenate((activations[ii], xrPrime.cpu().data.numpy()), axis=dim2)
                    


import os
import dill
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# avoids error while loading dill object on windows
dill._dill._reverse_typemap['ClassType'] = type

# raises max pixels of a texture that can be converted to ppm
Image.MAX_IMAGE_PIXELS = 1073741825

output_image_figures = []

def is_windows():
    return os.name == 'nt'

def get_path(folders, file = ""):
    path = ""
    runs_on_windows = is_windows()
    
    for i in range(len(folders)):
        if runs_on_windows:
            path += folders[i] + "\\"
        else:
            path += folders[i] + "/"
            
    path += file
    
    return path

def add_fig_to_image(fig, append_horizontal = False, append_vertical = False):    
    fig.canvas.draw()

    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    if append_horizontal:
        f = output_image_figures.pop()
        f = np.hstack((f, data))        
        output_image_figures.append(f)
    elif append_vertical:
        f = output_image_figures.pop()
        f = np.hstack((f, data))        
        output_image_figures.append(f)
    else:
        output_image_figures.append(data)
    
    plt.close(fig)

def save_combined_image(save_path, show_image = False, combine_vertical = True):
    global output_image_figures
    
    img_merge = None
    if combine_vertical:
        img_merge = np.vstack((output_image_figures))
    else:
        img_merge = np.hstack((output_image_figures))
    
    img_merge = Image.fromarray(img_merge)
    img_merge.save(save_path)
    
    if show_image:
        plt.imshow(img_merge)
        plt.show()
    
    _clear_combined_images()
    
def _clear_combined_images():
    global output_image_figures
    output_image_figures = []

def combine_images_horizontally(imgs, save_path):
    images = [Image.open(i) for i in imgs]
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset,0))
        x_offset += im.size[0]

    new_im.save(save_path)
    
def save_object(file_path, obj):
    with open(file_path, 'wb') as fp:
        dill.dump(obj, fp)
        
        
def load_object(file_path):
    with open (file_path, 'rb') as fp:
        return dill.load(fp)

def convert_texture(file_path):
    im = Image.open(file_path)
    im.save(file_path[:-3] + "ppm")

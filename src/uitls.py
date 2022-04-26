import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from mpl_toolkits.axes_grid1 import ImageGrid

def resize_maps(maps, target_size):
    """Resize feature maps to a given size.

    Args:
        maps (np.array): 4D array of feature maps (batch_size, height, width, channels)
        target_size (tuple): target size (height, width)

    Returns:
        np.array: resized feature maps
    """
    resized_maps = []
    for i in range(maps.shape[0]):
        resized_map = np.empty((target_size[0], target_size[1], maps.shape[3]))
        for j in range(maps.shape[3]):
            resized_map[:,:,j] = resize(maps[i,:,:,j], target_size)
        resized_maps.append(resized_map)
        
    return np.array(resized_maps)


def plot_channels(im, labels=None, figsize=None):
    fig = plt.figure(figsize=figsize)
    
    axes = ImageGrid(
        fig, 111,
        nrows_ncols=(im.shape[0], 4),
        axes_pad=0.05,
        share_all=True
    )
    
    for i in range(im.shape[0]):
        axes[4*i].imshow(im[i,:,:,:])
        axes[4*i].set_xticks([])
        axes[4*i].set_yticks([])
    
        for j in range(3):
            axes[4*i+j+1].imshow(im[i,:,:,j], cmap="gray")
            axes[4*i+j+1].set_xticks([])
            axes[4*i+j+1].set_yticks([])
    
    # column names
    axes[0].set_title("color")
    for i in range(3):
        axes[i+1].set_title(f"channel{i+1}")
        
    if labels is not None:
        for i in range(im.shape[0]):
            axes[4*i].set_ylabel(labels[i])


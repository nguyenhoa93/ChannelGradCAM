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
            
def vis_feature_map(feature_maps, channel_cams=None, size_factor=1.5, alpha=0.8):
    if feature_maps.shape[3] < 8:
        ncols = feature_maps.shape[3]
        nrows = 1
        figsize = (size_factor*ncols, size_factor)
    else:
        ncols = min(8,int(np.floor(np.sqrt(feature_maps.shape[3]))))
        nrows = ncols
        figsize = (size_factor*ncols,ncols*size_factor)
    
    fig = plt.figure(figsize=figsize)
    axes = ImageGrid(
        fig, 111,
        nrows_ncols=(nrows, ncols),
        axes_pad=0.05,
        share_all=True
    )
    
    for i in range(feature_maps.shape[3]):
        axes[i].imshow(feature_maps[0,:,:,i], cmap="gray")
        axes[i].axis("off")
        if channel_cams is not None:
            axes[i].imshow(channel_cams[0,:,:,i], alpha=alpha, vmin=0, vmax=np.max(channel_cams))


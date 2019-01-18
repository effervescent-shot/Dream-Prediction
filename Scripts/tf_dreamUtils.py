import math as m
import numpy as np
import scipy.io
from sklearn.decomposition import PCA

from functools import reduce
import math as m
from scipy.interpolate import griddata
from sklearn.preprocessing import scale
np.random.seed(1235)


def lab(y, sample_size):
    return [y]*sample_size

def tt_split(data, labels, ratio=0.1, myseed=2019):
    """Split the dataset based on the split ratio. 
    :param data: data to be slit
    :param labels: data labels
    :param ratio: training data ratio 
    :param myseed: seed for randomm number generator
    """
    # set seed
    np.random.seed(623946494)
    # generate random indices
    num_row = len(labels)
    indices = np.random.permutation(num_row)
    index_split = int(np.floor(ratio * num_row))
    index_tr = indices[: index_split]

    # seperate 20% of test data as validation data
    valid_split = index_split + int((num_row-index_split)*0.20)
    index_va = indices[index_split:valid_split]
    index_te = indices[valid_split:]

    # create split
    x_tr = data[index_tr]
    x_te = data[index_te]
    x_va = data[index_va]
    y_tr = labels[index_tr]
    y_te = labels[index_te]
    y_va = labels[index_va]
    return x_tr,y_tr, x_va, y_va, x_te, y_te


def cart2sph(x, y, z):
    """
    source : https://github.com/pbashivan/EEGLearn/utils.py
    
    Transform Cartesian coordinates to spherical
    :param x: X coordinate
    :param y: Y coordinate
    :param z: Z coordinate
    :return: radius, elevation, azimuth
    """
    x2_y2 = x**2 + y**2
    r = m.sqrt(x2_y2 + z**2)                    # r
    elev = m.atan2(z, m.sqrt(x2_y2))            # Elevation
    az = m.atan2(y, x)                          # Azimuth
    return r, elev, az


def pol2cart(theta, rho):
    """
    source : https://github.com/pbashivan/EEGLearn/utils.py
    
    Transform polar coordinates to Cartesian
    :param theta: angle value
    :param rho: radius value
    :return: X, Y
    """
    return rho * m.cos(theta), rho * m.sin(theta)




def azim_proj(pos):
    """
    source : https://github.com/pbashivan/EEGLearn/eeg_cnn_lib.py
    
    Computes the Azimuthal Equidistant Projection of input point in 3D Cartesian Coordinates.
    Imagine a plane being placed against (tangent to) a globe. If
    a light source inside the globe projects the graticule onto
    the plane the result would be a planar, or azimuthal, map
    projection.

    :param pos: position in 3D Cartesian coordinates
    :return: projected coordinates using Azimuthal Equidistant Projection
    """
    [r, elev, az] = cart2sph(pos[0], pos[1], pos[2])
    return pol2cart(az, m.pi / 2 - elev)

def centerize_reference(x):
    electrode_means = x.mean(axis=1, keepdims=True)
    x = x-electrode_means
    #print(x.shape)
    return x


def normalize_among_channels(x):
    """
     Normalizes the values of each electrodes relatively to each other.Normalizes the value for each time stemp    
    :param x: trail to be normalized
    :return: normalized trail for each row
    """
    mean_time = np.expand_dims(x.mean(axis=1),axis=1)
    std_time = np.expand_dims(x.std(axis=1), axis=1)
    return (x-mean_time)/std_time


def normalize_through_time(x):
    """
    Normalized the value for each electrode throuhout the trail time 
    
    :param x: trail to be normalized
    :return: normalized trail for each column
    """
    mean_channel = np.expand_dims(x.mean(axis=0),axis=0)
    std_channel = np.expand_dims(x.std(axis=0), axis=0)
    std_channel[0,x.shape[1]-1]=1
    return (x-mean_channel)/std_channel


def map_to_2d(locs_3D):
    """
    Maps the 3D positions of the electrodes into 2D plane with AEP algorithm 
    
    :param locs_3D: matrix of shape number_of_electrodes x 3, for X,Y,Z coordinates respectively
    :return: matrix of shape number_of_electrodes x 2
    """
    locs_2D = []
    for e in locs_3D:
        locs_2D.append(azim_proj(e))
    
    return np.array(locs_2D)


def gen_images(locs, features, n_gridpoints, normalize=False,
               augment=False, pca=False, std_mult=0.1, n_components=2, edgeless=False):
    """
    source : https://github.com/pbashivan/EEGLearn/eeg_cnn_lib.py
    
     Generates EEG images given electrode locations in 2D space and multiple feature values for each electrode

    :param locs: An array with shape [n_electrodes, 2] containing X, Y
                        coordinates for each electrode.
    :param features: Feature matrix as [n_samples, n_features]
                                Features are as columns.
                                Features corresponding to each frequency band are concatenated.
                                (alpha1, alpha2, ..., beta1, beta2,...)
    :param n_gridpoints: Number of pixels in the output images
    :param normalize:   Flag for whether to normalize each band over all samples
    :param augment:     Flag for generating augmented images
    :param pca:         Flag for PCA based data augmentation
    :param std_mult     Multiplier for std of added noise
    :param n_components: Number of components in PCA to retain for augmentation
    :param edgeless:    If True generates edgeless images by adding artificial channels
                        at four corners of the image with value = 0 (default=False).
    :return:            Tensor of size [samples, colors, W, H] containing generated
                        images.
    """
    feat_array_temp = []
    nElectrodes = locs.shape[0]     # Number of electrodes
    # Test whether the feature vector length is divisible by number of electrodes
    #assert features.shape[1] % nElectrodes == 0
    n_colors = int(features.shape[1] / nElectrodes)
    for c in range(n_colors):
        feat_array_temp.append(features[:, c * nElectrodes : nElectrodes * (c+1)])

    nSamples = features.shape[0]
    # Interpolate the values
    grid_x, grid_y = np.mgrid[
                     min(locs[:, 0]):max(locs[:, 0]):n_gridpoints*1j,
                     min(locs[:, 1]):max(locs[:, 1]):n_gridpoints*1j
                     ]
    temp_interp = []
    for c in range(n_colors):
        temp_interp.append(np.zeros([nSamples, n_gridpoints, n_gridpoints]))
    # Generate edgeless images
    if edgeless:
        min_x, min_y = np.min(locs, axis=0)
        max_x, max_y = np.max(locs, axis=0)
        locs = np.append(locs, np.array([[min_x, min_y], [min_x, max_y],[max_x, min_y],[max_x, max_y]]),axis=0)
        for c in range(n_colors):
            feat_array_temp[c] = np.append(feat_array_temp[c], np.zeros((nSamples, 4)), axis=1)
    # Interpolating
    for i in range(nSamples):
        for c in range(n_colors):
            temp_interp[c][i, :, :] = griddata(locs, feat_array_temp[c][i, :], (grid_x, grid_y),
                                    method='cubic', fill_value=np.nan)
        print('Interpolating {0}/{1}\r'.format(i+1, nSamples), end='\r')
    # Normalizing
    for c in range(n_colors):
        if normalize:
            temp_interp[c][~np.isnan(temp_interp[c])] = \
                scale(temp_interp[c][~np.isnan(temp_interp[c])])
        temp_interp[c] = np.nan_to_num(temp_interp[c])
        
    print('Interpolated {0}/{1}\r'.format(nSamples, nSamples), end='\r')
    return np.swapaxes(np.asarray(temp_interp), 0, 1) 

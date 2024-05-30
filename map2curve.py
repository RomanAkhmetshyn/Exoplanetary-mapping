import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from importlib.metadata import version
from packaging.version import parse
from scipy.signal import convolve
from astropy.convolution import convolve_fft
import cartopy.crs as ccrs
import math


def kernel(x):
    '''
    Kernel for convolution. Returns max(cos(x),0)

    Parameters
    ----------
    x : array
        Array of π-wide window for convolution. 
        X Must be from -π/2 to π/2. 

    Returns
    -------
    kernel : array
        y data of the kernel.

    '''
    kernel = np.maximum(np.cos(x), 0)
    return kernel

def map2curve(slice_num, J, points_per_slice=10, kernel=kernel, true_len=None, plot=False):
    '''
    
    Generate full rotation phase curve from N longitudinal slices with brightness of J.

    Parameters
    ----------
    slice_num : int
        Number of longitudinal slices in the N-slice model which divide the sphere.
    J : array
        An array of slices brightness with a size of [slice_num,]. Brightness is prefered to range from 0 to 1.
    points_per_slice : int, optional
        Number of data points each slice takes in a longitudinal function. The more the smoother
        the result is. The default is 100.
    kernel : function, optional
        Kernel for seeing profile that calculates how much flux each slice
        contributes relative to its position on the observable hemisphere. 
        The default is truncated cosine kernel from N. Cowan & E. Agol (2008) (4). The
        function takes an array of x values which is [points_per_slice * slice_num / 2,]-long,
        in other words spans from -π/2 to π/2, and outputs an array of y values that will be 
        used in the convolution.
    plot : bool, optional
        Output plots of expanded data, kernel, convolution result and Aikoff projection
        of the self-luminous sphere with individual slices brightness. The default is False.

    Returns
    -------
    Array of the normalized phase curve size of [slice_num * points_per_slice,] 
    which corresponds to a full rotation.

    '''

    '''generate map and kernel'''

    points_per_slice=math.ceil(points_per_slice) #round up if the number is not int
    if true_len != None:
        data_points = true_len
    else:
        data_points=points_per_slice*slice_num #number of total datapoints for map and phase curve
    
    map_data = np.zeros(data_points)#initialize longitudinal map

    #fill in descreete values of J into longitudinal slices
    for i in range(slice_num):
        start_idx = i * points_per_slice
        end_idx = (i + 1) * points_per_slice   
        map_data[start_idx:end_idx] = J[i]
    
    # Triplicate the map
    triplicated_map = np.tile(map_data, 3)
    x_triplicated = np.linspace(-3 * np.pi, 3 * np.pi, 3 * data_points) #triplicated longitudes 
    
    #number of data points for kernel, which is half of total map points, because we observe half of the planet
    kernel_size = int(data_points/2) 
    
    #generate x values for the cosine kernel
    x_kernel = np.linspace(-np.pi/2, np.pi/2, kernel_size)
    
    kernel_data=kernel(x_kernel) #kernel y data
    
    #plot triplicated map and the kernel
    if plot==True:
    
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.xlim(-3*np.pi, 3*np.pi)
        x_ticks = [-3*np.pi, -2*np.pi, -np.pi, 0, np.pi, 2 * np.pi, 3 * np.pi]
        x_tick_labels = ['-3π', '-2π', '-π','0', 'π', '2π', '3π']
        plt.xticks(x_ticks, x_tick_labels)
        plt.axvline(-np.pi, color='r', linestyle='--', label='-π')
        plt.axvline(np.pi, color='r', linestyle='--', label='π')
        plt.plot(x_triplicated, triplicated_map, label="Map")
        plt.title("Map")
        plt.subplot(1, 2, 2)
        x_ticks = [ -np.pi/2, 0, np.pi/2]
        x_tick_labels = [ '-π/2','0', 'π/2']
        plt.xticks(x_ticks, x_tick_labels)
        plt.plot(x_kernel, kernel_data, label="Kernel")
        plt.title("Kernel")
        plt.show()

    
    convolution_result = np.zeros(data_points) #initialize phase curve 
    
    '''perform convolution'''
    
    #initial indexing for map data that is convolved at the beginning
    start_idx = int(data_points-data_points/4) 
    end_idx = int(data_points+data_points/4)
    
    #slide convolution window over whole longitudinal map 
    for i in range(data_points):
    
        window_slice = triplicated_map[start_idx+i:end_idx+i] #map data slice currently observed
        
        #check if window and kernel are the same size, this is needed when data is not divisible by slice_num etc
        if len(window_slice) < kernel_size:
            window_slice = np.pad(window_slice, (kernel_size - len(window_slice), 0))
        elif len(window_slice) > kernel_size:
            window_slice = window_slice[:kernel_size]
        
        #weighted sum of kernel and current part of the map 
        convolution_result[i] = np.sum(window_slice * kernel_data)
        
        #uncomment code below to see the progress ploted 
    
        # plt.figure(figsize=(12, 6))
        # plt.subplot(1, 2, 1)
        # plt.xlim(-3*np.pi, 3*np.pi)
        # x_ticks = [-3*np.pi, -2*np.pi, -np.pi, 0, np.pi, 2 * np.pi, 3 * np.pi]
        # x_tick_labels = ['-3π', '-2π', '-π','0', 'π', '2π', '3π']
        # plt.xticks(x_ticks, x_tick_labels)
        # plt.axvline(x_triplicated[start_idx+i], color='r', linestyle='--')
        # plt.axvline(x_triplicated[end_idx+i], color='r', linestyle='--')
        # plt.plot(x_triplicated, triplicated_map, label="Map")
        # plt.title("Map")
        
        
        # plt.subplot(1, 2, 2)
        # plt.plot(np.linspace(-np.pi, np.pi, data_points), convolution_result, label="Convolution")
        # plt.title("Convolution Result")
        # x_ticks = [ -np.pi, -np.pi/2, 0, np.pi/2, np.pi]
        # x_tick_labels = [ '-π','-π/2','0', 'π/2', 'π']
        # plt.ylim(0,1.5)
        # plt.xlabel('phase')
        # plt.xticks(x_ticks, x_tick_labels)
        # plt.ylabel('flux / mean flux')
        # plt.show()
    
    #normalize the result 
    convolution_result=convolution_result
    
    #plot convolved phase curve
    if plot==True:
        plt.plot(np.linspace(-np.pi, np.pi, data_points), convolution_result, label="Convolution")
        plt.title("Convolution Result")
        x_ticks = [ -np.pi, -np.pi/2, 0, np.pi/2, np.pi]
        x_tick_labels = [ '-π','-π/2','0', 'π/2', 'π']
        plt.xlabel('phase')
        plt.xticks(x_ticks, x_tick_labels)
        plt.ylabel('flux / mean flux')
        plt.show()
    
    #plot planet with slices and the phase curve
    if plot==True:
        lon = np.linspace(-180, 180, 360)
        lat = np.linspace(-90, 90, 180)
        lon2d, lat2d = np.meshgrid(lon, lat)
        data = np.zeros_like(lon2d)
        
        slices = np.array_split(lon, slice_num)
        
        # Assign each slice an integer value
        for i, slice_lon in enumerate(slices):
            data[(lon2d >= slice_lon[0]) & (lon2d <= slice_lon[-1])] = J[i]
        
        data_crs = ccrs.PlateCarree()
        
        fig = plt.figure(figsize=(6, 9))
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
        
        # Plot the first data in the upper subplot
        ax0 = plt.subplot(gs[0], projection=ccrs.Aitoff())
        ax0.set_title('Object in Aikoff projection')
        ax0.set_global()
        ax0.gridlines()
        
        cbar_ticks=np.linspace(0,0.1, 10)
        contour = ax0.contourf(lon, lat, data, transform=data_crs, cmap='gray', 
                               vmin=np.mean(cbar_ticks[:2]), 
                               vmax=np.mean(cbar_ticks[-2:]),
                               levels=cbar_ticks)
        
        cbar = plt.colorbar(contour, orientation='horizontal', ax=ax0, ticks=cbar_ticks)  
        cbar.set_label('normalized brightness')
        
        # Plot the second data in the lower subplot
        ax1 = plt.subplot(gs[1])
        ax1.plot(np.linspace(0, 2*np.pi, data_points), convolution_result)
        ax1.set_xlabel('Phase (rad)')
        ax1.set_ylabel('normalized flux')
        ax1.set_ylim([0, 2])
        ax1.set_xlim([0, 2*np.pi])
        
        x_ticks = [0, np.pi/2, np.pi, 3*np.pi/2 ,2 * np.pi]
        x_tick_labels = ['0', 'π/2' , 'π', '3π/2', '2π']
        
        ax1.set_xticks(x_ticks)
        ax1.set_xticklabels(x_tick_labels)
        
        plt.show()
        
    if true_len!=None:
        return convolution_result[:true_len]
    else:
        return convolution_result

if __name__ == "__main__":
    
    N=6
    J=np.random.uniform(0.0, 0.1, N)
    phase_curve=map2curve(N, J, plot=True)

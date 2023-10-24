import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from importlib.metadata import version
from packaging.version import parse
from scipy.signal import convolve
from astropy.convolution import convolve_fft
import cartopy.crs as ccrs


def kernel(x):
    kernel = np.maximum(np.cos(x_kernel), 0)
    return kernel
    

slice_num=6
J=np.random.uniform(0.0, 1.0, slice_num)

points_per_slice=100
plot=True

data_points=points_per_slice*slice_num
map_data = np.zeros(data_points)
#%%



for i in range(slice_num):
    start_idx = i * points_per_slice
    end_idx = (i + 1) * points_per_slice

    map_data[start_idx:end_idx] = J[i]

# Triplicate the map
triplicated_map = np.tile(map_data, 3)
x_triplicated = np.linspace(-3 * np.pi, 3 * np.pi, 3 * data_points)


kernel_size = int(data_points/2)
x_kernel = np.linspace(-np.pi/2, np.pi/2, kernel_size)

kernel_data=kernel(x_kernel)

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
#%%

convolution_result = np.zeros(data_points)

# Perform convolution

start_idx = int(data_points-data_points/4)
end_idx = int(data_points+data_points/4)

for i in range(data_points):

    # Get the slice of the map data within the window
    window_slice = triplicated_map[start_idx+i:end_idx+i]

    convolution_result[i] = np.sum(window_slice * kernel_data)

    # plt.xlim(-3*np.pi, 3*np.pi)
    # x_ticks = [-3*np.pi, -2*np.pi, -np.pi, 0, np.pi, 2 * np.pi, 3 * np.pi]
    # x_tick_labels = ['-3π', '-2π', '-π','0', 'π', '2π', '3π']
    # plt.xticks(x_ticks, x_tick_labels)
    # plt.axvline(x_triplicated[start_idx+i], color='r', linestyle='--')
    # plt.axvline(x_triplicated[end_idx+i], color='r', linestyle='--')
    # plt.plot(x_triplicated, triplicated_map, label="Map")
    # plt.title("Map")
    # plt.show()

    
convolution_result=convolution_result/np.mean(convolution_result)
plt.plot(np.linspace(-np.pi, np.pi, data_points), convolution_result, label="Convolution")
plt.title("Convolution Result")
x_ticks = [ -np.pi, -np.pi/2, 0, np.pi/2, np.pi]
x_tick_labels = [ '-π','-π/2','0', 'π/2', 'π']
plt.xlabel('phase')
plt.xticks(x_ticks, x_tick_labels)
plt.ylabel('flux / mean flux')
plt.show()

#%%
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

cbar_ticks=np.linspace(0,1.0, 10)
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




plt.tight_layout()
plt.show()

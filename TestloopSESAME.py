#!/usr/bin/env python
# coding: utf-8

# %%

from obspy.clients.fdsn import Client
from obspy import UTCDateTime
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
from matplotlib import pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from netCDF4 import Dataset





# %%
# Get stations

minlatitude = 28.5
maxlatitude = 36.5

minlongitude =  -87
maxlongitude =  -79


client = Client("IRIS")
starttime = UTCDateTime("2009-01-01")
endtime = UTCDateTime("2100-01-02")
inventory = client.get_stations(network="Z9", station="**",
                                starttime=starttime,minlatitude=minlatitude, maxlatitude=maxlatitude,
                                minlongitude=minlongitude, maxlongitude=maxlongitude,
                                endtime=endtime)

inventory.plot(projection="local",label=False,color_per_network=True) 
len('station')
if len  == '7':
                 print ("it is 7")
else:
                print ("it is more than 7 stations")
# %%
# Do the HVSR calculation

# Number of days of data to use in HVSR calculation
Ndays = 1

# Check if inventory is non-empty
if inventory:
   
    for network in inventory:
        for station in network:
            # Compute HVSR for each station
            
            start = (station.creation_date + 1*365*24*60*60).strftime('%Y-%m-%d')
            end = (station.creation_date + 1*365*24*60*60 + Ndays*24*60*60).strftime('%Y-%m-%d')
            
            sta = station.code
            loc = '--'
            net = network.code
            method = '4'
            get_ipython().system('python ./bin/computeHVSR.py net={net} sta={sta} loc={loc} chan=BHZ,BHN,BHE start={start} end={end} plot=1 plotbad=0 plotpsd=0 plotpdf=1 verbose=1 ymax=20 xtype=frequency n=1 removeoutliers=0 method={method}')




# %%
# Gather HVSR peak amplitudes and frequencies

#pathlist = Path('./data/hvsr/M4').glob('**/*.txt')
# pathlist = Path('/Users/birotimi/Library/CloudStorage/OneDrive-SyracuseUniversity/Desktop/HVSR-master/data/hvsr/M4').glob('**/*.txt')
pathlist = Path('./data/hvsr/M4').glob('**/*.txt')

plt.figure()
max_hvsr=np.array([])
max_freq=np.array([])
lats=np.array([])
lons=np.array([])
ii = 0
for path in pathlist:
    # because path is object not string
    path_in_str = str(path)   
    
    df = pd.read_csv(path_in_str, delimiter=' ')
#     print(df.head(5))
    
    net = path_in_str.split('/')[-1].split('.')[0]
    sta = path_in_str.split('/')[-1].split('.')[1]
    loc = path_in_str.split('/')[-1].split('.')[2]
    startime = path_in_str.split('/')[-1].split('.')[3]
    endtime = path_in_str.split('/')[-1].split('.')[4]
    
    # Read station lat lon
    inventory = client.get_stations(network=net, station=sta, starttime=starttime, endtime=endtime)
    lat = inventory[0][0].latitude
    lon = inventory[0][0].longitude
    lats = np.append(lats, lat)
    lons = np.append(lons, lon)
    
    # Plot HVSR
    plt.plot(df['frequency'],df['HVSR'],label=sta)
    plt.ylim((0,20))
    plt.xlim((0.09,18))
    plt.legend(bbox_to_anchor=(1.05,1.0),loc='upper left')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('HVSR')
    plt.xscale("log")
    
    # Find max HVSR and corresponding frequency
    ind_freq = df['frequency'] > 0.09
    ind = df['HVSR'][ind_freq].idxmax()
    max_hvsr = np.append(max_hvsr, df['HVSR'][ind])
    max_freq = np.append(max_freq, df['frequency'][ind])
    
    plt.plot(max_freq,max_hvsr,'ro')
    
    ii = ii + 1
    
plt.show()








# %%
# Load Topography dataset
#/Users/birotimi/Library/CloudStorage/OneDrive-SyracuseUniversity/Desktop/PHD/HVSR-project/HVSR-master
path2grdfile = '/Users/birotimi/Library/CloudStorage/OneDrive-SyracuseUniversity/Desktop/PHD/HVSR-project/HVSR-master/etopo1.grd'
etopodata = Dataset(path2grdfile)
lons_e = np.linspace(etopodata.variables['x_range'][0],
                    etopodata.variables['x_range'][1],
                    etopodata.variables['dimension'][0])
lats_e = np.linspace(etopodata.variables['y_range'][0],
                    etopodata.variables['y_range'][1],
                    etopodata.variables['dimension'][1])
etopo = etopodata.variables['z'][:]
etopo = np.reshape(etopo,etopodata.variables['dimension'][::-1])

# %%
# PLOT PEAK FREQUENCY

# Map projection
data_projection = ccrs.PlateCarree()

# Mask the bad stations
mask_bad_hvsr = (max_hvsr < 2) | (max_hvsr > 100) | (max_freq>18)

# Generate the map
ax = plt.axes(projection=data_projection)
ax.set_extent([minlongitude-0.5, maxlongitude+0.5, minlatitude-0.5, maxlatitude+0.5], crs=data_projection)
img_extent = (-180, 180, -90, 90)
pos = ax.imshow(etopo, origin='upper', extent=img_extent, transform=data_projection,
            cmap='gray', alpha=1, zorder=-1)
cb = plt.colorbar(pos, ax=ax, shrink=0.8)
cb.ax.set_title('Elev. (m)')
cmax = etopo.max()
cmin = etopo.min()
pos.set_clim(-200,1000)
ax.add_feature(cfeature.OCEAN,facecolor=cfeature.COLORS['water'])
ax.add_feature(cfeature.LAKES.with_scale('10m'), facecolor=cfeature.COLORS['water'], zorder=0)
# ax.add_feature(cfeature.RIVERS)
ax.coastlines(zorder=0)

# Plot bad stations first
ax.scatter(lons[mask_bad_hvsr],lats[mask_bad_hvsr],marker='o',s=50,c='white',edgecolor='black', transform=data_projection)
# Now plot good stations
sc = ax.scatter(
    lons[~mask_bad_hvsr],lats[~mask_bad_hvsr], marker='o', s=50, 
    # c=max_hvsr[~mask_bad_hvsr], 
    c=np.log10(max_freq[~mask_bad_hvsr]),
    cmap='Spectral',edgecolor='black', transform=data_projection)
cb = plt.colorbar(sc)
cb.set_label('Peak Frequency (Hz)')
# ax.set_xlim(minlongitude,maxlongitude)
# ax.set_ylim(minlatitude,maxlatitude)

gl = ax.gridlines(crs= data_projection, draw_labels=True,
                  linewidth=1, color='gray', alpha=0.5, linestyle='--', zorder=0)
gl.xlabels_top = False
gl.ylabels_right = False

plt.savefig('hvsr_peak_frequency.pdf')


# %%
# PLOT PEAK AMPLITUDE

# Generate the map
ax = plt.axes(projection=data_projection)
ax.set_extent([minlongitude-0.5, maxlongitude+0.5, minlatitude-0.5, maxlatitude+0.5], crs=data_projection)
img_extent = (-180, 180, -90, 90)
pos = ax.imshow(etopo, origin='upper', extent=img_extent, transform=data_projection,
            cmap='gray', alpha=1, zorder=-1)
cb = plt.colorbar(pos, ax=ax, shrink=0.8)
cb.ax.set_title('Elev. (m)')
pos.set_clim(-200,1000)
ax.add_feature(cfeature.OCEAN,facecolor=cfeature.COLORS['water'])
ax.add_feature(cfeature.LAKES.with_scale('10m'), facecolor=cfeature.COLORS['water'], zorder=0)
# ax.add_feature(cfeature.RIVERS)
ax.coastlines(zorder=0)

# Plot bad stations first
ax.scatter(lons[mask_bad_hvsr],lats[mask_bad_hvsr],marker='o',s=50,c='white',edgecolor='black', transform=data_projection)
# Now plot good stations
sc = ax.scatter(
    lons[~mask_bad_hvsr],lats[~mask_bad_hvsr], marker='o', s=50, 
    c=max_hvsr[~mask_bad_hvsr], 
    # c=max_freq[~mask_bad_hvsr],
    cmap='Spectral',edgecolor='black', transform=data_projection)
cb = plt.colorbar(sc)
cb.set_label('Peak HVSR')
sc.set_clim(2,10)
# ax.set_xlim(minlongitude,maxlongitude)
# ax.set_ylim(minlatitude,maxlatitude)

gl = ax.gridlines(crs=data_projection, draw_labels=True,
                  linewidth=1, color='gray', alpha=0.5, linestyle='--', zorder=0)
gl.xlabels_top = False
gl.ylabels_right = False

plt.savefig('hvsr_peak_amplitude.pdf')

# %%


# %%
# %%
# PLOT PEAK FREQUENCY

# Map projection
data_projection = ccrs.PlateCarree()

# Mask the bad stations
mask_bad_hvsr = (max_hvsr < 2) | (max_hvsr > 100) | (max_freq>18)

# Generate the map
ax = plt.axes(projection=data_projection)
ax.set_extent([minlongitude-0.5, maxlongitude+0.5, minlatitude-0.5, maxlatitude+0.5], crs=data_projection)
img_extent = (-180, 180, -90, 90)
pos = ax.imshow(etopo, origin='upper', extent=img_extent, transform=data_projection,
            cmap='gray', alpha=1, zorder=-1)
cb = plt.colorbar(pos, ax=ax, shrink=0.8)
cb.ax.set_title('Elev. (m)')
cmax = etopo.max()
cmin = etopo.min()
pos.set_clim(-200,1000)
ax.add_feature(cfeature.OCEAN,facecolor=cfeature.COLORS['water'])
ax.add_feature(cfeature.LAKES.with_scale('10m'), facecolor=cfeature.COLORS['water'], zorder=0)
ax.add_feature(cfeature.RIVERS)
ax.coastlines(zorder=0)

# Plot bad stations first
ax.scatter(lons[mask_bad_hvsr],lats[mask_bad_hvsr],marker='o',s=50,c='white',edgecolor='black', transform=data_projection)

# Now plot good stations
sc = ax.scatter(
    lons[~mask_bad_hvsr],lats[~mask_bad_hvsr], marker='o', s=50, 
    # c=max_hvsr[~mask_bad_hvsr], 
    c=max_freq[~mask_bad_hvsr],
    cmap='Spectral',edgecolor='black', transform=data_projection)
cb = plt.colorbar(sc)
cb.set_label('Peak Frequency (Hz)')

#test changed
#ax.set_xlim(minlongitude,maxlongitude)
#ax.set_ylim(minlatitude,maxlatitude)

gl = ax.gridlines(crs= data_projection, draw_labels=True,
                  linewidth=1, color='gray', alpha=0.5, linestyle='--', zorder=0)
gl.xlabels_top = False
gl.ylabels_right = False

plt.savefig('hvsr_peak_frequency.pdf')

# %%





pathlist = Path('.//Users/birotimi/Library/CloudStorage/OneDrive-SyracuseUniversity/Desktop/PHD/HVSR-project/HVSR-master/data/hvsr/M4').glob('**/*.txt')
plt.figure(figsize=(10, 8))
max_hvsr=np.array([])
max_freq=np.array([])
lats=np.array([])
lons=np.array([])
ii = 0
for path in pathlist:
    # because path is object not string
    path_in_str = str(path)   

    df = pd.read_csv(path_in_str, delimiter=' ')
    print(df.head(5))

    net = path_in_str.split('/')[-1].split('.')[0]
    sta = path_in_str.split('/')[-1].split('.')[1]
    loc = path_in_str.split('/')[-1].split('.')[2]
    startime = path_in_str.split('/')[-1].split('.')[3]
    endtime = path_in_str.split('/')[-1].split('.')[4]

    # Read station lat lon
    inventory = client.get_stations(network=net, station=sta, starttime=starttime, endtime=endtime)
    lat = inventory[0][0].latitude
    lon = inventory[0][0].longitude
    lats = np.append(lats, lat)
    lons = np.append(lons, lon)

    # Plot HVSR
    plt.plot(df['frequency'],df['HVSR'],label=sta)
    plt.ylim((0,20))
    plt.xlim((0.09,18))
    plt.legend(bbox_to_anchor=(1.05,1.0),loc='upper left')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('HVSR')
    plt.xscale("log")

    # Find max HVSR and corresponding frequency
    ind_freq = df['frequency'] > 0.09
    ind = df['HVSR'][ind_freq].idxmax()
    max_hvsr = np.append(max_hvsr, df['HVSR'][ind])
    max_freq = np.append(max_freq, df['frequency'][ind])

    plt.plot(max_freq,max_hvsr,'ro')

    ii = ii + 1


    plt.show()
    plt.figure()
    plt.plot(max_freq,max_hvsr,'o')
    plt.ylim((1,20))
    plt.xlim((0.09,18))
    plt.xlabel('Peak Frequency (Hz)')
    plt.ylabel('Peak HVSR')
    plt.xscale("log")
    plt.yscale("log")

    plt.plot(max_freq,max_hvsr,'ro')







# %%
pathlist = Path('./Users/birotimi/Library/CloudStorage/OneDrive-SyracuseUniversity/Desktop/PHD/HVSR-project/HVSR-master/data/hvsr/M4').glob('**/*.txt')

max_hvsr = np.array([])
max_freq = np.array([])
lats = np.array([])
lons = np.array([])


fig, ax = plt.subplots()

for path in pathlist:
    # because path is object not string
    path_in_str = str(path)

    df = pd.read_csv(path_in_str, delimiter=' ')
    #     print(df.head(5))

    net = path_in_str.split('/')[-1].split('.')[0]
    sta = path_in_str.split('/')[-1].split('.')[1]
    loc = path_in_str.split('/')[-1].split('.')[2]
    startime = path_in_str.split('/')[-1].split('.')[3]
    endtime = path_in_str.split('/')[-1].split('.')[4]

    # Read station lat lon
    inventory = client.get_stations(network=net, station=sta, starttime=starttime, endtime=endtime)
    lat = inventory[0][0].latitude
    lon = inventory[0][0].longitude
    lats = np.append(lats, lat)
    lons = np.append(lons, lon)

    # Plot HVSR for the current file
    ax.plot(df['frequency'], df['HVSR'], label=sta)


ax.set_ylim(0, 20)
ax.set_xlim(0.09, 18)
ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('HVSR')
ax.set_xscale("log")

# Display the figure
plt.show()
# %%
pathlist = Path('./Users/birotimi/Library/CloudStorage/OneDrive-SyracuseUniversity/Desktop/PHD/HVSR-project/HVSR-master/data/hvsr/M4').glob('**/*.txt')

max_hvsr = np.array([])
max_freq = np.array([])
lats = np.array([])
lons = np.array([])

plt.figure()
plt.plot(max_hvsr, max_freq, 'o')
plt.ylim((1,20))
plt.xlim((0.09,18))
plt.ylabel('max_hvsr')
plt.xlabel('max_freq')
plt.xscale("log")
plt.yscale("log")

plt.plot(max_freq,max_hvsr,'ro')
plt.show()

# %%
pathlist = Path('.//Users/birotimi/Library/CloudStorage/OneDrive-SyracuseUniversity/Desktop/PHD/HVSR-project/HVSR-master/data/hvsr/M4').glob('**/*.txt')
for path in pathlist:
    # because path is object not string
    path_in_str = str(path)

    df = pd.read_csv(path_in_str, delimiter=' ')
    print(df.head(5))
   
# %%
 net = path_in_str.split('/')[-1].split('.')[0]
    sta = path_in_str.split('/')[-1].split('.')[1]
    loc = path_in_str.split('/')[-1].split('.')[2]
    startime = path_in_str.split('/')[-1].split('.')[3]
    endtime = path_in_str.split('/')[-1].split('.')[4]
    
    # Read station lat lon
    inventory = client.get_stations(network=net, station=sta, starttime=starttime, endtime=endtime)
    lat = inventory[0][0].latitude
    lon = inventory[0][0].longitude
    lats = np.append(lats, lat)
    lons = np.append(lons, lon)

    max_hvsr=np.array([])
    max_freq=np.array([])
    lats=np.array([])
    lons=np.array([])
    plt.figure()
    ii = 0
    
    # Plot HVSR
    plt.plot(df['frequency'],df['HVSR'],label=sta)
    plt.ylim((0,20))
    plt.xlim((0.09,18))
    plt.legend(bbox_to_anchor=(1.05,1.0),loc='upper left')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('HVSR')
    plt.xscale("log")
    
    # Find max HVSR and corresponding frequency
    #ind_freq = df['frequency'] > 0.09
    #ind = df['HVSR'][ind_freq].idxmax()
    #max_hvsr = np.append(max_hvsr, df['HVSR'][ind])
    #max_freq = np.append(max_freq, df['frequency'][ind])
    
    #plt.plot(max_freq,max_hvsr,'ro')
    
#ii = ii + 1
    
plt.show()

# %%
pathlist = Path('.//Users/birotimi/Library/CloudStorage/OneDrive-SyracuseUniversity/Desktop/PHD/HVSR-project/HVSR-master/data/hvsr/M4').glob('**/*.txt')

# Create a single figure and axis for plotting
fig, ax = plt.subplots()

for path in pathlist:
    # because path is object not string
    path_in_str = str(path)

    df = pd.read_csv(path_in_str, delimiter=' ')

    net = path_in_str.split('/')[-1].split('.')[0]
    sta = path_in_str.split('/')[-1].split('.')[1]
    loc = path_in_str.split('/')[-1].split('.')[2]
    startime = path_in_str.split('/')[-1].split('.')[3]
    endtime = path_in_str.split('/')[-1].split('.')[4]

    # Read station lat lon
    inventory = client.get_stations(network=net, station=sta, starttime=starttime, endtime=endtime)
    lat = inventory[0][0].latitude
    lon = inventory[0][0].longitude

    # Plot HVSR on the same figure
    ax.plot(df['frequency'], df['HVSR'], label=sta)

    # Find max HVSR and corresponding frequency
    ind_freq = df['frequency'] > 0.09
    ind = df['HVSR'][ind_freq].idxmax()
    max_hvsr = df['HVSR'][ind]
    max_freq = df['frequency'][ind]

    # Plot the maximum HVSR point on the same figure
    ax.plot(max_freq, max_hvsr, 'ro')

# Set plot limits, labels, and legend
ax.set_ylim(0, 20)
ax.set_xlim(0.09, 18)
ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('HVSR')
ax.set_xscale("log")

plt.show()
# %%

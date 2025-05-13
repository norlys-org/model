import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
import cartopy.crs as ccrs
import json

with open('tests/secs_py.json', 'r') as file:
    data1 = json.load(file)
lons1 = np.array([v['lon'] for v in data1])
lats1 = np.array([v['lat'] for v in data1])
values1 = np.array([v['i'] for v in data1])

with open('tests/secs_rs.json', 'r') as file:
    data2 = json.load(file)
lons2 = np.array([v['lon'] for v in data2])
lats2 = np.array([v['lat'] for v in data2])
values2 = np.array([v['i'] for v in data2])

fig = plt.figure(figsize=(20, 10))

ax1 = fig.add_subplot(1, 2, 1, projection=ccrs.Orthographic(central_longitude=0, central_latitude=90))
ax1.coastlines()
ax1.gridlines()
cmap = cm.Reds
scatter1 = ax1.scatter(lons1, lats1,
                      transform=ccrs.PlateCarree(),
                      c=values1,
                      cmap=cmap,
                      vmin=0, vmax=400,
                      s=50)
plt.colorbar(scatter1, ax=ax1, label='Value (0-500)')
ax1.set_title('File 1: Orthographic Projection Over North Pole')

ax2 = fig.add_subplot(1, 2, 2, projection=ccrs.Orthographic(central_longitude=0, central_latitude=90))
ax2.coastlines()
ax2.gridlines()
scatter2 = ax2.scatter(lons2, lats2,
                      transform=ccrs.PlateCarree(),
                      c=values2,
                      cmap=cmap,
                      vmin=0, vmax=400,
                      s=50)
plt.colorbar(scatter2, ax=ax2, label='Value (0-500)')
ax2.set_title('File 2: Orthographic Projection Over North Pole')

plt.tight_layout()
plt.show()

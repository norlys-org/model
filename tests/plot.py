import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
import cartopy.crs as ccrs
import json

with open('tests/secs_py.json', 'r') as file:
    data = json.load(file)

lons = np.array([ v['lon'] for v in data ])
lats = np.array([ v['lat'] for v in data ])
values = np.array([ v['i'] for v in data ])

plt.figure(figsize=(10, 10))
ax = plt.axes(projection=ccrs.Orthographic(central_longitude=0, central_latitude=90))

ax.coastlines()
ax.gridlines()

cmap = cm.Reds

scatter = ax.scatter(lons, lats, 
                     transform=ccrs.PlateCarree(),
                     c=values, 
                     cmap=cmap,
                     vmin=0, vmax=400,
                     s=50)

plt.colorbar(scatter, label='Value (0-500)')

plt.title('Orthographic Projection Over North Pole')
plt.tight_layout()
plt.show()

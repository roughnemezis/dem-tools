import xarray as xr
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('module://matplotlib-backend-kitty')


dem_filepath = "/Users/merzisenh/Library/CloudStorage/ProtonDrive-h.merzisen@proton.me-folder/DATA/DEM_alp_L93_bilinear.tif"

dem = xr.open_dataset(dem_filepath).isel(band=0)
dem["band_data"] = dem.band_data.transpose("x","y")


dem.isel(x=slice(None, None, 10), y = slice(None, None, 10)).band_data.plot()
plt.show()

# sun elevation: 45°
# sun coming from south

dem

## approximations:
# - courbure de la terre negligée, ça peut se corriger (voir d'abord si c'est problématique)
# 

# attention aux coords: y d'abord, x ensuite
dem['plan'] = ((dem.y - dem.y.min())*0.5) + 4807



diff = (dem.plan - dem.band_data).transpose("x","y")

size_x = dem.sizes['x']
size_y = dem.sizes['y']

diff.min()
import numpy as np
import time


# begin step
start_time = time.time()

# shadow_mask = xr.full_like(dem.band_data, False, dtype=bool)
# peak_mask = xr.full_like(dem.band_data, False, dtype=bool)

# diff
# diff.values
index_of_min_in_y =  np.nanargmin(diff.values, axis=1) # valeurs de y ou le max est atteint pour chaque x (indices)
# attention les valeurs de la coordonnée y sont décroissantes donc les plus petits y correspondent aux plus grands indices

min_in_y_full_domain = np.broadcast_to(index_of_min_in_y, (size_y, size_x)).transpose()
y_full_domain = np.broadcast_to(range(size_y), (size_x, size_y))
mask_y_after  = (y_full_domain >= min_in_y_full_domain) # points derrière le relief pr au soleil
P_min_values_in_y = diff.values[range(size_x), index_of_min_in_y]

# xr.DataArray(P_min_values_in_y).plot()
# plt.show()


# diff.min()
# shadow_mask
# index_of_min_in_y

mask = mask_y_after & (diff.values > diff.values[range(size_x), index_of_min_in_y][:,None])

print(f"Overall processing took {round(time.time() - start_time, 2)} s")
# end step
# TODO: wrap in a function and try to iterate. How long does it take?

mask_xr = xr.DataArray(mask.astype(np.uint8))

mask_xr.isel(dim_1=slice(9450,None)).plot()
plt.show()

# it works!

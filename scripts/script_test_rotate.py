import xarray as xr
import numpy as np
from dem_tools.data import get_dem_path
from dem_tools.rotate import Transformer, get_rotated_dem, get_target_grid
from dem_tools.shadows import VectorisedShadowIterator


dem_o = (xr.open_dataset(get_dem_path())
       .isel(band=0).band_data
       .isel(x=slice(1500, 2000), y=slice(3000,3500))
       )

T = Transformer(dem_o)

dem = dem_o.pipe(T.center_dem)

theta = np.pi/8


target_grid = get_target_grid(dem_o)
demrot = get_rotated_dem(dem, target_grid, theta)
demback = get_rotated_dem(demrot, dem, -1*theta)



demback

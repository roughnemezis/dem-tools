import xarray as xr
import numpy as np
import matplotlib.pyplot as plt


class Transformer:

    def __init__(self, dem):
        self.x_center = dem.x.mean().item()
        self.y_center = dem.y.mean().item()

    def center_dem(self, dem):
        dem = dem.copy()
        dem.coords['x'] = dem.x - self.x_center
        dem.coords['y'] = dem.y - self.y_center
        return dem


# theta = np.pi/4
# theta = 0
# theta = np.pi/8



def get_target_grid(centered_dem):
    """
    strategy: create a grid large enough to fit all the rotations of the dem
    we could extend that of the dem? should be approx what we do
    grid is arbitrary: we won't necessarily find points that map in the original dem we have to choose wisely
    """
    dx = (centered_dem.x[1] - centered_dem.x[0]).item()
    nx = np.sqrt(2)*centered_dem.sizes['x']
    ny = np.sqrt(2)*centered_dem.sizes['y']
    xf = np.arange(-1*int(nx/2), int(nx/2))*dx
    yf = np.arange(-1*int(ny/2), int(ny/2))*dx
    grid = xr.DataArray(
        dims=("y", "x"),
        coords={"x": xf, "y": yf}
    )
    return grid



def get_rotated_dem(dem, target_grid, theta):
    x_map = target_grid.x * np.cos(theta) + target_grid.y * np.sin(theta)
    y_map = -1* target_grid.x * np.sin(theta) + target_grid.y * np.cos(theta)
    # easy to map each point to its index on the original grid since it's regular
    dx = (dem.x[1] - dem.x[0]).item()
    # are x / y in ascending / descending order?
    sgn_x = 2 * ((dem.x[-1] - dem.x[0]).item() > 0) - 1
    sgn_y = 2 * ((dem.y[-1] - dem.y[0]).item() > 0) - 1
    i_map = (sgn_x * np.rint((x_map - dem.x[0])/dx)).astype('int')
    j_map = (sgn_y * np.rint((y_map - dem.y[0])/dx)).astype('int')
    # # remove indexes falling outside of original grid
    valid = (
        (i_map >= 0) & (i_map < dem.sizes["x"]) &
        (j_map >= 0) & (j_map < dem.sizes["y"])
    )
    # have to rename if we isel with same dim names in the selection array its ambiguous
    demrot = dem.isel(
            x = i_map.where(valid, 0).rename(dict(x="xf", y="yf")),
            y = j_map.where(valid, 0).rename(dict(x="xf", y="yf"))
            ).transpose("yf", "xf")
    del demrot.coords['x']
    del demrot.coords['y']
    # # grid.values = demrot.values
    demrot = demrot.rename(dict(xf="x", yf="y"))
    demrot = demrot.where(valid, None)
    return demrot


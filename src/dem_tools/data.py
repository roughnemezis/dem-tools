import xarray as xr
import os
from pathlib import Path

for path in [
    Path("/rd/cenfic3/cenobs/home/merzisenh/shared_data/dem_tools"),
    Path(os.environ["HOME"]) / ".dem-tools-data",
]:
    if path.is_dir():
        root_data_path = path
        break


if os.path.isdir(path):
    root_data_path = path


def get_dem_path():
    return root_data_path / "DEM_alp_L93_bilinear.tif"

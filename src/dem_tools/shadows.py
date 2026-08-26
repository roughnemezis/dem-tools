import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import marimo as mo


class ShadowIterator():

    def __init__(self, dem, sun_elevation, strategy="max"):
        self.sun_elevation  = sun_elevation
        self.strategy = strategy
        self.dem = dem
        self.diff_dem = dem.shift({'y':1}, fill_value=None) - dem
        self.shadows = xr.full_like(dem, False, dtype=bool)
        self.peaks_to_explore = (
                xr.full_like(dem, True, dtype=bool)
                .where(self.diff_dem/30 <= -1*sun_elevation, False)
                )
        self.steps = []


    def iterate(self):
        if self.strategy == "max":
            max_y_idx = self.dem.where(self.peaks_to_explore).argmax(dim="y").item()
        else:
            max_y_idx = np.nonzero(self.peaks_to_explore.values)[0][-1]
        max_elevation = self.dem.isel(y=max_y_idx).item()
        max_y = self.dem.y.isel(y=max_y_idx).item()
        top_solar_ray = max_elevation-self.sun_elevation*(self.dem.y - max_y)
        shadows = ((self.dem < top_solar_ray) & (self.dem.y >= max_y))
        self.shadows |= shadows
        self.peaks_to_explore &= ~shadows
        self.peaks_to_explore[max_y_idx] = False
        # step related parameters
        self.steps.append(
                dict(
                    max_elevation= max_elevation,
                    max_y_idx=max_y_idx,
                    max_y=max_y,
                    shadows=shadows,
                    top_solar_ray=top_solar_ray,
                    explored=max_y_idx
                    )

                )
        return self


    def compute(self):
        try:
            while True:
                self.iterate()
        except ValueError, IndexError:
            print("iteration finished")
            return


class VectorisedShadowIterator:

    def __init__(self, dem, sun_elevation, name_x="x", name_y="y"):
        """
        aucune diff avec la version non vectorisée!
        """
        self.sun_elevation  = sun_elevation
        self.dem = dem.rename({name_x: "x", name_y: "y"})
        self.size_x = self.dem.sizes["x"]
        self.size_y = self.dem.sizes["y"]
        self.diff_dem = self.dem.shift({'y':1}, fill_value=None) - self.dem
        self.shadows = xr.full_like(self.dem, False, dtype=bool)
        self.peaks_to_explore = (
                xr.full_like(self.dem, True, dtype=bool)
                .where(self.diff_dem/30 <= -1*sun_elevation, False)
                )
        self.current_iteration = 0
        self.steps = []
        self.document_step()



    @property
    def current_y(self):
        return self.size_y - self.current_iteration - 1


    def iterate(self):
        all_idx_x =  np.nonzero(self.peaks_to_explore.isel(y=self.current_y).values)[0]
        elevations_explored = self.dem[dict(y=self.current_y, x=all_idx_x)]
        y_explored = elevations_explored.y.item()
        top_solar_ray = elevations_explored - self.sun_elevation*(self.dem.y - y_explored)
        self.shadows[dict(x=all_idx_x)] = (
                self.shadows[dict(x=all_idx_x)] 
                | ((self.dem < top_solar_ray) & (self.dem.y >= y_explored))
                 )
        self.peaks_to_explore &= ~self.shadows
        self.peaks_to_explore[dict(y=self.current_y, x=all_idx_x)] = False
        # documenting the step
        self.document_step()
        return self

    def document_step(self):
        self.steps.append(
                self.peaks_to_explore.isel(y=slice(None, self.current_y)).sum(dim="x")
                )


    def compute(self):
        for y in range(self.size_y):
            self.current_iteration = y
            self.iterate()


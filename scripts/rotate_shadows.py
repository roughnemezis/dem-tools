import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import xarray as xr
    import numpy as np
    from dem_tools.data import get_dem_path
    from dem_tools.rotate import Transformer, get_rotated_dem, get_target_grid
    from dem_tools.shadows import VectorisedShadowIterator


    dem = (xr.open_dataset(get_dem_path())
           .isel(band=0).band_data
           .isel(x=slice(3000, 3500), y=slice(3500,4000))
           )

    T = Transformer(dem)

    dem_centered = dem.pipe(T.center_dem)

    target_grid = get_target_grid(dem_centered)
    return (
        VectorisedShadowIterator,
        dem,
        dem_centered,
        get_rotated_dem,
        mo,
        np,
        plt,
        target_grid,
    )


@app.cell
def _(mo):
    theta_ui= mo.ui.slider(start=-1, stop=1, step=0.05, debounce=True)
    elevation_ui = mo.ui.slider(start=0.05, stop=2, step=0.05, debounce=True)
    mo.md(rf"$\theta$ {theta_ui},  Z{elevation_ui}")
    return elevation_ui, theta_ui


@app.cell
def _(elevation_ui, np, theta_ui):
    theta = theta_ui.value*np.pi
    Z_sun = elevation_ui.value
    return Z_sun, theta


@app.cell
def _(dem_centered, get_rotated_dem, target_grid, theta):
    demrot = get_rotated_dem(dem_centered, target_grid, theta)

    return (demrot,)


@app.cell
def _(VectorisedShadowIterator, Z_sun, demrot):
    # demback = get_rotated_dem(demrot, dem, -1*theta)

    SI = VectorisedShadowIterator(demrot, Z_sun)
    SI.compute()
    shadowsrot = SI.shadows
    return SI, shadowsrot


@app.cell
def _(
    SI,
    dem,
    dem_centered,
    demrot,
    elevation_ui,
    get_rotated_dem,
    mo,
    plt,
    shadowsrot,
    theta,
    theta_ui,
):
    shadows_theta = get_rotated_dem(SI.shadows, dem_centered, -1*theta)

    _fig0, _ax0 = plt.subplots()
    _fig1, _ax1 = plt.subplots()
    _fig2, _ax2 = plt.subplots()
    _fig3, _ax3 = plt.subplots()

    (1-shadowsrot).plot(ax=_ax0)
    (1-shadows_theta).plot(ax=_ax1)
    dem.plot(ax=_ax2)
    demrot.plot(ax=_ax3)

    mo.vstack([
        mo.md(rf"$\theta$ {theta_ui},  Z{elevation_ui}"),
        mo.hstack([_ax2, _ax3]),
        mo.hstack([_ax0, _ax1])
    ])
    return


if __name__ == "__main__":
    app.run()

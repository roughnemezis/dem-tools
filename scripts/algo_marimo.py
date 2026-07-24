import marimo

__generated_with = "0.23.15"
app = marimo.App(width="full", layout_file="layouts/algo_marimo.slides.json")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Démonstration de l'algo de calcul des masques solaires

    Prototype d'algorithme purement géométrique pour le calcul des masques solaires. Cette pré-version calcule les masques d'ombre sur les Alpes:
    - ☀️ pour l'élévation du soleil donnée en argument
    - ☀️ pour un DEM donné dans un système de coordonnées planaires (ici epsg2154)
    - 🔧 seulemnent pour un azimuth solaire correspondant à la direction Sud
    - 🔧 ne prend pas en compte la courbure du géoïde

    Les deux derniers points pourront être résolus par un pré-traitement du DEM (rotation du DEM, ajout d'une correction liée à la forme du géoîde).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(rf"""
    ## Besoins
    """)
    return


@app.cell
def _(plot_dem):
    plot_dem
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Sur ce DEM à 30 m pour les Alpes française, calculer une base de données des ombrages pour un maillage des azimuths et élévations solaires. La connaissance de la position solaire à un instant donné permettra alors de calculer les ombres pour cet instant.

    Cela représente quelques milliers de runs de l'algo. Le DEM contient environ 100 millions de points.

    On a besoin d'un algorithme rapide, même en one-shot, pour pouvoir calculer les masques en un temps de l'ordre de la semaine.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Principe de l'algorithme

    On raisonne sur une coupe Sud-Nord du DEM pour un rayonnement solaire venant du Sud. On parcourt les points du DEM du Sud au Nord, et pour chaque point on calcule son ombre portée en traçant le rayon de soleil passant par ce point et en déterminant tous les points en aval qui sont sous ce rayon - qui seront à l'ombre.

    On pourrait parcourir tous les points du DEM pour calculer leur ombrage, mais on peut éliminer un grand nombre de points de la recherche:
    - sur une condition de pente: si la pente est inférieure à la pente des rayons lumineux
    - si un point est dans l'ombrage d'un autre point, il n'est pas nécessaire de l'explorer
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Démonstration en image sur une coupe
    """)
    return


@app.cell
def _(elevation_pick, mo, plot_algo):
    mo.vstack([elevation_pick, plot_algo])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Le DEM est parcouru dans le sens des y croissants, les croix rouges correspondent aux points dont on calcule l'ombrage (non éliminés par les critères de pente ou ombrage)
    """)
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import xarray as xr
    from dem_tools.shadows import ShadowIterator
    from dem_tools.data import get_dem_path
    import matplotlib.pyplot as plt

    return ShadowIterator, get_dem_path, mo, plt, xr


@app.cell
def _(get_dem_path, xr):
    full_dem = xr.open_dataset(get_dem_path()).isel(band=0).band_data
    dem = full_dem.isel(x=4000)
    return dem, full_dem


@app.cell
def _(full_dem):
    plot_dem = full_dem.isel(x=slice(None,None,10), y=slice(None,None,10)).plot()
    return (plot_dem,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Et ensuite...

    - [x] vectoriser l'algorithme selon l'axe `x` (voir le notebook `algo_vectorized_marimo.py`)
    - [ ] prétraitement: appliquer au DEM une correction liée au géoïde
    - [ ] effectuer une rotation du DEM en prétraitement pour pouvoir calculer l'ombrage sur tous les azimuts
    - [ ] comparer l'algo vectorisé à d'autres méthodes (par exemple [le code SebVel de Laurane](https://github.com/LauraneC/SeBVel/tree/main) qui s'appuie sur [la librairie horayzon](https://github.com/ChristianSteger/HORAYZON))
    - [ ] construire une base de données d'ombrages pour un échantillonage des azimuts et élévation du soleil)
    """)
    return


@app.cell
def _(ShadowIterator, dem, elevation_pick):
    import time
    SI = ShadowIterator(dem, sun_elevation=elevation_pick.value, strategy="first")
    SI.compute()
    # 51 ms good!
    return (SI,)


@app.cell
def _(SI, mo, plt):
    _fig,_ax = plt.subplots(figsize=(16,5))

    _idx_explored = [int(idx["explored"]) for idx in SI.steps]
    _explored = SI.dem[_idx_explored]


    _ax.plot(SI.dem.y, SI.dem.values, linestyle="dashed")
    _shadows  = SI.dem.where(SI.shadows.values)
    _ax.plot(_shadows.y.values, _shadows.values, color="black", linewidth=3)
    _ax.plot(_explored.y.values, _explored.values, color="red", marker="+", markersize=8, linewidth=0)
    for _step in SI.steps:
        _max_idx = _step['max_y_idx']
        _top_ray = _step['top_solar_ray'].isel(y=slice(_max_idx-30, _max_idx+10))
        _ax.plot(_top_ray.y.values, _top_ray.values, linewidth=2, color="red")

    plot_algo = mo.mpl.interactive(_ax)
    # plot_algo = _ax
    return (plot_algo,)


@app.cell
def _(mo):
    elevation_pick = mo.ui.slider(start=0, stop=0.9, step=0.05, value=0.5, show_value=True, label="Changer la pente des rayons solaires", full_width=True, debounce=True)
    return (elevation_pick,)


if __name__ == "__main__":
    app.run()

# -*- coding: utf-8 -*-

############################################# To modify #########################################

"""
Indétermination si degré ou radiant étant un obstacle à la compréhension du code
A commenter avec générosité
Le caclcul des angles d'horizon peut être utilisé pour retourer le SVF plutôt que le calculer séparément
"""

############################################# Imports #########################################

from utils import *

from topocalc import gradient
from topocalc import viewf
from topocalc import horizon

import pvlib

import time
############################################# SVF #########################################

def compute_svf_regular_grid(
    run_dir: str = "." ,
    fic_topo : str  = "dem.nc",
    save_name : str = "topo_params.nc",
    ):
    """
    Compute SVF for the whole .nc topography in input
    """
    
    start = time.time()
    
    path_run_dir = Path(run_dir)
        
    ds_topo = xr.open_dataset(path_run_dir / fic_topo)
        
    # La projection Lambert 93 (ou epsg 2154) est en mètre donc conforme pour le calcul, pas besoin de changer en 32632
    #xs, ys = convert_epsg_pts(ds_topo.x.values,ds_topo.x.values, epsg_src = epsg_init, epsg_tgt=32632)
    dx = np.median(np.diff(ds_topo.x.values))
    dy = np.median(np.diff(ds_topo.y.values))

    svf = viewf.viewf(np.double(ds_topo.ZS.values), dx)[0] # Attention ZS et zs
    slope, aspect = gradient.gradient_d8(ds_topo.ZS.values, dx, dy)
    
    # Sauvegarde sous .nc
    ds_topo_params = xr.Dataset(
        data_vars=dict(
            ZS=(["y", "x"], ds_topo.ZS.values.astype(np.float64)),
            svf=(["y", "x"], svf.astype(np.float64)),
            slope=(["y", "x"], np.rad2deg(slope.astype(np.float64))),
            aspect=(["y", "x"], aspect.astype(np.float64)),
        ),
        coords=dict(
            x=("x", ds_topo.x.values),
            y=("y", ds_topo.y.values),
        )
    )
    
    ds_topo_params.ZS.attrs={'units': 'm', 'standard_name': 'elevation', 'long_name': 'elevation'}
    ds_topo_params.x.attrs = {'units': 'm'}
    ds_topo_params.y.attrs = {'units': 'm'}
    ds_topo_params.slope.attrs = {'units': 'rad'}
    ds_topo_params.aspect.attrs = {'units': 'rad'}
    ds_topo_params.svf.attrs = {'units': 'ratio', 'standard_name': 'svf', 'long_name': 'Sky view factor'}
    
    ds_topo_params.to_netcdf(path_run_dir / save_name)
    
    end = time.time()
    
    print("Topographic parameters OK")
    print(f"Total runtime calculating sky view factor : {round(end - start,3)} seconds")
    
############################################# Solar parameters #########################################

def from_dates_to_solar_angles(
    time_slice,
    run_dir : str,
    fic_dem : str,
    ):
    
    """
    Using pvlib, compute a netcdf of the elevation and azimuthal position of the sun
    during the input time_slice.
    """
    
    path_run_dir = Path(run_dir)
    
    ds_topo = xr.open_dataset(path_run_dir / fic_dem)
    
    xx,yy = np.meshgrid(ds_topo.x.values,ds_topo.y.values) # attention shape ny,nx
    lons,lats = convert_epsg_pts(xx,yy, epsg_src=2154, epsg_tgt=4326)
    
    nt = len(time_slice)
    ny,nx = lons.shape
    solar_pos = np.zeros((nt, ny, nx,2))
    
    # L'appel de pvlib n'est pas vectorisée sur le temps, en effet il suffirait de créer un array
    # avec une dimension temporelle répétant les longitudes/latitudes dans ses autres dimensions 
    # pvlib n'accepte que des tableaux de même longueur, pas de broadcasting implicite
    # Ce tableaux serait trop grand pour la mémoire si on considère toutes les Alpes
    # pour une année entière
    
    for t, timestamp in enumerate(time_slice): 
        
        # création d’un tableau de dates avec le même forme que lats.ravel()
        time_array = np.full(lats.ravel().shape, timestamp, dtype='datetime64[ns]')
        
        # Utilisation de 'elevation' et pas 'apparent_elevation' (prenant en plus en compte
        # la réfraction atmosphérique, surtout utile pour la levée/le coucher de soleil) pour des
        # calculs purement géométrique
        solar = pvlib.solarposition.get_solarposition(
            latitude  = lats.ravel(),   # tableau 1D de tous les pixels
            longitude = lons.ravel(),
            time      = time_array,
            altitude = ds_topo.ZS.values.ravel()
        )
        solar_pos[t, :, :, 0] = solar["azimuth"].values.reshape(ny, nx)
        solar_pos[t, :, :, 1] = solar["elevation"].values.reshape(ny, nx)
                
    ds_solar = xr.Dataset(
        data_vars=dict(
            elevation=(["time","y", "x"], solar_pos[:,:,:,1]),
            azimuth=(["time","y", "x"], solar_pos[:,:,:,0]),
        ),
        coords=dict(
            time=("time", time_slice),
            x=("x", ds_topo.x.values),
            y=("y", ds_topo.y.values),
        )
    )
    ds_solar.elevation.attrs={'units': 'degre', 'standard_name': 'elevation', 'long_name': 'solar elevation'}
    ds_solar.azimuth.attrs = {'units': 'degre', 'standard_name': 'azimuth', 'long_name': 'solar azimuth'}
    
    return ds_solar
    
############################################# Projection et ombres #########################################

@nb.njit(cache=True)
def proj(
    slope : float,
    aspect : float,
    azimuth : float,
    elevation : float,
    )-> float :
    
    # Function ...
    # Tout les angles en input sont en degrés, prendre l'angle solaire zénithal à la place de l'élévation
    # Donc le pi/2-elevation = zénithal
    fact = np.cos(np.pi/2-np.deg2rad(elevation))*np.cos(np.deg2rad(slope)) + np.sin(np.pi/2-np.deg2rad(elevation))*np.sin(np.deg2rad(slope))*np.cos(np.deg2rad(azimuth) - np.deg2rad(aspect))
    
    
    if fact < 0 :
        return 0
    else :    
        return fact
        
def horizon_angles_tab(
    DEM : np.ndarray,
    spacing : float,
    azimuths : np.ndarray, # Un azimuth tout les 5° pour plus d'efficacité
    )->np.ndarray :

    a,b = DEM.shape
    ha = np.zeros((len(azimuths),a,b))

    for i,azimuth in enumerate(azimuths):
        
        # azimuth-180 because the function horizon.horizon must receive azimuth bewteen -180/180°. With the 0 beeing the north
        # The horizon.horizon function returns the cosine of the horizon angle measured from the zenith, pi/2 - arccos(cos(ha)) 
        # gives the horizon angle as measured from the horizon
        ha[i,:,:] = np.pi/2-np.arccos(horizon.horizon(azimuth-180,DEM.astype(np.float64),spacing))
        
    return ha

@nb.njit(cache=True, parallel = True) 
def shadow(
    ha : np.ndarray, # en degrés, 3D
    azimuths_t : np.ndarray, # en degrés, 2D
    elevations_t : np.ndarray, # en degrés, 2D
    slope : np.ndarray, # degrees
    aspect : np.ndarray, # degrees
    pas_azimuth : int,
    )-> np.ndarray :
    
    """
    [Input]
    - Horinzon angle np.ndarray (72,nx,ny)
    - Azimuthal angle np.ndarray (nx,ny)
    - Elevation angle np.ndarray (nx,ny)
    - slope angle np.ndarray (nx,ny)
    - aspect angle np.ndarray (nx,ny)
    
    [Output]
    - Projection & shadow goal topography
    
    """

    a,b = elevations_t.shape
    res = np.zeros((a,b))

    # Using Numba, ~ no need to vectorize, loops are quickly executed
    # The parallelization is set only for the shadow_tab function to avoid nested parallelisme,
    # i.e. each t thread intenting using its own threads on i. It seems that it would slow the
    # calculation instead of making it more efficient
    
    for i in range(a):
        for j in range(b):
            
            # Finding the azimuthal direction to use inside the horizon angle netcdf
            azi_index = int((azimuths_t[i,j]+180) % 360 // pas_azimuth )
            
            # C'est bien l'élévation et pas l'angle zenithal : elevation + zenithal = pi/2
            if ha[azi_index,i,j] > elevations_t[i,j] :
                res[i,j] = 0
                
            else :
                res[i,j] = proj(slope = slope[i,j],
                                aspect = aspect[i,j],
                                azimuth = azimuths_t[i,j],
                                elevation = elevations_t[i,j])
    
    
    return res 

@nb.njit(cache=True, parallel = True)
def shadow_tab(
    ha : np.ndarray, # np.ndarray (72,nx,ny)
    elevations : np.ndarray, # np.ndarray (nt,nx,ny)
    azimuths : np.ndarray, # np.ndarray (nt,nx,ny)
    slope : np.ndarray, # np.ndarray (nx,ny)
    aspect : np.ndarray, # np.ndarray (nx,ny)
    pas_azimuth : int,
    )-> np.ndarray:
    
    """
    Extract in the (nt,nx,ny) solar parameters netcdfs the 2D np.ndarray of the considered time
    to handle it to the shadow function.
    """
    
    nt,nx,ny = elevations.shape
    shad = np.ones((nt,nx,ny))
    
    # Possibilité de paralléliser cette boucle sur du prange de njit ?
    for t in nb.prange(nt):
        
        shad[t,:,:] = shadow(
            ha = ha,
            azimuths_t = azimuths[t],
            elevations_t = elevations[t],
            slope = slope,
            aspect = aspect,
            pas_azimuth = pas_azimuth)
        
    return shad

def shadow_dataset(
    fic_topo_params : str = "topo_params.nc",
    fic_shadow : str = "ds_shadow.nc",
    run_dir: str  = "." ):
    
    """
    [Input]
    - name (str) of topographic parameters netcdf
    - name (str) of shadow mask netcdf to be saved
    - name (str) of run directory
    
    Using the topographic parameters file, compute the solar position for all 2026 year
    for each point of the MNT. Then computes the horizon angle netcdf storing the horizon angles 
    for all the point of the MNT in 72 azimuthal directions (one every 5 degrees). Using the two 
    latter netcdf, a shadow mask is computed and saved. The value shadow_mask of the shadow_mask 
    netcdf is in [0,1], account for self-shadow (slope angle superior to local elevation angle in
    the azmiuthal direction), projected shadow (shadding by surrounding topography) and direct beam 
    local projection 
    
    """
    
    start_time = time.time()
    
    path_run_dir = Path(run_dir)
    
    ############# Solar parameters ##################
    
    print("Computing solar parameters")
    
    # Constructing a time slice with pandas for all the year 2026
    start = pd.Timestamp('2026-01-01 00:00:00')
    end   = pd.Timestamp('2026-01-02 00:00:00') #  '2026-12-31 23:00:00'
    slice_t = slice(start, end)
    time_slice = pd.date_range(start, end, freq='h')
    
    path_run_dir = Path(run_dir)
    
    # Load topographic params with variables ZS,slope,aspect,svf
    ds_topo = xr.open_dataset(path_run_dir / fic_topo_params)
    
    dx = np.median(np.diff(ds_topo.x.values)) 
    xx, yy = np.meshgrid(ds_topo.x.values,ds_topo.y.values) # shape (ny, nx)
    
    # Changement systeme de coordonnées de Lambert 93 (epsg 2154) à epsg 4326 pour from_dates_to_solar_angles
    # Fonction de changment de coordonnées issue de TopoPyScale : https://github.com/ArcticSnow/TopoPyScale
    lons,lats = convert_epsg_pts(xx,yy, epsg_src=2154, epsg_tgt=4326)

    # Compute solar elevation and azimuths for all the year 2026 on all the DEM
    ds_solar = from_dates_to_solar_angles(
        run_dir = run_dir,
        fic_dem = fic_topo_params,
        time_slice = time_slice)
        
    # To analyse solar parameters if needed
    # ds_solar.to_netcdf(path_run_dir / "ds_solar.nc")
    
    print("Solar features OK")
    
    ############# Horizon angles ##################
    
    print("Computing horizon angles")
    
    # Compute the horizon angles for all the azimuths  (one transect every 5 azimuthal degree, 72 transect in total)
    pas_azimuth = 5
    azimuths = np.arange(0,360,pas_azimuth)
    ha = np.rad2deg(horizon_angles_tab(
                DEM = ds_topo.ZS.values,
                spacing = dx,
                azimuths = azimuths))

    # Gridded with xarray
    ds_ha = xr.Dataset(
        data_vars=dict(
            ha=(["azimuth","y", "x"], ha)
        ),
        coords=dict(
            azimuth=("azimuth", azimuths),
            y=("y", ds_topo.y.values),
            x=("x", ds_topo.x.values),
        )
    )
    ds_ha.ha.attrs={'units': 'degree','standard_name': 'Horizon angle', 'long_name': 'Horizon angle'}
    
    # To analyse horizon angles if needed
    #ds_ha.to_netcdf(path_run_dir / "ds_ha.nc")
    
    print("Horizon angles OK")
    
    ############# Shadow mask ##################
    
    print("Computing shadow mask")
    
    # Compute shadow array based on horizon angles, topographic parameters and solar position
    shadows = shadow_tab(
        ha = ds_ha.ha.values, # degrees
        elevations = ds_solar.elevation.values, # degrees
        azimuths = ds_solar.azimuth.values, # degrees
        slope = ds_topo.slope.values, # degrees
        aspect = ds_topo.aspect.values, # degrees
        pas_azimuth = pas_azimuth)  
    
    # Sauvegarde sous .nc
    ds_shadow = xr.Dataset(
        data_vars=dict(
            shadow_mask=(["time","y", "x"], shadows[:,:,:])
        ),
        coords=dict(
            time=("time", time_slice),
            x=("x", ds_topo.x.values),
            y=("y", ds_topo.y.values),
        )
    )
    ds_shadow.shadow_mask.attrs={'standard_name': 'Topo SWdir proj', 'long_name': 'Topographic SWdir projection factor'}
    
    ds_shadow.to_netcdf(path_run_dir / fic_shadow)
    
    end_time = time.time()
    
    print("Shadow mask OK")
    print(f"Total runtime calculating shadow mask and projection : {round(end_time - start_time,3)} seconds")
    
############################################# Calling function #########################################

if len(sys.argv) != 4:
    print("Usage: python3 compute_permanent_features.py run_dir dem.nc nom_experience")
    sys.exit(1)
    
# Les deux fonctions qui se succèdent ici procèdent en réalité aux même étapes. Une amélioration 
# notoire consisterait à utiliser le calcul optimisé des angles d'horizon dans tous les azimuths
# de la fonction viewf de topocalc pour le mask d'ombrage
    
# Computing svf
compute_svf_regular_grid(
    run_dir = sys.argv[1],
    fic_topo = sys.argv[2],
    save_name = f"topo_params_{sys.argv[3]}.nc")
    
# Computing shadow mask
shadow_dataset(
    run_dir = sys.argv[1],
    fic_topo_params = f"topo_params_{sys.argv[3]}.nc",
    fic_shadow = f"shadow_mask_{sys.argv[3]}.nc")
    

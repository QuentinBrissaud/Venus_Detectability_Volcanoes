from importlib import reload
import numpy as np
import pandas as pd
import Volcano_detectability.proba_modules as pm

cols = ['Lat_Center', 'Lon_Center', 'Height_km', 'mean_gaussian', 'std_gaussian', 'n_eruptions_year', 'std_n_eruptions_year', 'type_v']
folder_volcano = '/staff/quentin/Documents/Projects/2025_Sophus_MSc/data_volcanoes/'
files_volcanoes=[f'{folder_volcano}05_large_greaterthan100_sophus.csv', ]
type_volcanoes = ['large',]
pd_volcanoes = pd.DataFrame()
for file, type_v in zip(files_volcanoes, type_volcanoes):
    pd_volcanoes_loc = pm.get_volcano_stats(file)
    pd_volcanoes_loc['type_v'] = type_v
    pd_volcanoes = pd.concat([pd_volcanoes, pd_volcanoes_loc.loc[:,pd_volcanoes_loc.columns.isin(cols)]])
pd_volcanoes.reset_index(drop=True, inplace=True)
pd_volcanoes.loc[pd_volcanoes.Height_km>4.].to_csv(f'{folder_volcano}05_large_greaterthan100_sophus_Heightabove4km.csv', header=True, index=False)


opt_TL = dict(
    pattern_TL='balloon_tl_map_ust{}h_freq{}Hz_rangedep_60km.nc', 
    times_TL=np.arange(0,24),
    dropoff_hour_UST=0,
    folder_TL='/projects/restricted/infrasound/data/infrasound/2026_Venus_global_TLs/', 
)

folder_volcano = '/staff/quentin/Documents/Projects/2025_Sophus_MSc/data_volcanoes/'
#scaling_Byrne_to_iris = 42/(4.14*6)  
opt_volcanoes = dict(
    files_volcanoes=[#f'{folder_volcano}03_intermediate_5to100km.csv',
                     #f'{folder_volcano}04_all_greaterthan50km.csv',
                     #f'{folder_volcano}05_large_greaterthan100.csv', 
                     #f'{folder_volcano}05_large_greaterthan100_Heightabove4km.csv', 
                     #f'{folder_volcano}05_large_greaterthan100_wDEM_Heightabove4km.csv', 
                     f'{folder_volcano}05_large_greaterthan100_sophus.csv', 
                     f'{folder_volcano}05_large_greaterthan100_sophus_Heightabove4km.csv', 
                     ], 
    #n_eruptions_year=4.14*6, # original number from byrne
    #std_n_eruptions_year=2.13*6 # original number from byrne
    #n_eruptions_year=4.14*6*scaling_Byrne_to_iris, # correction Iris
    #std_n_eruptions_year=2.13*6*scaling_Byrne_to_iris # correction Iris
    n_eruptions_year=26.59, # correction Iris Table S3
    std_n_eruptions_year=6.25 # correction Iris Table S3
)

opt_amplitudes = dict(
    max_dist_linearfit = 19000., 
    min_dist_threshold_linearfit=1000.,
    dist_threshold_linearfit = 3000., 
    delta_dist = 1000.,
    perc_uncertainty = 1e-2,
    T_eruptions=np.logspace(0,2,6)[4:],
    r_to_vent=1000., 
    #rho_surface=65., 
    perc_rise_to_duration=0.05,
    use_eruption_magnitudes=True,
    use_stf_amplitudes=True, 
)

opt_probas = dict(
    M0s = np.linspace(1., 7., 30), 
    SNR_thresholds = np.linspace(0.1, 10., 50), 
    noise_level = 1e-2, 
    duration = 1./(365.*24.), 
    m_min = 1., 
    r_venus = 6052, 
    which_TL_distribution='normal', 
    s_batch_volcanoes=5
)

opt_traj = dict(
    time_max=3600*24*30*6, ## in seconds
    start_locations=[[-85.,0.], [-45.,0.], [0.,0.], [45.,0.], [85.,0.]], # lat, lon
    #start_locations=[[-45.,0.],], # lat, lon
    alt_balloon=60., 
    file_atmos='/staff/quentin/Documents/Projects/2024_Venus_Detectability/Venus_Detectability/data/VCD_atmos_globe_new.dat',
    trajectory_df=pd.read_csv('./trajectories/trajectory_paths.csv', header=[0])
)

# 0.5 Hz: Saturn
# 0.1 Hz: 
# 0.05 Hz: 
freqs = [0.5]
skip_ivolcano = None # None for no skip

## Sensitivity test: lat drop off, alt balloon, snr, freqs, volcanoes
from datetime import datetime
today_str = datetime.now().strftime("%Y-%m-%d")
today_str = '2026-05-01'
run_name = f'{opt_TL["folder_TL"]}trajectories_{today_str}_new' ## MODIFY EACH TIME
proba_models, trajectories = pm.get_all_probas(freqs, **opt_TL, **opt_volcanoes, **opt_amplitudes, **opt_probas, **opt_traj, run_name=run_name, skip_ivolcano=skip_ivolcano, folder_save_proba_map='/projects/restricted/infrasound/data/infrasound/2026_Venus_global_TLs/', overwrite=False)

#trajectories.to_csv('./trajectories/trajectories_scaling_24.02.2026.csv', header=True, index=False)
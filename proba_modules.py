from shapely.geometry import Polygon
from sklearn.cluster import DBSCAN
from tqdm import tqdm
import pandas as pd
from pyproj import Geod
import numpy as np
import geopandas as gpd 
import os
import seaborn as sns
from matplotlib.patches import Polygon as Polygon_mpl
from mpl_toolkits.basemap import Basemap
import matplotlib.cm as cm
import matplotlib.colors as mcol
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import FuncFormatter
import matplotlib.patches as mpatches
from scipy.signal import savgol_filter
from scipy import interpolate
from pyrocko import moment_tensor as pmt
from scipy import special
import matplotlib.pyplot as plt
from scipy.special import erf
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import curve_fit
from scipy.special import erf
import xarray as xr
import string
import json
import h5py

import sys
sys.path.append('/staff/quentin/Documents/Projects/2024_Venus_Detectability/Venus_Detectability')
import VCD_trajectory_modules as VCD

def get_regions(dir_venus_data):

    PATH_VENUS_DATA = os.path.join(dir_venus_data)
    PATH_VENUS = os.path.join(f"{PATH_VENUS_DATA}tectonic_settings_Venus")
    VENUS = {
        'corona': gpd.read_file(f"{PATH_VENUS}/corona.shp"),
        'rift': gpd.read_file(f"{PATH_VENUS}/rifts.shp"),
        'ridge': gpd.read_file(f"{PATH_VENUS}/ridges.shp")
    }
    return VENUS

def get_volcano_locations(file_volcano):
    volcanoes = pd.read_csv(file_volcano, header=[0])
    return volcanoes        

def get_slopes(file_slopes):
    pd_slopes = pd.read_csv(file_slopes, header=[0])
    return pd_slopes        

# Define the integrand function
def integrand(t, sigma):
    return 1 + erf(t / sigma)

# Perform the integration
def integral_f(Tmax, sigma, times=np.linspace(0., 1., 100)):
    
    init_shape = Tmax.shape
    Tmax = Tmax.ravel()
    sigma = sigma.ravel()
    id_TMAX = np.arange(Tmax.size)
    TIMES, IDs = np.meshgrid(times, id_TMAX)
    TIMES *= Tmax[IDs]
    DT = np.diff(TIMES, axis=1)[:,0]
    #print(IDs.shape, IDs.max(), Tmax.shape, SIGMA.shape, DT.shape)
    integral = np.sum(integrand(Tmax[IDs], sigma[IDs]), axis=1)*DT
    return integral.reshape(init_shape)

def compute_EI_from_VEI(alt_balloon=50., rho_air=65., std_shadow_zone=100., vei=np.linspace(0., 6., 100), Tmax=np.linspace(10., 100., 20), r=np.linspace(10., 400., 50)):

    TMAX_total, R_total = np.meshgrid(Tmax, r)
    SIGMA_total = TMAX_total/2.

    R_balloon = lambda R: np.sqrt(R**2+alt_balloon**2)
    acoustic_zone = lambda R: np.exp(-(R/std_shadow_zone)**2)

    factor_GF = (1./(integral_f(TMAX_total, SIGMA_total)))*(rho_air/(8*np.pi*1e3))*(1e9)
    factor_GF_median = np.median(factor_GF)
    factor_GF_q25 = np.quantile(factor_GF, q=0.25)
    factor_GF_q75 = np.quantile(factor_GF, q=0.75)
    TL_new = lambda dist, VEI: acoustic_zone(dist)*factor_GF_median*(1./R_balloon(dist))*10**(VEI-5)
    TL_new_qmin = lambda dist, VEI: acoustic_zone(dist)*factor_GF_q25*(1./R_balloon(dist))*10**(VEI-5)
    TL_new_qmax = lambda dist, VEI: acoustic_zone(dist)*factor_GF_q75*(1./R_balloon(dist))*10**(VEI-5)

    return TL_new, TL_new_qmin, TL_new_qmax

def convert_VEI_to_Mw(vjet = np.linspace(0.1, 0.5, 80), rho_tephra = np.linspace(1., 3.5, 40), vps = np.linspace(2., 5., 20), factor_scaling = np.logspace(-2, 1, 10)):

    FACTOR_conv, VP_conv, RHO_conv, VJET_conv = np.meshgrid(factor_scaling, vps, rho_tephra, vjet)

    factor_mw = lambda factor_scaling, vp, rho_tephra, vjet: -np.log10((factor_scaling*vp*1e3*vjet*1e3*rho_tephra*1e3))-4.
    output = factor_mw(FACTOR_conv, VP_conv, RHO_conv, VJET_conv)
    factor_mw_median = np.quantile(output, q=0.5)
    factor_mw_q25 = np.quantile(output, q=0.25)
    factor_mw_q75 = np.quantile(output, q=0.75)
    VEI_to_Mw_median = lambda vei: (vei - factor_mw_median)*(2./3.) -6.07
    VEI_to_Mw_q25 = lambda vei: (vei - factor_mw_q25)*(2./3.) -6.07
    VEI_to_Mw_q75 = lambda vei: (vei - factor_mw_q75)*(2./3.) -6.07

    return VEI_to_Mw_median, VEI_to_Mw_q25, VEI_to_Mw_q75

#####################
## AMPLITUDE MODEL ##
#####################

def get_TL_curves_with_EI(file_curve, dist_min=100., alt_balloon=50., rho0=50, rhob=1.):

    TL_seismic_new, TL_seismic_new_qmin, TL_seismic_new_qmax = get_TL_curves(file_curve, dist_min)

    VEI_to_Mw_median, VEI_to_Mw_q25, VEI_to_Mw_q75 = convert_VEI_to_Mw()
    TL_seismic_new_VEI = lambda dist, VEI: TL_seismic_new(dist, VEI_to_Mw_median(VEI))
    TL_seismic_new_qmin_VEI = lambda dist, VEI: TL_seismic_new_qmin(dist, VEI_to_Mw_median(VEI))
    TL_seismic_new_qmax_VEI = lambda dist, VEI: TL_seismic_new_qmax(dist, VEI_to_Mw_median(VEI))

    TL_EI_new, TL_EI_new_qmin, TL_EI_new_qmax = compute_EI_from_VEI(alt_balloon=alt_balloon)

    
    density_ratio = np.sqrt(rhob/rho0)

    TL_new = lambda dist, VEI: np.maximum(density_ratio*TL_EI_new(dist, VEI), TL_seismic_new_VEI(dist, VEI))
    TL_new_qmin = lambda dist, VEI: np.maximum(density_ratio*TL_EI_new_qmin(dist, VEI), TL_seismic_new_qmin_VEI(dist, VEI))
    TL_new_qmax = lambda dist, VEI: np.maximum(density_ratio*TL_EI_new_qmax(dist, VEI), TL_seismic_new_qmax_VEI(dist, VEI))

    return TL_new, TL_new_qmin, TL_new_qmax

def get_TL_curves_one_freq(pd_all_amps_in, freq, dist_min, rho0, rhob, cb, use_savgol_filter, scalar_moment, unknown):

    pd_all_amps = pd_all_amps_in.copy()
    if 'fmax' in pd_all_amps.columns:
        diff = (freq>=pd_all_amps.fmin) & (freq<=pd_all_amps.fmax) & ~((pd_all_amps.fmin==0.)&(pd_all_amps.fmax==1.)) # Remove the full spectrum case
        pd_all_amps = pd_all_amps.loc[diff]
        
    if 'median_rw' in pd_all_amps.columns:
        x = pd_all_amps.dist.values/1e3
        median = pd_all_amps['median_rw'].values
        q25 = pd_all_amps['median_q0.25_rw'].values
        q75 = pd_all_amps['median_q0.75_rw'].values
    else:
        x = pd_all_amps.dist.unique()/1e3
        median = pd_all_amps.groupby(['dist'])['amp_RW'].median().reset_index().amp_RW.values
        q25 = pd_all_amps.groupby(['dist'])['amp_RW'].quantile(q=0.25).reset_index().amp_RW.values
        q75 = pd_all_amps.groupby(['dist'])['amp_RW'].quantile(q=0.75).reset_index().amp_RW.values

    median /= scalar_moment
    q25 /= scalar_moment
    q75 /= scalar_moment

    if use_savgol_filter:
        
        fs = []
        for y in [median, q25, q75]:
            y_smooth = np.zeros_like(y)
            window_size = 5  # Must be odd
            poly_order = 3
            y_smooth[:] = savgol_filter(y[:], window_size, poly_order)
            f = interpolate.interp1d(x, y_smooth, bounds_error=False, fill_value=(y_smooth[0], y_smooth[-1]))
            fs.append(f)

        f_mean = fs[0]
        f_qmin = fs[1]
        f_qmax = fs[2]
    
    else:
        ## Rayleigh waves
        f_mean = interpolate.interp1d(x, median, bounds_error=False, fill_value=(median[0], median[-1]))
        f_qmin = interpolate.interp1d(x, q25, bounds_error=False, fill_value=(q25[0], q25.iloc[-1]))
        f_qmax = interpolate.interp1d(x, q75, bounds_error=False, fill_value=(q75[0], q75[-1]))

        ## Body waves
        """
        f_mean = interpolate.interp1d(degrees2kilometers(pd_all_amps.dist.values), pd_all_amps.median_body.values, bounds_error=False, fill_value=(pd_all_amps.median_body.iloc[0], pd_all_amps.median_body.iloc[-1]))
        f_qmin = interpolate.interp1d(degrees2kilometers(pd_all_amps.dist.values), pd_all_amps['median_q0.25_body'].values, bounds_error=False, fill_value=(pd_all_amps['median_q0.25_body'].iloc[0], pd_all_amps['median_q0.25_body'].iloc[-1]))
        f_qmax = interpolate.interp1d(degrees2kilometers(pd_all_amps.dist.values), pd_all_amps['median_q0.75_body'].values, bounds_error=False, fill_value=(pd_all_amps['median_q0.75_body'].iloc[0], pd_all_amps['median_q0.75_body'].iloc[-1]))
        """

    TL_base_seismic = lambda dist, m0: pmt.magnitude_to_moment(m0)*f_mean(dist)
    #TL_base_seismic_std = lambda dist, m0: pmt.magnitude_to_moment(m0)*f_std(dist)
    TL_base_seismic_qmin = lambda dist, m0: pmt.magnitude_to_moment(m0)*f_qmin(dist)
    TL_base_seismic_qmax = lambda dist, m0: pmt.magnitude_to_moment(m0)*f_qmax(dist)

    ## Mag vs amp relationship -> https://gfzpublic.gfz-potsdam.de/rest/items/item_65142/component/file_292577/content
    """
    TL_base_seismic_disp = lambda dist, m0: 10**(m0 -1.66*np.log10(kilometers2degrees(dist)) -3.3)*period # eq. 3
    TL_base_seismic = lambda dist, m0: 1e-6*TL_base_seismic_disp(dist, m0)*2*np.pi/period
    """

    density_ratio = np.sqrt(rho0/(rhob))
    if unknown == 'pressure':
        density_ratio *= rhob*cb
        
    #TL_base = lambda dist, m0: density_ratio*(TL_base_seismic(dist,m0)*1e-6)/(2*np.pi*period) # Raphael
    TL_base = lambda dist, m0: density_ratio*(TL_base_seismic(dist,m0))
    TL_base_qmin = lambda dist, m0: density_ratio*(TL_base_seismic_qmin(dist,m0))
    TL_base_qmax = lambda dist, m0: density_ratio*(TL_base_seismic_qmax(dist,m0))
    
    TL_new = lambda dist, m0: TL_base(dist, m0)*(dist>=dist_min) + TL_base(dist_min, m0)*(dist<dist_min)
    TL_new_qmin = lambda dist, m0: TL_base_qmin(dist, m0)*(dist>=dist_min) + TL_base_qmin(dist_min, m0)*(dist<dist_min)
    TL_new_qmax = lambda dist, m0: TL_base_qmax(dist, m0)*(dist>=dist_min) + TL_base_qmax(dist_min, m0)*(dist<dist_min)

    return TL_new, TL_new_qmin, TL_new_qmax

def get_TL_curves(file_curve, freq, depths=[0, 50], dist_min = 100., rho0=50., rhob=1., cb=250., use_savgol_filter=False, plot=False, scalar_moment=1, unknown='pressure', return_dataframe=False):

    only_one_TL = False
    if isinstance(freq, float):
        freq = [freq]
        only_one_TL = True
    elif not isinstance(freq, list):
        print("The variable is neither a float nor a list.")
        return None, None, None

    pd_all_amps = pd.read_csv(file_curve, header=[0])
    pd_all_amps = pd_all_amps.loc[(pd_all_amps.depth>=depths[0]*1e3)&(pd_all_amps.depth<=depths[1]*1e3)]
    
    TL_new, TL_new_qmin, TL_new_qmax = dict(), dict(), dict()
    for one_freq in freq:
        TL_new_loc, TL_new_qmin_loc, TL_new_qmax_loc = get_TL_curves_one_freq(pd_all_amps, one_freq, dist_min, rho0, rhob, cb, use_savgol_filter, scalar_moment, unknown)
        TL_new[one_freq] = TL_new_loc
        TL_new_qmin[one_freq] = TL_new_qmin_loc
        TL_new_qmax[one_freq] = TL_new_qmax_loc

        #if plot:
        #    plot_TL(file_curve, TL_new_loc, TL_new_qmin_loc, TL_new_qmax_loc, unknown)

    if only_one_TL:
        TL_new, TL_new_qmin, TL_new_qmax = TL_new_loc, TL_new_qmin_loc, TL_new_qmax_loc

    if return_dataframe:
        return TL_new, TL_new_qmin, TL_new_qmax, pd_all_amps
    else:
        return TL_new, TL_new_qmin, TL_new_qmax

from scipy.stats import lognorm
def get_lognormal_one_freq(pd_all_amps_in, freq, dist_min, rho0, rhob, cb, use_savgol_filter, scalar_moment, unknown, factor_lower_shape):

    pd_all_amps = pd_all_amps_in.copy()
    if 'fmax' in pd_all_amps.columns:
        diff = (freq>=pd_all_amps.fmin) & (freq<=pd_all_amps.fmax) & ~((pd_all_amps.fmin==0.)&(pd_all_amps.fmax==1.)) # Remove the full spectrum case
        pd_all_amps = pd_all_amps.loc[diff]


    xloc = pd_all_amps.dist.unique()
    x, shape, scale = np.zeros(xloc.size), np.zeros(xloc.size), np.zeros(xloc.size)
    idist = -1
    for dist, pd_all_amps_dist in pd_all_amps.groupby('dist'):
        idist += 1
        s1, _, c1 = lognorm.fit( pd_all_amps_dist.amp_RW.values, floc=0)
        shape[idist] = s1/factor_lower_shape
        scale[idist] = c1
        x[idist] = dist/1e3

        if False:
            Y = lognorm(s=s1, loc=0., scale=c1)
            x = np.logspace(np.log10(pd_all_amps_dist.amp_RW.values.min()), np.log10(pd_all_amps_dist.amp_RW.values.max()), 20)
            plt.figure()
            plt.hist(pd_all_amps_dist.amp_RW.values, bins=x)
            #plt.plot(x, Y.pdf(x))
            plt.xscale('log')
            plt.yscale('log')
            plt.title(f'{freq}Hz - {dist}')
            break

    scale /= scalar_moment

    if use_savgol_filter:
        
        fs = []
        for y in [shape, scale,]:
            y_smooth = np.zeros_like(y)
            window_size = 5  # Must be odd
            poly_order = 3
            y_smooth[:] = savgol_filter(y[:], window_size, poly_order)
            f = interpolate.interp1d(x, y_smooth, bounds_error=False, fill_value=(y_smooth[0], y_smooth[-1]))
            fs.append(f)

        f_shape = fs[0]
        f_scale = fs[1]
    
    else:
        ## Rayleigh waves
        f_shape = interpolate.interp1d(x, shape, bounds_error=False, fill_value=(shape[0], shape[-1]))
        f_scale = interpolate.interp1d(x, scale, bounds_error=False, fill_value=(scale[0], scale[-1]))

    shape_base_seismic = lambda dist: f_shape(dist)
    scale_base_seismic = lambda dist, m0: pmt.magnitude_to_moment(m0)*f_scale(dist)

    density_ratio = np.sqrt(rho0/(rhob))
    if unknown == 'pressure':
        density_ratio *= rhob*cb
        
    #TL_base = lambda dist, m0: density_ratio*(TL_base_seismic(dist,m0)*1e-6)/(2*np.pi*period) # Raphael
    #shape_base = lambda dist, m0: density_ratio*(shape_base_seismic(dist,m0))
    scale_base = lambda dist, m0: density_ratio*(scale_base_seismic(dist,m0))
    
    shape_new = lambda dist: shape_base_seismic(dist)*(dist>=dist_min) + shape_base_seismic(dist_min)*(dist<dist_min)
    scale_new = lambda dist, m0: scale_base(dist, m0)*(dist>=dist_min) + scale_base(dist_min, m0)*(dist<dist_min)
    
    #plt.figure()
    #plt.plot(x, scale_new(x, 3.))

    #dists = np.arange(0, 18000., 5.)
    #plt.figure()
    #plt.plot(scale_new(dists, 3))

    return shape_new, scale_new

def get_lognormal_curves(file_curve, freq, depths=[0, 50], dist_min = 100., rho0=50., rhob=1., cb=250., use_savgol_filter=False, plot=False, scalar_moment=1, unknown='pressure', threshold_amp=1e-28, return_dataframe=False, factor_lower_shape=1.):

    only_one_TL = False
    if isinstance(freq, float):
        freq = [freq]
        only_one_TL = True
    elif not isinstance(freq, list):
        print("The variable is neither a float nor a list.")
        return None, None, None

    pd_all_amps = pd.read_csv(file_curve, header=[0])
    pd_all_amps = pd_all_amps.loc[(pd_all_amps.depth>=depths[0]*1e3)&(pd_all_amps.depth<=depths[1]*1e3)&(pd_all_amps.amp_RW>=threshold_amp)]
    
    shape_new, scale_new = dict(), dict()
    for one_freq in freq:
        shape_new_loc, scale_new_loc = get_lognormal_one_freq(pd_all_amps, one_freq, dist_min, rho0, rhob, cb, use_savgol_filter, scalar_moment, unknown, factor_lower_shape)
        shape_new[one_freq] = shape_new_loc
        scale_new[one_freq] = scale_new_loc

    if only_one_TL:
        shape_new, scale_new = shape_new_loc, scale_new_loc

    if return_dataframe:
        return shape_new, scale_new, pd_all_amps
    else:
        return shape_new, scale_new

def return_one_interp(TL, density_ratio, amp_scaling):

    f_mean = interpolate.interp1d(TL.distance.values, TL.amp_median.values, bounds_error=False, fill_value=(TL.amp_median.values[0], TL.amp_median.values[-1]))
    f_qmin = interpolate.interp1d(TL.distance.values, TL.amp_qmin.values, bounds_error=False, fill_value=(TL.amp_qmin.values[0], TL.amp_qmin.values[-1]))
    f_qmax = interpolate.interp1d(TL.distance.values, TL.amp_qmax.values, bounds_error=False, fill_value=(TL.amp_qmax.values[0], TL.amp_qmax.values[-1]))

    TL_new = lambda dist, m0: density_ratio*pmt.magnitude_to_moment(m0)*f_mean(dist)/amp_scaling
    TL_new_qmin = lambda dist, m0: density_ratio*pmt.magnitude_to_moment(m0)*f_qmin(dist)/amp_scaling
    TL_new_qmax = lambda dist, m0: density_ratio*pmt.magnitude_to_moment(m0)*f_qmax(dist)/amp_scaling

    return TL_new, TL_new_qmin, TL_new_qmax


def return_one_interp_lognormal(TL, density_ratio, amp_scaling):

    f_shape = interpolate.interp1d(TL.distance.values, TL['shape'].values, bounds_error=False, fill_value=(TL['shape'].values[0], TL['shape'].values[-1]))
    f_scale = interpolate.interp1d(TL.distance.values, TL['scale'].values, bounds_error=False, fill_value=(TL['scale'].values[0], TL['scale'].values[-1]))

    shape_new = lambda dist: f_shape(dist)
    scale_new = lambda dist, m0: density_ratio*pmt.magnitude_to_moment(m0)*f_scale(dist)/amp_scaling
    #print(density_ratio, amp_scaling)
    #print(f_scale(100.), scale_new(100., 3.))

    return shape_new, scale_new

def get_TL_curves_precomputed(file_curve, rho0=50., rhob=1., c0=400., cb=250., unknown='pressure', model='Cold100'):

    TL_all = pd.read_csv(file_curve, header=[0])
    comp = 'v' if unknown in ['pressure', 'velocity'] else 'u'
    TL_all = TL_all.loc[(TL_all.model == model)&(TL_all.comp == comp)]
    m0_default = TL_all.m0_default.iloc[0]
    amp_scaling = pmt.magnitude_to_moment(m0_default)

    density_ratio = np.sqrt(rho0*c0/(rhob*cb))
    if unknown == 'pressure':
        density_ratio *= rhob*cb

    TL_new, TL_new_qmin, TL_new_qmax = dict(), dict(), dict()
    for one_freq, TL in TL_all.groupby('freq'):

        TL_new_loc, TL_new_qmin_loc, TL_new_qmax_loc = return_one_interp(TL, density_ratio, amp_scaling)

        TL_new[one_freq] = TL_new_loc
        TL_new_qmin[one_freq] = TL_new_qmin_loc
        TL_new_qmax[one_freq] = TL_new_qmax_loc

    return TL_new, TL_new_qmin, TL_new_qmax

def get_lognormal_precomputed(file_curve, rho0=50., rhob=1., c0=400., cb=250., unknown='pressure', model='Cold100'):

    TL_all = pd.read_csv(file_curve, header=[0])
    comp = 'v' if unknown in ['pressure', 'velocity'] else 'u'
    TL_all = TL_all.loc[(TL_all.model == model)&(TL_all.comp == comp)]
    m0_default = TL_all.m0_default.iloc[0]
    amp_scaling = pmt.magnitude_to_moment(m0_default)

    density_ratio = np.sqrt(rho0*c0/(rhob*cb))
    if unknown == 'pressure':
        density_ratio *= rhob*cb

    shape_new, scale_new = dict(), dict()
    for one_freq, TL in TL_all.groupby('freq'):

        shape_new_loc, scale_new_loc = return_one_interp_lognormal(TL, density_ratio, amp_scaling)

        shape_new[one_freq] = shape_new_loc
        scale_new[one_freq] = scale_new_loc

    return shape_new, scale_new

#######################
## PROBA CLASS MODEL ##
#######################

def lognorm_cdf(x, shape, loc=0.0, scale=1.0):
    x = np.asarray(x)
    #m = x > loc
    #z = np.log((x[m] - loc) / scale) / shape
    z = np.log((x - loc) / scale) / shape
    out = np.zeros_like(z, dtype=float)             # CDF = 0 for x <= loc
    # Numerically stable form (avoid 1 - tiny):
    #print(out.shape, m.shape, z.shape)
    #out[m] = 0.5 * special.erfc(-z / np.sqrt(2.0))
    out = 0.5 * special.erfc(-z / np.sqrt(2.0))
    return out

def haversine(lon1, lat1, lon2, lat2, r = 6052.):
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    # Calculate the result
    return c * r

def poly2d(dists, coefs):
    return np.sum([coefs[:,:,icoef]*(dists**icoef) for icoef in range(coefs.shape[-1])], axis=0)

def line_through_two_points(x1, y1, x2, y2):
    a = (y2 - y1) / (x2 - x1)
    b = y1 - a * x1
    return a, b

class proba_model:

    def __init__(self, pd_volcanoes, pd_amplitudes, max_dist_linearfit = 19000., min_dist_threshold_linearfit=100, dist_threshold_linearfit = 500., delta_dist = 1000., perc_uncertainty=1e-2, T_eruption=100., r_to_vent=1000., rho_surface=65., rho_exit=300., perc_rise_to_duration=0.05, TL_array_provided=None, use_eruption_magnitudes=False,):
        
        self.T_eruption = T_eruption # duration in s
        self.r_to_vent = r_to_vent # Distance from vent in m
        self.rho_surface = rho_surface # surface atmospheric density
        self.rho_exit = rho_exit # tephra density
        self.perc_rise_to_duration = perc_rise_to_duration
        self.error_height_volcano_acceptable = 2 # km, when looking for right TL receiver altitude, what is the max error acceptable compared to the actual volcano height

        self.use_eruption_magnitudes = use_eruption_magnitudes
        self._get_mags_vei_distrib()
        #self.scaler_stf = 1.
        #self.apply_scaler_stf = apply_scaler_stf
        #self.freq_scaling = freq_scaling
        #if apply_scaler_stf is not None:
        #    self.scaler_stf = self._get_stf_scaler(apply_scaler_stf) # apply_scaler_stf sould be a frequency

        self.pd_volcanoes = pd_volcanoes
        self.lat_volcanoes = pd_volcanoes.Lat_Center.values
        self.lon_volcanoes = pd_volcanoes.Lon_Center.values
        self.height_volcanoes = pd_volcanoes.Height_km.values

        self.mean_gaussian = pd_volcanoes.mean_gaussian.values
        self.std_gaussian = pd_volcanoes.std_gaussian.values
        self.n_eruptions_year = pd_volcanoes.n_eruptions_year.values
        self.std_n_eruptions_year = pd_volcanoes.std_n_eruptions_year.values
        #self.slopes = pd_volcanoes.slope.values
        #self.intercepts = pd_volcanoes.intercept.values # dists

        self.max_dist_linearfit = max_dist_linearfit
        self.min_dist_threshold_linearfit = min_dist_threshold_linearfit
        self.dist_threshold_linearfit = dist_threshold_linearfit # (km) distance at which we start linear fit of TL data
        self.delta_dist = delta_dist # (km) spatial step of linear fit after self.dist_threshold_linearfit
        self.perc_uncertainty = perc_uncertainty

        self.pd_amplitudes = pd_amplitudes
        self._prepare_amplitude_model(TL_array_provided) # dists x loc

    """
    def _get_stf_scaler(self, freq_input,):

        t_rise = self.perc_rise_to_duration*self.T_eruption
        a = 0.5*self.T_eruption
        b = 0.5*(self.T_eruption + 2*t_rise)   # so b-a = t_rise

        omega_c = 2.0 / np.sqrt((a + b) * (b - a))
        env_global = lambda freq: (a + b) / (1.0 + ((2.*np.pi*freq)/omega_c)**2)
        return env_global(freq_input)/env_global(self.freq_scaling)
    """

    def _get_mags_vei_distrib(self):
        a_mu, b_mu = line_through_two_points(1., 1.5, 5., 5.) # in eruption mags, Figure 2 https://doi.org/10.1029/2022RG000783
        self.mu_vei = lambda VEI: a_mu*VEI + b_mu
        self.sigma_vei = lambda VEI: 0.35 + 0.*VEI # in eruption mags, Figure 2 https://doi.org/10.1029/2022RG000783
        
    def _prepare_amplitude_model(self, TL_array_provided):

        if TL_array_provided is None:
            g = Geod(ellps='WGS84')

            #n_loc = self.pd_amplitudes.groupby(['lat', 'lon'])['TL'].first().reset_index().shape[0]
            n_lat = self.pd_amplitudes.lat.unique().size
            n_lon = self.pd_amplitudes.lon.unique().size
            n_az = self.pd_amplitudes.az.unique().size
            n_dist = self.pd_amplitudes.dist.unique().size
            #dists_unique = self.pd_amplitudes.dist.unique()
            n_height = self.pd_amplitudes.height.unique().size
            shape_init = (n_az, n_dist, n_height) #(48, 96, 12, 300, 7)

            self.lat_balloons_default = np.zeros((n_lat, n_lon))
            self.lon_balloons_default = np.zeros((n_lat, n_lon))
            self.TL_array = np.zeros((n_lat, n_lon, self.lat_volcanoes.size)) # lat x lon x dists
            #self.TL_array = []

            ilat = -1
            for lat, pd_amplitudes_balloon_lat in tqdm(self.pd_amplitudes.groupby(['lat']), total=n_lat, desc='Preparing amplitudes'): ## Loop over balloon locations
                ilat += 1

                ilon = -1
                for lon, pd_amplitudes_balloon in pd_amplitudes_balloon_lat.groupby(['lon']): ## Loop over balloon locations
                    
                    debug = False
                    if debug:
                        cond_ok = abs(lon-150.)<4. and abs(lat-51.6)<3
                        if not cond_ok:
                            continue
                    
                    ilon += 1
                    azimuths = pd_amplitudes_balloon.az.values.reshape(shape_init)
                    dists = pd_amplitudes_balloon.dist.values.reshape(shape_init)
                    TL_loc = pd_amplitudes_balloon.TL.values.reshape(shape_init)
                    heights = pd_amplitudes_balloon.height.values.reshape(shape_init)

                    dist_max = dists[0,:,0].max()
                    dist_extrapolation = np.arange(dist_max+1., self.max_dist_linearfit, self.delta_dist)
                    s_extrapolation = dist_extrapolation.size
                    #idx = dists[0,:,0] > self.dist_threshold_linearfit
                    idx_bef = (dists[0,:,0] >= self.min_dist_threshold_linearfit)&(dists[0,:,0] <= self.dist_threshold_linearfit)
                    def _fit_1d(y):
                        coeffs = np.polyfit(dists[0,idx_bef,0], y, 1)  # degree 1 = linear
                        """
                        if coeffs[0] > 0:
                            print(f'Positive coef: {coeffs[0]}')
                        else:
                            print(f'Negative coef: {coeffs[0]}')
                        """
                        coeffs[0] = min(coeffs[0],0)
                        coeffs[1] = coeffs[1] if coeffs[0] < 0 else y[-1]
                        extrapolation = coeffs[0]*dist_extrapolation + coeffs[1]

                        #extrapolation = y[0]-10*np.log10(dist_extrapolation-dists[0,idx,0].min())
                        return extrapolation
                    
                    TL_new = np.apply_along_axis(_fit_1d, 1, TL_loc[:,idx_bef,:])
                    #TL_loc = np.concatenate((TL_loc[:,~idx,:], TL_new), axis=1)
                    #print('TL_loc', TL_loc.shape, TL_new.shape)
                    TL_loc = np.concatenate((TL_loc[:,:,:], TL_new), axis=1)

                    def expand_to_all_axes(unknown, axis):
                        new_shape = ()
                        new_shape_output = ()
                        for ishape, shape in enumerate(TL_new.shape):
                            if ishape == axis:
                                new_shape += (1,)
                                new_shape_output += (unknown.size,)
                            else:
                                new_shape += (shape,)
                                new_shape_output += (1,)
                        unknown_expanded = unknown.reshape(new_shape_output)
                        return np.tile(unknown_expanded, new_shape)

                    dist_extrapolation = expand_to_all_axes(dist_extrapolation, 1)
                    #dists = np.concatenate((dists[:,~idx,:], dist_extrapolation), axis=1)
                    dists = np.concatenate((dists[:,:,:], dist_extrapolation), axis=1)

                    if debug:
                        if cond_ok:
                            TL_new_base = TL_new[:,0,0]
                            dist_extrapolation_base = np.arange(dist_max+self.delta_dist, self.max_dist_linearfit, self.delta_dist)
                            plt.figure()
                            for ii in range(TL_loc.shape[0]):
                                print(f'---- {azimuths[ii,0,0]} ----')
                                print(TL_new_base[ii])
                                print(dist_extrapolation_base)
                                print(np.sqrt(dist_extrapolation_base[0]/dist_extrapolation_base))
                                plt.plot(dists[ii,:,0], TL_loc[ii,:,0], label=azimuths[ii,0,0])
                                plt.plot(dist_extrapolation_base, TL_new_base[ii]*np.sqrt(dist_extrapolation_base[0]/dist_extrapolation_base), color='black', ls='--')
                            plt.axvline(self.min_dist_threshold_linearfit, color='black')
                            plt.axvline(self.dist_threshold_linearfit, color='black')
                            plt.axvline(dist_max, color='black')
                            
                            plt.legend()
                            plt.xlim([0, 4000])
                            plt.ylim([-500, 0])

                    #azimuths = np.concatenate((azimuths[:,~idx,:], np.repeat(azimuths[:,0:1,:], s_extrapolation, axis=1)), axis=1)
                    #heights = np.concatenate((heights[:,~idx,:], np.repeat(heights[:,0:1,:], s_extrapolation, axis=1)), axis=1)
                    azimuths = np.concatenate((azimuths[:,:,:], np.repeat(azimuths[:,0:1,:], s_extrapolation, axis=1)), axis=1)
                    heights = np.concatenate((heights[:,:,:], np.repeat(heights[:,0:1,:], s_extrapolation, axis=1)), axis=1)

                    azimuths, dists, TL_loc, heights = azimuths.ravel(), dists.ravel(), TL_loc.ravel(), heights.ravel()

                    endlon, endlat, _ = g.fwd(lon+np.zeros_like(azimuths), lat+np.zeros_like(azimuths), azimuths, dists*1e3)
                    
                    id_TL, id_volcanoes = np.meshgrid(np.arange(azimuths.size), np.arange(self.lat_volcanoes.size)) # dists x n_points_TL
                    
                    idx_closest_TL = np.sqrt((endlon[id_TL]-self.lon_volcanoes[id_volcanoes])**2+(endlat[id_TL]-self.lat_volcanoes[id_volcanoes])**2)
                    idx_closest_H = abs(self.height_volcanoes[id_volcanoes]-heights[id_TL])
                    idx_closest_TL[idx_closest_H>self.error_height_volcano_acceptable] += 1e6
                    idx_closest_TL = idx_closest_TL.argmin(axis=1)

                    self.TL_array[ilat, ilon,:] = TL_loc[idx_closest_TL]
                
                    self.lat_balloons_default[ilat, ilon] = lat
                    self.lon_balloons_default[ilat, ilon] = lon

            #self.TL_array = np.array(self.TL_array).T # dists x loc
            #self.lat_balloons_default = np.array(self.lat_balloons_default) # dists 
            #self.lon_balloons_default = np.array(self.lon_balloons_default) # dists 

        else:
            self.TL_array, self.lat_balloons_default, self.lon_balloons_default = TL_array_provided

    def _compute_uncertainty_pressure(self, mu):
        sigma = self.perc_uncertainty*mu/self.noise_level # Pa / noise in Pa
        return sigma

    @staticmethod
    def return_number_per_VEI_and_volcano(mw, n_eruptions, mean_gaussian, std_gaussian):
        #return 10**(np.log10(mw[:,None])*slopes[None,:]+intercepts[None,:])
        #return n_eruptions * (1/(std_gaussian[None,:]*np.sqrt(2*np.pi))) * np.exp(-((mw[:,None] - mean_gaussian[None,:])**2) / (2 * std_gaussian[None,:]**2))
        return n_eruptions * ( 1. - 0.5 * (1+erf((mw[:,None] - mean_gaussian[None,:])/(std_gaussian[None,:]*np.sqrt(2)))) )

    @staticmethod
    def _TL_to_pressure(p0, TL_volcanoes):
        return p0*10**(TL_volcanoes.T[None,:,None,:]/20.) # self.TL_array: loc x dists

    def _VEI_to_pressure(self, VEI):

        volumes = 10**(4.+VEI)

        rise_time = self.perc_rise_to_duration*self.T_eruption

        #return (self.rho_surface/(4*np.pi*self.r_to_vent))*(1/(rise_time))*(volumes/(self.T_eruption+rise_time)) ## Assuming VEI
        return (1./(4*np.pi*self.r_to_vent))*(1/(rise_time))*(volumes/(self.T_eruption+rise_time)) ## Assuming eruption magnitude

    def _mags_to_VEI(self, cums, MAG_r):

        cums_c = cums.copy()
        mags = MAG_r[:,0,0,0]
        dmags = abs(mags[1]-mags[0])
        for ivei, vei in enumerate(mags):
            mu_vei = self.mu_vei(vei)
            sigma_vei = self.sigma_vei(vei)
            #gauss_test = (1./(sigma_vei*np.sqrt(2.*np.pi)))*np.exp(-0.5*((mags[ivei]-mu_vei)/(sigma_vei))**2)
            #print(f'VEI {vei}: mu={mu_vei:.2f} sigma={sigma_vei:.2f} - gauss_test: {gauss_test:.2f}')
            cums[ivei,:,:,:] = np.sum(cums_c*(1./(sigma_vei*np.sqrt(2.*np.pi)))*np.exp(-0.5*((MAG_r[:,0:1,0:1,0:1]-mu_vei)/(sigma_vei))**2), axis=0)*dmags

        return cums

    def compute_cum_pdf(self, DISTS, MAG, DETECT_T, TL_volcanoes):

        DISTS_r, MAG_r, DETECT_T_r =  DISTS.reshape(self.shape_init), MAG.reshape(self.shape_init), DETECT_T.reshape(self.shape_init)
        # shape_init: M0s x dists x SNR x loc
        
        if self.which_TL_distribution == 'normal':
            mu = self._TL_to_pressure(self._VEI_to_pressure(MAG_r[:,0:1,0:1,0:1]), TL_volcanoes)/self.noise_level # M0s x dists x SNR x loc
            #raise ValueError("Debug")
            sigma = self._compute_uncertainty_pressure(mu)

            cums = 1-0.5*(1+special.erf( (DETECT_T_r[0:1,0:1,:,0:1]-mu)/(sigma*np.sqrt(2.)) )) # M0s x dists x SNR x loc

            if self.use_eruption_magnitudes:
                #print(f'MAG_r> {MAG_r.shape} ({self.shape_init})')
                cums = self._mags_to_VEI(cums, MAG_r)

        elif self.which_TL_distribution == 'lognormal':

            raise ValueError("Fix lognormal first")

            x = DISTS_r[:,:,0:1,:] # M0s x dists x 1
            scale = self.scaler(x, MAG_r[:,0:1,0:1,0:1])/self.noise_level
            if self.shaper_has_mw:
                shape = self.shaper(x, MAG_r[:,0:1,0:1,0:1])
            else:
                shape = self.shaper(x)
            F_total = lognorm_cdf(DETECT_T_r[0:1,0:1,:,0:1], shape, scale=scale)
            
            cums      = 1.0 - F_total # M0s x dists x SNR x loc
            
        else:
            raise ValueError("TL distribution must be normal or lognormal")
        
        return cums.ravel()
        
    def integrate_cum_pdf(self, DISTS, MAG, DETECT_T, F_MAGS, TL_volcanoes):
        total_pdf = self.compute_cum_pdf(DISTS, MAG, DETECT_T, TL_volcanoes) # M0s x dists x SNR x loc
        
        #print(DISTS.shape, MAG.shape, DETECT_T.shape, F_MAGS.shape, TL_volcanoes.shape, self.shape_init, total_pdf.shape, )

        integrated = self.dproba_M0s*(F_MAGS*total_pdf.reshape(self.shape_init)).sum(axis=(0,)) # dists x SNR x loc

        #print(integrated)
        #raise ValueError("Debug")
    
        return integrated

    def compute_rate(self, DISTS, MAG, DETECT_T, F_MAGS, TL_volcanoes, idx_volcanoes):

        integrals = self.integrate_cum_pdf(DISTS, MAG, DETECT_T, F_MAGS, TL_volcanoes) # dists x SNR x loc

        #print('integrals', integrals.min(), integrals.max())
        #raise ValueError("Debug")

        Nerup_over_mag_min = self.return_number_per_VEI_and_volcano(np.array([self.m_min]), self.n_eruptions_year[idx_volcanoes], self.mean_gaussian[idx_volcanoes], self.std_gaussian[idx_volcanoes]) # 1 (M0s) x dists
        
        #print(self.shape_init, integrals.shape, Nquake_over_mag.shape, self.dproba_M0s)
        #print(self.dproba_M0s*integrals.sum(axis=0))
        #raise ValueError("Debug")

        return (Nerup_over_mag_min[0,:,None,None]*integrals).sum(axis=0) # SNR x loc

    def compute_Poisson(self, rates):
        
        #print(rates.min(), rates.max())
        #print(rates)
        #raise ValueError("Debug")

        return 1.-np.exp(-self.duration*rates) # SNR x loc

    def get_ratios_famp(self, idx_volcanoes):

        Nerup_over_mag = self.return_number_per_VEI_and_volcano(self.M0s, self.n_eruptions_year[idx_volcanoes], self.mean_gaussian[idx_volcanoes], self.std_gaussian[idx_volcanoes]) # M0s x dists
        Nerup_over_mag_min = self.return_number_per_VEI_and_volcano(np.array([self.m_min]), self.n_eruptions_year[idx_volcanoes], self.mean_gaussian[idx_volcanoes], self.std_gaussian[idx_volcanoes]) # 1 (M0s) x dists

        Nerup_mag = (Nerup_over_mag_min-Nerup_over_mag)/Nerup_over_mag_min
        f_mag = np.gradient(Nerup_mag, self.dproba_M0s, edge_order=2, axis=0)

        return f_mag[:,:,np.newaxis,np.newaxis] # M0s x dists x 1 (SNR) x 1 (loc)

    def _init_discretization(self):

        self.dproba_M0s = self.M0s[1]-self.M0s[0]

    def init_parameter_space(self, lat_volcanoes, lon_volcanoes, lats, lons, locations):

        idx_dists = np.arange(lon_volcanoes.size)
        #print(idx_dists.size, self.M0s.size, self.SNR_thresholds.size, locations.size)
        IDX_DISTS, MAG, DETECT_T, LOC = np.meshgrid(idx_dists, self.M0s, self.SNR_thresholds, locations) # M0s x dists x SNR x loc
        #print(IDX_DISTS.shape)
        self.shape_init = IDX_DISTS.shape
        IDX_DISTS, MAG, DETECT_T, LOC = IDX_DISTS.ravel(), MAG.ravel(), DETECT_T.ravel(), LOC.ravel()

        DISTS = haversine(lons[LOC], lats[LOC], lon_volcanoes[IDX_DISTS], lat_volcanoes[IDX_DISTS])
        #print(DISTS.shape)

        return DISTS, MAG, DETECT_T

    def compute_proba_map(self, M0s, SNR_thresholds, noise_level, duration, m_min, r_venus, which_TL_distribution='normal', default_p_uncertainty=10, s_batch_volcanoes=5, verbose=False, disable_tqdm=False):

        self.M0s = M0s
        self.SNR_thresholds = SNR_thresholds
        self.noise_level = noise_level
        self.duration = duration
        self.all_lats = self.lat_balloons_default[:,0] ## TO fix maybe to allow custom set of lats/lons + interpolation
        self.all_lons = self.lon_balloons_default[0,:]
        self.all_lons[self.all_lons<0] += 360.
        self.m_min = m_min
        self.r_venus = r_venus
        self.which_TL_distribution = which_TL_distribution
        self.default_p_uncertainty = default_p_uncertainty
        self.s_batch_volcanoes = s_batch_volcanoes
        self.verbose = verbose

        self._init_discretization()
        
        self.proba_all = np.zeros((self.SNR_thresholds.size, self.all_lats.size, self.all_lons.size)) # SNR x loc x lon
        self.rates_all = None

        l_ivolcano_start = np.arange(0, self.lon_volcanoes.size+self.s_batch_volcanoes, self.s_batch_volcanoes)
        for ilon, lon in tqdm(enumerate(self.all_lons), total=len(self.all_lons), disable=disable_tqdm, desc='Computing proba'): ## Loop over each longitude line

            lats_orig, lons_orig = self.all_lats, np.array([lon])
            lats, lons = np.meshgrid(lats_orig, lons_orig)
            lats, lons = lats.ravel(), lons.ravel()
            locations = np.arange(lats.size)

            rates = np.zeros((self.SNR_thresholds.size, lats.size))
            for ivolcano_start in l_ivolcano_start: ## Loop over batches of volcanoes

                idx_volcanoes = np.arange(ivolcano_start, ivolcano_start+self.s_batch_volcanoes)
                idx_volcanoes = idx_volcanoes[idx_volcanoes<self.lat_volcanoes.size]

                if idx_volcanoes.size == 0:
                    continue

                #print(ilon, lon, idx_volcanoes)
                lat_volcanoes, lon_volcanoes = self.lat_volcanoes[idx_volcanoes], self.lon_volcanoes[idx_volcanoes]
                TL_volcanoes = self.TL_array[:,ilon,idx_volcanoes]
                DISTS, MAG, DETECT_T = self.init_parameter_space(lat_volcanoes, lon_volcanoes, lats, lons, locations) # M0s x dists x SNR x loc

                F_MAGS = self.get_ratios_famp(idx_volcanoes) # M0s x dists x SNR x loc

                #raise ValueError("check debug")
                #print('shape_init', self.shape_init)
                #print('F_MAGS', F_MAGS.shape)

                self.m_min = self.M0s.min() ## TODO: Check this, why not using mmin?
                rates += self.compute_rate(DISTS, MAG, DETECT_T, F_MAGS, TL_volcanoes, idx_volcanoes)
                
            proba = self.compute_Poisson(rates) # SNR x loc

            self.proba_all[:,:,ilon] = proba
            #raise ValueError("check debug")

    @staticmethod
    def _save_dataframe_to_h5(group, name, df):
        sub = group.create_group(name)
        sub.attrs["columns"] = json.dumps(list(df.columns))

        for col in df.columns:
            values = df[col].to_numpy()

            if pd.api.types.is_bool_dtype(df[col]):
                ds = sub.create_dataset(col, data=values.astype(np.uint8))
                ds.attrs["dtype"] = "bool"

            elif pd.api.types.is_numeric_dtype(df[col]):
                sub.create_dataset(col, data=values)

            else:
                # h5py on older stacks does not like numpy unicode (<U...)
                arr = df[col].astype(str).fillna("").to_numpy()
                arr = arr.astype("S")  # store as fixed-width bytes
                ds = sub.create_dataset(col, data=arr)
                ds.attrs["dtype"] = "str"

    @staticmethod
    def _load_dataframe_from_h5(group, name):
        sub = group[name]
        columns = json.loads(sub.attrs["columns"])
        data = {}

        for col in columns:
            ds = sub[col]
            values = ds[()]
            dtype_tag = ds.attrs.get("dtype", None)

            if dtype_tag == "bool":
                values = values.astype(bool)

            elif dtype_tag == "str":
                # decode bytes -> python str
                if isinstance(values, np.ndarray):
                    values = np.array([
                        v.decode("utf-8") if isinstance(v, (bytes, np.bytes_)) else str(v)
                        for v in values
                    ], dtype=object)
                else:
                    values = values.decode("utf-8") if isinstance(values, (bytes, np.bytes_)) else str(values)

            data[col] = values

        return pd.DataFrame(data, columns=columns)

    def save_h5(self, filename, save_outputs=True):
        with h5py.File(filename, "w") as f:
            meta = f.create_group("meta")
            meta.attrs["class_name"] = self.__class__.__name__

            params = f.create_group("params")
            for key in [
                "max_dist_linearfit",
                "min_dist_threshold_linearfit",
                "dist_threshold_linearfit",
                "delta_dist",
                "perc_uncertainty",
                "T_eruption",
                "r_to_vent",
                "rho_surface",
                "rho_exit",
                "perc_rise_to_duration",
                "use_eruption_magnitudes",
            ]:
                if hasattr(self, key):
                    params.attrs[key] = getattr(self, key)

            if hasattr(self, "freq"):
                params.attrs["freq"] = self.freq

            data = f.create_group("data")
            self._save_dataframe_to_h5(data, "pd_volcanoes", self.pd_volcanoes)
            self._save_dataframe_to_h5(data, "pd_amplitudes", self.pd_amplitudes)

            arrays = f.create_group("arrays")
            if hasattr(self, "TL_array"):
                arrays.create_dataset("TL_array", data=self.TL_array)
            if hasattr(self, "lat_balloons_default"):
                arrays.create_dataset("lat_balloons_default", data=self.lat_balloons_default)
            if hasattr(self, "lon_balloons_default"):
                arrays.create_dataset("lon_balloons_default", data=self.lon_balloons_default)

            if save_outputs:
                outputs = f.create_group("outputs")
                for key in [
                    "proba_all",
                    "rates_all",
                    "M0s",
                    "SNR_thresholds",
                    "noise_level",
                    "duration",
                    "m_min",
                    "r_venus",
                    "which_TL_distribution",
                    "default_p_uncertainty",
                    "s_batch_volcanoes",
                    "verbose",
                    "all_lats",
                    "all_lons",
                ]:
                    if not hasattr(self, key):
                        continue

                    value = getattr(self, key)
                    if value is None:
                        outputs.attrs[f"{key}__is_none"] = True
                    elif np.isscalar(value):
                        if isinstance(value, str):
                            outputs.attrs[key] = value
                        else:
                            outputs.attrs[key] = value
                    else:
                        outputs.create_dataset(key, data=value)

    @classmethod
    def load_h5(cls, filename):
        with h5py.File(filename, "r") as f:
            params = dict(f["params"].attrs.items())

            pd_volcanoes = cls._load_dataframe_from_h5(f["data"], "pd_volcanoes")
            pd_amplitudes = cls._load_dataframe_from_h5(f["data"], "pd_amplitudes")

            TL_array = f["arrays"]["TL_array"][()]
            lat_balloons_default = f["arrays"]["lat_balloons_default"][()]
            lon_balloons_default = f["arrays"]["lon_balloons_default"][()]
            TL_array_provided = (TL_array, lat_balloons_default, lon_balloons_default)

            init_kwargs = dict(
                pd_volcanoes=pd_volcanoes,
                pd_amplitudes=pd_amplitudes,
                max_dist_linearfit=params["max_dist_linearfit"],
                min_dist_threshold_linearfit=params["min_dist_threshold_linearfit"],
                dist_threshold_linearfit=params["dist_threshold_linearfit"],
                delta_dist=params["delta_dist"],
                perc_uncertainty=params["perc_uncertainty"],
                T_eruption=params["T_eruption"],
                r_to_vent=params["r_to_vent"],
                rho_surface=params["rho_surface"],
                rho_exit=params["rho_exit"],
                perc_rise_to_duration=params["perc_rise_to_duration"],
                TL_array_provided=TL_array_provided,
                use_eruption_magnitudes=bool(params["use_eruption_magnitudes"]),
            )

            if "freq" in params:
                init_kwargs["freq"] = params["freq"]

            obj = cls(**init_kwargs)

            if "outputs" in f:
                outputs = f["outputs"]

                for key, value in outputs.attrs.items():
                    if key.endswith("__is_none"):
                        setattr(obj, key[:-8], None)
                    else:
                        setattr(obj, key, value)

                for key in outputs.keys():
                    setattr(obj, key, outputs[key][()])

        return obj

class proba_model_stf(proba_model):

    def __init__(self, pd_volcanoes, pd_amplitudes, max_dist_linearfit = 19000., min_dist_threshold_linearfit=100, dist_threshold_linearfit = 500., delta_dist = 1000., perc_uncertainty=1e-2, T_eruption=100., r_to_vent=1000., rho_surface=65., rho_exit=300., perc_rise_to_duration=0.05, TL_array_provided=None, use_eruption_magnitudes=False, freq=1.):
        
        super().__init__(pd_volcanoes, pd_amplitudes, max_dist_linearfit=max_dist_linearfit, min_dist_threshold_linearfit=min_dist_threshold_linearfit, dist_threshold_linearfit = dist_threshold_linearfit, delta_dist = delta_dist, perc_uncertainty=perc_uncertainty, T_eruption=T_eruption, r_to_vent=r_to_vent, rho_surface=rho_surface, rho_exit=rho_exit, perc_rise_to_duration=perc_rise_to_duration, TL_array_provided=TL_array_provided, use_eruption_magnitudes=use_eruption_magnitudes,)

        self.freq = freq

    def _VEI_to_pressure(self, VEI):

        volumes = 10**(4.+VEI)

        rise_time = self.perc_rise_to_duration*self.T_eruption

        #print(f'Original: {(1/(rise_time))}')
        #print(f'New: {(2./np.sqrt(1.+np.pi*self.freq*rise_time))}')

        return (1./(4*np.pi*self.r_to_vent))*(2./np.sqrt(1.+np.pi*self.freq*rise_time))*(volumes/(self.T_eruption+rise_time)) ## Assuming eruption magnitude

####################################
## BALLOON TRAJECTORY INTEGRATION ##
####################################

def plot_maps_all_trajectories(pd_final_probas, lons, lats, mission_durations, cmap_bounds=np.linspace(0., 0.8, 15)):

    LONS, LATS = np.meshgrid(lons, lats)

    cmap = cm.get_cmap("Reds", lut=len(cmap_bounds))
    norm = mcol.BoundaryNorm(cmap_bounds, cmap.N)
        
    fig = plt.figure(figsize=(10,10))
    grid = fig.add_gridspec(len(mission_durations), 3)
    iduration = -1
    for duration, one_duration in pd_final_probas.groupby(['duration']):
        iduration += 1
        isnr = -1
        for snr, one_snr in one_duration.groupby(['snr']):
            isnr +=1
            field = one_snr.proba.values.reshape(lons.size, lats.size)
            
            
            ax = fig.add_subplot(grid[iduration, isnr])
            m = Basemap(projection='robin', lon_0=0, ax=ax)
            m.drawmeridians(np.linspace(-180., 180., 5), labels=[0, 0, 0, 1], fontsize=10,)
            m.drawparallels(np.linspace(-90., 90., 5), labels=[1, 0, 0, 0], fontsize=10,)
            x, y = m(LONS.ravel(), LATS.ravel())
            x, y = x.reshape(LONS.shape), y.reshape(LONS.shape)
            sc = m.pcolormesh(x, y, field.T, cmap=cmap, norm=norm)
            #plt.colorbar(sc)
            plt.title(f'SNR {snr} - {duration:.0f} days')

    fmt = lambda x, pos: '{:.2f} %'.format(x*1e2) # 
    axins = inset_axes(ax, width="6%", height="250%", loc='lower left', bbox_to_anchor=(1.05, 0., 1, 1.), bbox_transform=ax.transAxes, borderpad=0)
    axins.tick_params(axis='both', which='both', labelbottom=False, labelleft=False, bottom=False, left=False)
    cbar = fig.colorbar(sc, format=FuncFormatter(fmt), cax=axins, orientation='vertical', extend='both', ticks=cmap_bounds[1:],)

def plot_proba_all_trajectories(pd_final_probas, mission_durations, xlim=[0, 1], ylim=[0, 40.]):

    fig = plt.figure(figsize=(14,3))
    grid = fig.add_gridspec(1, len(mission_durations))
    iduration = -1
    for duration, one_duration in pd_final_probas.groupby(['duration']):
        iduration += 1
        ax = fig.add_subplot(grid[0, iduration])
        ax.set_title(f'Mission {duration:.0f} days')
        for snr, one_snr in one_duration.groupby(['snr']):
            #values, bins = np.histogram(one_snr.proba, bins=20)
            #bin_centers = 0.5 * (bins[:-1] + bins[1:])
            #ax.bar(bin_centers, values/, width=(bins[1] - bins[0]), label=snr)
            ax.hist(one_snr.proba, bins=20, label=snr, density=True)
        ax.set_xlabel('Detection Probability')
        if iduration == 0:
            ax.legend(title='SNR', frameon=False)
            ax.set_ylabel('Probability Density Function')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

def compute_proba_one_trajectory(trajectory_in, snrs, lats, lons, probas, times_TL, snrs_selected=[1.,2.,5.], norm_factor_time=3600., disable_bar=False, kind='nearest', dropoff_hour_UST=0, earth_days_UST=116.75):

    trajectory = trajectory_in.copy()
    isnrs = [np.argmin(abs(snrs-snr)) for snr in snrs_selected]

    g = Geod(ellps='WGS84')

    print('- Building trajectory dt and bin_dt')

    ## Assign time bins to each entry of the trajectory
    trajectory['dt'] = trajectory.time/norm_factor_time # time in hours
    bins = np.arange(int(trajectory['dt'].max())+1)
    trajectory['bin_dt'] = np.searchsorted(bins, trajectory['dt'].values)

    print('- Building matrices')
    
    ## Find corresponding probabilities along trajectory
    LATS_STAT, LONS_STAT = np.meshgrid(lats, lons)
    LATS_STAT, LONS_STAT = LATS_STAT.ravel(), LONS_STAT.ravel()

    id_bins, id_map = np.meshgrid(np.arange(trajectory.shape[0]), np.arange(LATS_STAT.size))
    id_bins_shape = id_bins.shape
    id_bins, id_map = id_bins.ravel(), id_map.ravel()
    _, _, dists = g.inv(trajectory.lon.values[id_bins], trajectory.lat.values[id_bins], LONS_STAT[id_map], LATS_STAT[id_map])
    dists = dists.reshape(id_bins_shape)
    iclosest = dists.argmin(axis=0)
    
    f_traj_lat = interpolate.interp1d(trajectory.dt.values, trajectory.lat.values, bounds_error=False, fill_value=(trajectory.lat.values[0], trajectory.lat.values[-1]), kind=kind)
    f_traj_lon = interpolate.interp1d(trajectory.dt.values, trajectory.lon.values, bounds_error=False, fill_value=(trajectory.lon.values[0], trajectory.lon.values[-1]), kind=kind)

    print('- Looping over snr and time')

    new_trajectory = pd.DataFrame()
    for isnr, snr in tqdm(zip(isnrs, snrs_selected), total=len(isnrs), disable=disable_bar):

        
        """
        new_trajectory_loc = pd.DataFrame()
        if several_times_in_probas is None:
            probas_snr_ravelled = probas[isnr].ravel()[iclosest]
            f_proba = interpolate.interp1d(trajectory.dt.values, probas_snr_ravelled, bounds_error=False, fill_value=(probas_snr_ravelled[0], probas_snr_ravelled[-1]))

            times = np.arange(0., trajectory.dt.max()+1., 1.)
            new_trajectory_loc['time'] = times*norm_factor_time
            new_trajectory_loc['proba'] = 1.-np.cumprod(1.-f_proba(times))
            new_trajectory_loc['lat'] = f_traj_lat(times)
            new_trajectory_loc['lon'] = f_traj_lon(times)
            new_trajectory_loc['snr'] = snr
            new_trajectory = pd.concat([new_trajectory, new_trajectory_loc])

        else: ## assuming that "several_times_in_probas" contains the time stamps for each proba map
        """
        current_time = dropoff_hour_UST + trajectory.dt.values/earth_days_UST
        current_time_h = (current_time)%24
        icurrent_time = np.abs(current_time_h[:, None] - times_TL).argmin(axis=1)

        print(f'--- SNR: {snr}')
        #last_cumprod = 0.
        isection_time = -1
        for itime in range(times_TL.size):

            idx_all = np.where(icurrent_time==itime)[0]
            if idx_all.size == 0:
                continue

            probas_snr_ravelled = probas[itime][isnr].ravel()[iclosest]
            f_proba = interpolate.interp1d(trajectory.dt.values, probas_snr_ravelled, bounds_error=False, fill_value=(probas_snr_ravelled[0], probas_snr_ravelled[-1]))

            start_idxs = np.r_[0, np.where(np.diff(idx_all)>1)[0]+1, idx_all.size] # If the balloon wraps over all available local hours then you need to split between each time period

            print(f'itime {itime}')
            for start_idx, end_idx in zip(start_idxs[:-1], start_idxs[1:]):

                isection_time += 1
                idx = idx_all[start_idx:end_idx]
                current_time_h_earth = (current_time[idx]-dropoff_hour_UST)*earth_days_UST
                #print(current_time_h_earth.min(), current_time_h_earth.max(), np.diff(current_time_h_earth).max())
                times = np.arange(current_time_h_earth[0], current_time_h_earth[-1]+1., 1.)
                #print(snr, itime, current_time[idx], current_time_h_earth)
                #break
                
                new_trajectory_loc = pd.DataFrame()
                new_trajectory_loc['time'] = times*norm_factor_time
                #cumprods = (1.-np.cumprod(1.-f_proba(times))*(1.-last_cumprod))
                #last_cumprod = cumprods[-1]
                cumprods = 1.-np.cumprod(1.-f_proba(times))
                new_trajectory_loc['proba_loc'] = cumprods
                new_trajectory_loc['lat'] = f_traj_lat(times)
                new_trajectory_loc['lon'] = f_traj_lon(times)
                new_trajectory_loc['snr'] = snr
                new_trajectory_loc['time_ust'] = times_TL[itime]
                new_trajectory_loc['isection_time'] = isection_time
                new_trajectory = pd.concat([new_trajectory, new_trajectory_loc])   
            
        
    new_trajectory.sort_values(by=['snr','time'], inplace=True)
    new_trajectory.reset_index(drop=True, inplace=True)

    new_trajectory['proba'] = -1.
    for snr, new_trajectory_snr in new_trajectory.groupby('snr'):
        for isection_time in new_trajectory_snr.isection_time.unique():
            new_trajectory_isection_time = new_trajectory_snr.loc[new_trajectory_snr.isection_time==isection_time]
            ifirst_entry = new_trajectory_isection_time.iloc[0].name
            if new_trajectory_isection_time.iloc[0].time == 0:
                new_trajectory.loc[new_trajectory.index.isin(new_trajectory_isection_time.index), 'proba'] = new_trajectory.loc[new_trajectory.index.isin(new_trajectory_isection_time.index), 'proba_loc']
                continue
            prev_entry = new_trajectory.loc[new_trajectory.index==ifirst_entry-1].iloc[0]
            cumprod_last = 1.-prev_entry.proba
            new_trajectory.loc[new_trajectory.index.isin(new_trajectory_isection_time.index), 'proba'] = 1.-(1.-new_trajectory_isection_time.proba_loc)*cumprod_last
            
    return new_trajectory

import proba_volcanoes_modules as pvm 
def compute_multiple_trajectories_vectorized(mission_durations, interpolators, wind_direction_interpolator, wind_strength_interpolator, snrs_selected, inputs):

    icpu, LATS, LONS = inputs

    dt = 1*3600/6. # (in s) 600 s
    times = np.arange(0, np.max(mission_durations)*24*3600, dt) # mission duration given in days but should be in seconds at the end
    TIMES, ID_LAT0 = np.meshgrid(times, np.arange(LATS.size))
    shape_TIMES = TIMES.shape # balloon init loc/t0 x balloon flight time 
    TIMES, ID_LAT0 = TIMES.ravel(), ID_LAT0.ravel() 

    latitudes, longitudes = pvm.compute_positions_vectorized_w_interpolator(LATS[ID_LAT0].reshape(shape_TIMES), LONS[ID_LAT0].reshape(shape_TIMES), wind_direction_interpolator, wind_strength_interpolator, TIMES.reshape(shape_TIMES), None)
    longitudes[longitudes<0] += 360.

    pd_final_probas = pd.DataFrame()
    for snr, interpolator in tqdm(interpolators.items(), total=len(interpolators), disable=not (icpu==0)):
        
        probas_projected = interpolator.ev(longitudes, latitudes).reshape(shape_TIMES)
        probas_int = 1.-np.cumprod(1.-probas_projected.reshape(shape_TIMES)[:,::int(3600/dt)], axis=1)

        for mission_duration in mission_durations:
            iduration = np.argmin(abs(times[::int(3600/dt)] - mission_duration*24*3600.))
            pd_final_probas_loc = pd.DataFrame(np.c_[LATS, LONS, probas_int[:,iduration]], columns=['lat', 'lon', 'proba'])
            pd_final_probas_loc['snr'] = snr
            #new_trajectory_loc['duration'] = times[::int(3600/dt)][iduration]/(24*3600)
            pd_final_probas_loc['duration'] = mission_duration
            pd_final_probas = pd.concat([pd_final_probas, pd_final_probas_loc])

    return pd_final_probas

def compute_multiple_trajectories_allprobas_vectorized(mission_duration, interpolators, wind_direction_interpolator, wind_strength_interpolator, inputs):

    icpu, LATS, LONS = inputs

    dt = 1*3600/6. # (in s)
    times = np.arange(0, np.max(mission_duration)*24*3600, dt)  # Time vector from 0 to 3600 seconds in steps of 600 seconds
    TIMES, ID_LAT0 = np.meshgrid(times, np.arange(LATS.size))
    shape_TIMES = TIMES.shape # balloon init loc/t0 x balloon flight time 
    TIMES, ID_LAT0 = TIMES.ravel(), ID_LAT0.ravel() 

    ## Compute balloon trajectories
    latitudes, longitudes = pvm.compute_positions_vectorized_w_interpolator(LATS[ID_LAT0].reshape(shape_TIMES), LONS[ID_LAT0].reshape(shape_TIMES), wind_direction_interpolator, wind_strength_interpolator, TIMES.reshape(shape_TIMES), None)
    longitudes[longitudes<0] += 360.

    ## Integrate probabilities along balloon trajectories
    snrs = [snr for snr, _ in interpolators.items()]
    probas_ints = np.zeros((len(snrs), shape_TIMES[0], times[::int(3600/dt)].size))
    for isnr, (snr, interpolator) in tqdm(enumerate(interpolators.items()), total=len(interpolators), disable=not (icpu==0)):
        
        probas_projected = interpolator.ev(longitudes, latitudes).reshape(shape_TIMES)
        probas_ints[isnr,:,:] = 1.-np.cumprod(1.-probas_projected.reshape(shape_TIMES)[:,::int(3600/dt)], axis=1)
        #print(probas_ints.shape, probas_int.shape, shape_TIMES)
        #probas_ints[isnr,:,:] = probas_int

    return times[::int(3600/dt)], longitudes.reshape(shape_TIMES)[:,::int(3600/dt)], latitudes.reshape(shape_TIMES)[:,::int(3600/dt)], probas_ints

def compute_multiple_trajectories(snrs, lats, lons, probas, times_TL, winds, mission_durations, max_number_months, inputs, dropoff_hour_UST=1):

    icpu, LATS, LONS = inputs

    opt_trajectory = dict(
        time_max=3600*24*30*max_number_months,
        save_trajectory=False,
        folder = './data/',
    )
    
    pd_final_probas = pd.DataFrame()
    for lat, lon in tqdm(zip(LATS, LONS), total=LATS.size, disable=not icpu==0):
        start_location = [lat, lon] # lat, lon
        trajectory = VCD.compute_trajectory(winds, start_location, **opt_trajectory)
        new_trajectories = compute_proba_one_trajectory(trajectory, snrs, lats, lons, probas, times_TL, dropoff_hour_UST=dropoff_hour_UST, snrs_selected=[1.,2.,5.], norm_factor_time=3600., disable_bar=True) ## Venusquake
        
        for target_duration in mission_durations:
            
            days = new_trajectories.time/(3600*24)
            pd_final_proba = new_trajectories.loc[days<=target_duration,:].groupby('snr').last().reset_index()[['snr', 'proba']]
            pd_final_proba['lat'] = lat
            pd_final_proba['lon'] = lon
            pd_final_proba['duration'] = target_duration
            pd_final_probas = pd.concat([pd_final_probas, pd_final_proba])

    pd_final_probas.reset_index(drop=True, inplace=True)
    return pd_final_probas

from functools import partial
from multiprocessing import get_context
from scipy.interpolate import RectBivariateSpline
def compute_multiple_trajectories_vectorized_CPUs(proba_model, wind_direction_interpolator, wind_strength_interpolator, LATS, LONS, mission_durations, snrs_selected=[1.,2.,5.], nb_CPU=10):

    snrs = proba_model.SNR_thresholds
    lats, lons = proba_model.all_lats, proba_model.all_lons
    probas = proba_model.proba_all.copy() # SNR x lats x lons

    ## Probability interpolators
    interpolators = dict()
    for snr_target in snrs:
        proba_snr = probas[np.argmin(abs(snrs-snr_target)),:,:]
        interpolators[snr_target] = RectBivariateSpline(lons, lats, proba_snr, kx=1, ky=1)

    ## Wind field interpolators
    #wind_direction_interpolator, wind_strength_interpolator, _ = VCD.get_winds_interpolator(None, alt_balloon*1e3, winds=winds)

    ## Partial call of trajectory builder
    partial_compute_multiple_trajectories = partial(compute_multiple_trajectories_vectorized, mission_durations, interpolators, wind_direction_interpolator, wind_strength_interpolator, snrs_selected)

    nb_chunks = LATS.shape[0]
    idx_start_all = np.arange(nb_chunks)
    
    N = min(nb_CPU, nb_chunks)
    ## If one CPU requested, no need for deployment
    if N == 1:
        print('Running serial')
        pd_final_probas = partial_compute_multiple_trajectories((0, LATS, LONS))

    ## Otherwise, we pool the processes
    else:
    
        step_idx =  nb_chunks//N
        list_of_lists = []
        idxs = []
        for i in range(N):
            idx = np.arange(i*step_idx, (i+1)*step_idx)
            if i == N-1:
                idx = np.arange(i*step_idx, nb_chunks)
            idxs.append(idx_start_all[idx][0])
            list_of_lists.append( (i, LATS[idx_start_all[idx]], LONS[idx_start_all[idx]]) )

        with get_context("spawn").Pool(processes = N) as p:
            print(f'Running across {N} CPU')
            all_pd_final_probas = p.map(partial_compute_multiple_trajectories, list_of_lists)
            p.close()
            p.join()

        pd_final_probas = gpd.GeoDataFrame()
        for idx, pd_final_probas_loc in zip(idxs, all_pd_final_probas):
            #gdf_loc['iscenario'] += idx
            pd_final_probas = pd.concat([pd_final_probas, pd_final_probas_loc], ignore_index=True)
        pd_final_probas.reset_index(drop=True, inplace=True)

    return pd_final_probas

def compute_multiple_trajectories_CPUs(proba_model, winds, LATS, LONS, mission_durations, max_number_months=4, nb_CPU=10):

    snrs = proba_model.SNR_thresholds
    lats, lons = proba_model.all_lats, proba_model.all_lons
    probas = proba_model.proba_all.copy() # SNR x lats x lons

    partial_compute_multiple_trajectories = partial(compute_multiple_trajectories, snrs, lats, lons, probas, winds, mission_durations, max_number_months)
    #partial_compute_multiple_trajectories = partial(compute_multiple_trajectories, proba_model, winds, mission_durations, max_number_months)
    nb_chunks = LATS.shape[0]
    idx_start_all = np.arange(nb_chunks)
    
    N = min(nb_CPU, nb_chunks)
    ## If one CPU requested, no need for deployment
    if N == 1:
        print('Running serial')
        pd_final_probas = partial_compute_multiple_trajectories((0, LATS, LONS))

    ## Otherwise, we pool the processes
    else:
    
        step_idx =  nb_chunks//N
        list_of_lists = []
        idxs = []
        for i in range(N):
            idx = np.arange(i*step_idx, (i+1)*step_idx)
            if i == N-1:
                idx = np.arange(i*step_idx, nb_chunks)
            idxs.append(idx_start_all[idx][0])
            list_of_lists.append( (i, LATS[idx_start_all[idx]], LONS[idx_start_all[idx]]) )

        with get_context("spawn").Pool(processes = N) as p:
            print(f'Running across {N} CPU')
            all_pd_final_probas = p.map(partial_compute_multiple_trajectories, list_of_lists)
            p.close()
            p.join()

        pd_final_probas = gpd.GeoDataFrame()
        for idx, pd_final_probas_loc in zip(idxs, all_pd_final_probas):
            #gdf_loc['iscenario'] += idx
            pd_final_probas = pd.concat([pd_final_probas, pd_final_probas_loc], ignore_index=True)
        pd_final_probas.reset_index(drop=True, inplace=True)

    return pd_final_probas

#############################################
## VISUALIZATION PROBABILISTIC MODEL BELOW ##
#############################################

def draw_screen_poly(xy, ax, legend, color='red'):
    #x, y = m( lons, lats )
    #xy = zip(x,y)
    poly = Polygon_mpl( xy, facecolor=color, alpha=0.4, **legend )
    ax.add_patch(poly)

def extract_coords(gdf):
    points = []
    for geom in gdf.geometry:
        if geom.geom_type == 'LineString':
            points.extend(list(geom.coords))
    return points

def plot_regions(m, ax, VENUS, fontsize=None, use_active_corona=False, plot_lines=True, basedir='/staff/quentin/Documents/Projects/2024_Venus_Detectability/Venus_Detectability/data/'):

    if use_active_corona:
        active_corona = gpd.read_file(f"{basedir}/active_corona_shape/active_corona.shp").iloc[0].geometry

    color_dict = {'corona': 'tab:red', 'rift': 'tab:green', 'ridge': 'tab:blue', 'intraplate': 'white', 'wrinkles': 'tab:green'}
    for region in VENUS:

        if region == 'intraplate':
            continue

        print(f'Processing region {region}')
        
        if region == 'wrinkles':
            points = extract_coords(VENUS[region])
            points = np.array(points)
            x_bins = np.linspace(-180., 180., 300)
            y_bins = np.linspace(-90., 90., 150)
            density, x_edges, y_edges = np.histogram2d(points[:,0], points[:,1], bins=[x_bins, y_bins])
            x_edges, y_edges = np.meshgrid(x_edges, y_edges)
            shape_init = x_edges.shape
            x_edges, y_edges = x_edges.ravel(), y_edges.ravel()
            x_edges, y_edges = m(x_edges, y_edges)
            x_edges, y_edges = x_edges.reshape(shape_init), y_edges.reshape(shape_init)
            density[density>1] = 1
            m.pcolormesh(x_edges, y_edges, density.T, alpha=1, cmap='Greens',)
            continue

        for pol_in in tqdm(VENUS[region].explode(index_parts=False).reset_index(drop=True).geometry.values):

            pol = pol_in
            if pol.geom_type == 'LineString':
                pol = pol.buffer(0.01)

            ext_in = pol.exterior
            if ext_in is None:
                continue

            ext_all = [ext_in]
            if use_active_corona and region == 'corona':
                save = ext_all[0]
                ext_all = []
                for active_pol in active_corona.geoms:
                    ext_all_loc = active_pol.intersection(Polygon(save).buffer(0))
                    if ext_all_loc.geom_type == 'MultiPolygon':
                        ext_all_loc = [geo.exterior for geo in ext_all_loc.geoms]
                    else:
                        ext_all_loc = [ext_all_loc.exterior]
                    ext_all += ext_all_loc
            
            #print(m(-180., -40.), m(0., -40.), m(130., -40.), m(300., -40.), m(360., -40.))
            for ext in ext_all:

                surface2 = [m(lon,lat) for lon, lat in list(ext.coords)]
                if len(surface2) == 0:
                    continue
                
                clustering = DBSCAN(eps=500000, min_samples=5).fit(np.array(surface2))
                #lines = []
                for label in np.unique(clustering.labels_):
                    coords = np.array(surface2)[clustering.labels_==label]
                    legend = {}
                    draw_screen_poly(coords, ax, legend, color=color_dict[region])
                
    #if not region == 'wrinkles':
    if fontsize is None:
        fontsize = 10.
    change_label = {'wrinkles': 'Wrinkle Ridges', 'corona': 'Selected\ncoronae', 'ridge': 'Fold belts', 'rift': 'Rifts'}
    patches = [mpatches.Patch(facecolor=color_dict[region], label=change_label[region], alpha=0.5, edgecolor='black') for region in VENUS]
    if 'ridge' in VENUS: ## Add intraplate
        patches += [mpatches.Patch(facecolor='white', label='Intraplate', alpha=0.5, edgecolor='black')]
        bbox_to_anchor=(1.1, -0.1)
        ax.legend(handles=patches, frameon=False, bbox_to_anchor=bbox_to_anchor, ncol=4, bbox_transform=ax.transAxes, fontsize=fontsize-4., columnspacing=0.2)
    else:
        bbox_to_anchor=(0.7, -0.1)
        ax.legend(handles=patches, frameon=False, bbox_to_anchor=bbox_to_anchor, ncol=4, bbox_transform=ax.transAxes, fontsize=fontsize-4., columnspacing=0.2)
    if plot_lines:
        m.drawmeridians(np.linspace(-180., 180., 5), labels=[0, 0, 0, 1], fontsize=fontsize)
        m.drawparallels(np.linspace(-90., 90., 5), labels=[1, 0, 0, 0], fontsize=fontsize)
    
def interpolate_2d(current_map, lons_in, lats, toplot_in, dnew=1.):

    toplot = toplot_in.copy()
    lons = lons_in.copy()
    if lons_in.max() > 180.:
        lons[lons>=180.] -= 360.
    idx = np.argsort(lons)
    toplot = toplot[:,idx].T
    lons = lons[idx]

    #lons_new = np.arange(lons.min(), lons.max(), 2*dnew)
    lons_new = np.arange(-180., 180., 2*dnew)
    lats_new = np.arange(lats.min(), lats.max(), dnew)
    
    LAT, LON = np.meshgrid(lats_new, lons_new)
    shape = LON.shape
    x, y = current_map(LON.ravel(), LAT.ravel())
    x, y = x.reshape(shape), y.reshape(shape)
    
    #interpolator = RectBivariateSpline(lons, lats, toplot.T, kx=1, ky=1)
    interpolator = RectBivariateSpline(lons, lats, toplot, kx=1, ky=1)

    #print(lons_new.min(), lons_new.max(), lons.min(), lons.max())

    #LAT, LON = np.meshgrid(lats_new, lons_new)
    #shape_ = LON.shape
    
    toplot = interpolator.ev(LON.ravel(), LAT.ravel()).reshape(shape)
    #print(toplot.shape, LON.min(), LON.max(), LAT.min(), LAT.max(), x.min(), x.max(), y.min(), y.max())
    #plt.pcolormesh(y, x, toplot)


    return x, y, toplot, lons_new, lats_new

def one_map(ax, fig, proba, lats, lons, SNR_thresholds, snr, n_colors, show_title,  c_cbar = 'white', l_snr_to_plot=[], low_cmap=[], high_cmap=[], interpolate=True, proba_all_homo=None):
    
    m = Basemap(projection='robin', lon_0=0, ax=ax)
            
    LAT, LON = np.meshgrid(lats, lons)
    shape = LON.shape
    x, y = m(LON.ravel(), LAT.ravel())
    x, y = x.reshape(shape), y.reshape(shape)

    isnr = np.argmin(abs(SNR_thresholds-snr))
    toplot = proba[isnr,:]*1e2
    if proba_all_homo is not None:
        toplot /= proba_all_homo[isnr,0,0]*1e2
        toplot = 1e2*(1.-toplot)
    
    fmt = lambda x, pos: '{:.2f}'.format(x) # 
    if l_snr_to_plot:
        isnr = np.argmin(abs(SNR_thresholds-l_snr_to_plot[0]))
        toplot = proba[isnr,:]*1e2
        isnr = np.argmin(abs(SNR_thresholds-l_snr_to_plot[1]))
        toplot = 1e2*(1-(toplot - proba[isnr,:]*1e2)/toplot)
        fmt = lambda x, pos: '{:.2f}'.format(x) # 
        
    if show_title:
        if l_snr_to_plot:
            ax.set_title(f'$\%$ of probability decrease')
        else:
            ax.set_title(f'SNR {SNR_thresholds[isnr]:.0f}')
            
    
    if interpolate:
        x, y, toplot, lons_new, lats_new = interpolate_2d(m, lons, lats, toplot, dnew=1.)

    #print(low_cmap)
    if l_snr_to_plot or len(low_cmap)==0:
        cmap_bounds = np.linspace(toplot.min(), toplot.max(), n_colors)
        cmap = cm.get_cmap("Reds", lut=len(cmap_bounds))
    else:
        cmap_bounds = [0] + [c for c in low_cmap] + [c for c in high_cmap] # ISSI
    
        # Create low and high colormaps
        low_cmap_ = cm.get_cmap('Blues_r', len(low_cmap)+1)
        high_cmap_ = cm.get_cmap('Reds', len(high_cmap))

        # Combine the colors from both colormaps
        combined_colors = np.vstack((low_cmap_(np.linspace(0, 1, len(low_cmap)+1)), high_cmap_(np.linspace(0, 1, len(high_cmap)))))
        cmap = mcol.ListedColormap(combined_colors)

    norm = mcol.BoundaryNorm(cmap_bounds, cmap.N)
    
    sc = m.pcolormesh(x, y, toplot, alpha=1, cmap=cmap, norm=norm)
    m.drawmeridians(np.linspace(-180., 180., 5), labels=[0, 0, 0, 1], fontsize=12)
    m.drawparallels(np.linspace(-90., 90., 5), labels=[1, 0, 0, 0], fontsize=12)

    axins = inset_axes(ax, width="3%", height="100%", loc='lower left', bbox_to_anchor=(1.03, 0., 1, 1.), bbox_transform=ax.transAxes, borderpad=0)
    axins.tick_params(axis='both', which='both', labelbottom=False, labelleft=False, bottom=False, left=False)
    cbar = fig.colorbar(sc, format=FuncFormatter(fmt), cax=axins, orientation='vertical', extend='both', ticks=cmap_bounds[1:],)
   
    fontsize = 14.
    cbar.ax.tick_params(axis='both', colors=c_cbar, labelsize=fontsize)
    if not l_snr_to_plot:
        cbar.set_label('Detection probability (in %)', rotation=270, labelpad=15, color=c_cbar, fontsize=fontsize)

    return m
    
def plot_map(proba_model, VENUS, l_snr_to_plot=[], c_cbar='white', n_colors=20, show_title=True, low_cmap=[], high_cmap=[], proba_all_homo=None, plot_all_regions=False, plot_volcanoes=False, use_active_corona=False):
    
    lats = proba_model.all_lats
    lons = proba_model.all_lons
    SNR_thresholds = proba_model.SNR_thresholds
    proba = proba_model.proba_all

    #cmap_bounds = [0] + [c for c in np.arange(1.25e-2, 2.2e-2, 0.25e-2)] + [c for c in np.arange(1.2e-1, 1.5e-1, 0.05e-1)] # ISSI
    if not l_snr_to_plot:
        l_snr_to_plot = [1.]
    
    fig = plt.figure(figsize=(15,10))
    grid = fig.add_gridspec(2, len(l_snr_to_plot))
        
    for isnr_grid, snr in enumerate(l_snr_to_plot):
        
        ax = fig.add_subplot(grid[0, isnr_grid])
        m = one_map(ax, fig, proba, lats, lons, SNR_thresholds, snr, n_colors, show_title, c_cbar=c_cbar, low_cmap=low_cmap, high_cmap=high_cmap, proba_all_homo=proba_all_homo)
        if plot_volcanoes:
            x, y = m(proba_model.lon_volcanoes, proba_model.lat_volcanoes)
            m.scatter(x, y, marker='x', color='black', s=30)
        
        
    ax = fig.add_subplot(grid[1, 0])
    _ = one_map(ax, fig, proba, lats, lons, SNR_thresholds, snr, n_colors, show_title, c_cbar=c_cbar, l_snr_to_plot=l_snr_to_plot)
    
    if plot_all_regions and VENUS is not None:
        ax = fig.add_subplot(grid[1, 1])
        m = Basemap(projection='robin', lon_0=0, ax=ax)
        m.scatter(0., 0., latlon=True, s=0.1)
        plot_regions(m, ax, VENUS, use_active_corona=use_active_corona)
        
    fig.subplots_adjust(top=0.8, wspace=0.3)
    #fig.savefig('./test_data_Venus/map_probas.png', dpi=800., transparent=True)

def one_map_traj(fig, ax, lats, lons, new_trajectories_snr, VENUS, n_colors=10, c_cbar='white', fontsize=12., plot_time=False, alpha_traj=0.5, add_cbar=True):
    
    snr = new_trajectories_snr.snr.iloc[0]
    
    m = Basemap(projection='robin', lon_0=0, ax=ax)
            
    LAT, LON = np.meshgrid(lats, lons)
    shape = LON.shape
    x, y = m(LON.ravel(), LAT.ravel())
    x, y = x.reshape(shape), y.reshape(shape)

    fmt = lambda x, pos: '{:.0f}'.format(x) # 
    
    if plot_time:
        cmap_bounds = np.linspace(new_trajectories_snr.time.min()/(24*3600.), new_trajectories_snr.time.max()/(24*3600.), n_colors)
    else:
        cmap_bounds = np.linspace(new_trajectories_snr.proba.min()*1e2, new_trajectories_snr.proba.max()*1e2, n_colors)
    
    cmap = cm.get_cmap("viridis", lut=len(cmap_bounds))
    norm = mcol.BoundaryNorm(cmap_bounds, cmap.N)
    
    color_dict = {'corona': 'tab:red', 'rift': 'tab:green', 'ridge': 'tab:blue', 'intraplate': 'white'}
    if VENUS is not None:
        plot_regions(m, ax, VENUS, color_dict)
            
    #sc = m.pcolormesh(x, y, toplot, alpha=1, cmap=cmap, norm=norm)
    if plot_time:
        sc = m.scatter(new_trajectories_snr.lon, new_trajectories_snr.lat, c=new_trajectories_snr.time/(24*3600.), s=5, cmap=cmap, norm=norm, latlon=True, zorder=10, alpha=alpha_traj)
    else:
        sc = m.scatter(new_trajectories_snr.lon, new_trajectories_snr.lat, c=new_trajectories_snr.proba*1e2, s=5, cmap=cmap, norm=norm, latlon=True, zorder=10, alpha=alpha_traj)
    
    if add_cbar:
        axins = inset_axes(ax, width="80%", height="6%", loc='lower left', bbox_to_anchor=(0.1, -.2, 1, 1.), bbox_transform=ax.transAxes, borderpad=0)
        axins.tick_params(axis='both', which='both', labeltop=False, labelleft=False, top=False, left=False)
        cbar = fig.colorbar(sc, format=FuncFormatter(fmt), cax=axins, orientation='horizontal', extend='both', ticks=cmap_bounds[1:],)
        
        if plot_time:
            name_cbar = f'Time (days)'
        else:
            name_cbar = f'Detection probability (%) for SNR={snr:.1f}'
        cbar.set_label(name_cbar, rotation=0, labelpad=10, color=c_cbar, fontsize=fontsize)
        axins.xaxis.set_label_position('bottom')
        axins.xaxis.set_ticks_position('bottom')
        cbar.ax.tick_params(axis='both', colors=c_cbar, labelsize=fontsize-2., )

    if VENUS is not None:
        patches = [mpatches.Patch(facecolor=color_dict[region], label=region, alpha=0.5, edgecolor='black') for region in color_dict]
        ax.legend(handles=patches, frameon=False, bbox_to_anchor=(1., -0.1), columnspacing=0.5, handletextpad=0.25, ncol=4, bbox_transform=ax.transAxes, fontsize=fontsize, labelcolor=c_cbar)
    
    return m
    
def add_vertical_cbar(fig, ax, sc, cmap_bounds, fmt, c_cbar, name_cbar):

    axins = inset_axes(ax, width="3%", height="80%", loc='lower left', bbox_to_anchor=(1.03, 0.1, 1, 1.), bbox_transform=ax.transAxes, borderpad=0)
    axins.tick_params(axis='both', which='both', labelbottom=False, labelleft=False, bottom=False, left=False)
    cbar = fig.colorbar(sc, format=FuncFormatter(fmt), cax=axins, orientation='vertical', extend='both', ticks=cmap_bounds[:],)
    cbar.set_label(name_cbar, rotation=90, labelpad=10, color=c_cbar, fontsize=12)
    cbar.ax.tick_params(axis='both', colors=c_cbar, labelsize=12., )

def plot_trajectory(new_trajectories_total, proba_model, winds, VENUS=None, snr=1., n_colors=10, c_cbar='white', fontsize=15., ylim=[0., 20.], plot_time=False, plot_volcanoes=False, n_colors_proba = 15, n_colors_winds = 7, file='./figures/Figure_2_balloon_proba.pdf'):
    
    lats, lons = proba_model.all_lats, proba_model.all_lons
    
    fig = plt.figure(figsize=(14,8))
    grid = fig.add_gridspec(2, 6)
    s_maps = 3
    s_plots = 3
        
    ax = fig.add_subplot(grid[1, -s_maps:])
    ax_winds = fig.add_subplot(grid[0, -s_maps:])
    ax_vs_time = fig.add_subplot(grid[1, :s_plots])
    ax_vs_lon = fig.add_subplot(grid[0, :s_plots], sharex=ax_vs_time)
    
    iseismicity = -1
    linestyles = ['-', '--', ':']
    cmap = sns.color_palette('rocket', n_colors=new_trajectories_total.snr.unique().size,)
    lines_snr = []
    lines_seismicity = []
    for seismicity, new_trajectories in new_trajectories_total.groupby('seismicity'):
    
        iseismicity += 1
        if iseismicity == 0:
            new_trajectories_snr = new_trajectories.loc[new_trajectories.snr==snr]
            m = one_map_traj(fig, ax, lats, lons, new_trajectories_snr, VENUS, n_colors=n_colors, c_cbar=c_cbar, fontsize=fontsize, plot_time=plot_time)
            m_winds = one_map_traj(fig, ax_winds, lats, lons, new_trajectories_snr, None, n_colors=n_colors, c_cbar=c_cbar, fontsize=fontsize, plot_time=plot_time, add_cbar=False)
            
            if VENUS is None:
                
                fmt = lambda x, pos: '{:.4f} %'.format(x*1e2)
                idx_snr = np.argmin(abs(proba_model.SNR_thresholds-snr))
                x, y, toplot, _, _ = interpolate_2d(m, proba_model.all_lons, proba_model.all_lats, proba_model.proba_all[idx_snr,:,:], dnew=1.)
                cmap_bounds = np.linspace(toplot.min(), toplot.max(), n_colors_proba)
                cmap_p = cm.get_cmap("Reds", lut=len(cmap_bounds))
                norm = mcol.BoundaryNorm(cmap_bounds, cmap_p.N)
                sc_proba = m.pcolormesh(x, y, toplot, zorder=0, cmap=cmap_p, norm=norm)
                sc_proba.set_rasterized(True)
                m.drawmeridians(np.linspace(-180., 180., 5), labels=[0, 0, 0, 1], fontsize=12)
                m.drawparallels(np.linspace(-90., 90., 5), labels=[1, 0, 0, 0], fontsize=12)
                add_vertical_cbar(fig, ax, sc_proba, cmap_bounds, fmt, c_cbar, 'Hourly probability') 

                
                fmt = lambda x, pos: '{:.0f}'.format(x)
                unknown = 'wind_direction'
                vmax, vmin = -88, -92
                winds_grp = winds.groupby(['lat', 'lon']).first().reset_index()
                lat_size = winds_grp.lat.unique().size
                lon_size = winds_grp.lon.unique().size
                LON, LAT = np.meshgrid(winds_grp.lon.unique(), winds_grp.lat.unique())
                x, y = m_winds(LON.ravel(), LAT.ravel())
                x, y = x.reshape(lat_size, lon_size), y.reshape(lat_size, lon_size)
                cmap_bounds = np.linspace(vmin, vmax, n_colors_winds)
                cmap_w = cm.get_cmap("Greens", lut=len(cmap_bounds))
                norm = mcol.BoundaryNorm(cmap_bounds, cmap_w.N)
                sc_winds = m_winds.pcolormesh(x, y, winds_grp[unknown].values.reshape(lat_size, lon_size), norm=norm, cmap=cmap_w, alpha=0.8, zorder=5)
                sc_winds.set_rasterized(True)
                add_vertical_cbar(fig, ax_winds, sc_winds, cmap_bounds, fmt, c_cbar, 'Wind direction')    

                m_winds.drawmeridians(np.linspace(-180., 180., 5), labels=[0, 0, 0, 1], fontsize=12)
                m_winds.drawparallels(np.linspace(-90., 90., 5), labels=[1, 0, 0, 0], fontsize=12)
            
            if plot_volcanoes:
                x, y = m(proba_model.lon_volcanoes, proba_model.lat_volcanoes)
                m.scatter(x, y, marker='x', color='black', s=30)
                m_winds.scatter(x, y, marker='x', color='black', s=30)
            
        isnr = -1
        for snr, new_trajectories_snr in new_trajectories.groupby('snr'):
            isnr += 1
            line, = ax_vs_time.plot(new_trajectories_snr.time/(24*3600.), 1e2*new_trajectories_snr.proba, color=cmap[isnr], label=f'{snr:.0f}', linestyle=linestyles[iseismicity])
            if iseismicity == 0:
                lines_snr.append(line)
            line, = ax_vs_time.plot(new_trajectories_snr.time/(24*3600.), 1e2*new_trajectories_snr.proba, color=cmap[isnr], label=seismicity, linestyle=linestyles[iseismicity])
            if isnr == 0:
                lines_seismicity.append(line)
            
    ax_vs_lon.plot(new_trajectories_snr.time/(24*3600.), new_trajectories_snr.lon, label='longitude', color='black')
    ax_vs_lon.plot([0., 0.], [0., 0.], label='latitude', color='tab:red')
    ax_vs_lon.set_ylabel('Longitude', color=c_cbar, fontsize=fontsize)
    #ax_vs_lon.patch.set_alpha(0.0)
    #legend_ax_vs_lon = ax_vs_lon.legend(loc='upper left', labelcolor=c_cbar, fontsize=fontsize, frameon=True, facecolor='white', framealpha=0.8, edgecolor='none')
    #legend_ax_vs_lon.set_zorder(1000)
    ax_vs_lat = ax_vs_lon.twinx()  # instantiate a second Axes that shares the same x-axis
    #ax_vs_lat.set_zorder(ax_vs_lon.get_zorder() - 1)
    ax_vs_lat.plot(new_trajectories_snr.time/(24*3600.), new_trajectories_snr.lat, label='latitude', color='tab:red')
    ax_vs_lat.grid(alpha=0.4)
    ax_vs_lon.tick_params(axis='both', colors=c_cbar, labelsize=fontsize-2.)
    ax_vs_lat.tick_params(axis='both', colors=c_cbar, labelsize=fontsize-2.)
    ax_vs_lat.set_ylabel('Latitude', fontsize=fontsize, color='tab:red')
    ax_vs_lat.tick_params(axis='y', labelcolor='tab:red')

    handles, labels = ax_vs_lon.get_legend_handles_labels()
    ax_vs_lat.legend(handles, labels, loc='upper left', labelcolor=c_cbar, fontsize=fontsize-2., frameon=True, facecolor='white', framealpha=0.8, edgecolor='none')

    rect = plt.Rectangle(
        (0, 0), 1, 1,
        transform=ax_vs_lon.transAxes,  # Use axes coordinates
        color='white',
        zorder=-100,  # Place it below all other elements
    )
    ax_vs_lon.add_patch(rect)

    #ax_vs_time.legend(frameon=False, title='SNR')
    ax_vs_time_x = ax_vs_time.twinx()
    ax_vs_time_x.plot(new_trajectories_snr.time.iloc[:-1]/(24*3600.), 1e2*np.diff(new_trajectories_snr.proba), color='tab:blue')
    ax_vs_time_x.set_ylabel('Derivative $\partial_t\mathbb{P}$', color='tab:blue', fontsize=fontsize)
    #ax_vs_time_x.tick_params(axis='both', right=False, labelright=False)

    import matplotlib.ticker as ticker
    ax_vs_time_x.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1e'))
    ax_vs_time_x.tick_params(axis='both', labelcolor='tab:blue', labelsize=fontsize-2.)

    ax_vs_time.set_ylabel('Detection probability $\mathbb{P}$ (%)', color=c_cbar, fontsize=fontsize)
    ax_vs_time.set_xlabel('Time (days)', color=c_cbar, fontsize=fontsize)
    ax_vs_time.set_xlim([0., new_trajectories_snr.time.max()/(24*3600.)])
    ax_vs_time.set_ylim(ylim)
    ax_vs_time.grid(alpha=0.4)
    rect = plt.Rectangle(
        (0, 0), 1, 1,
        transform=ax_vs_time.transAxes,  # Use axes coordinates
        color='white',
        zorder=-100,  # Place it below all other elements
    )
    ax_vs_time.add_patch(rect)
    
    # Creating the first legend
    first_legend = ax_vs_time_x.legend(handles=lines_snr, loc='upper left', title='SNR', labelcolor=c_cbar, fontsize=fontsize-2., frameon=True, facecolor='white', framealpha=0.8, edgecolor='none')
    #ax_vs_time.legend(handles=first_legend.legendHandles, labels=[text.get_text() for text in first_legend.get_texts()], loc='upper left', title='SNR')

    # Creating and adding the second legend
    if VENUS is not None:
        second_legend = ax_vs_time.legend(handles=lines_seismicity, loc='upper left', title='Seismicity', bbox_to_anchor=(0.25, 1), frameon=False, labelcolor=c_cbar, fontsize=fontsize)
        #ax_vs_time.legend(handles=second_legend.legendHandles, bbox_to_anchor=(0.5, 1), labels=[text.get_text() for text in second_legend.get_texts()], loc='upper left', title='Seismicity')
        ax_vs_time.add_artist(first_legend)
        ax_vs_time.patch.set_alpha(0.5)
        plt.setp(second_legend.get_title(), color=c_cbar, fontsize=fontsize)
    
    ax_vs_time.tick_params(axis='both', colors=c_cbar, labelsize=fontsize-2.)
    plt.setp(first_legend.get_title(), color=c_cbar, fontsize=fontsize)
    
    fontsize_label = 20.
    ax_vs_lon.text(-0.07, 1., 'a)', fontsize=fontsize_label, ha='right', va='bottom', transform=ax_vs_lon.transAxes)
    ax_winds.text(-0.1, 1., 'b)', fontsize=fontsize_label, ha='right', va='bottom', transform=ax_winds.transAxes)
    ax_vs_time.text(-0.07, 1., 'c)', fontsize=fontsize_label, ha='right', va='bottom', transform=ax_vs_time.transAxes)
    ax.text(-0.1, 1., 'd)', fontsize=fontsize_label, ha='right', va='bottom', transform=ax.transAxes)
    
    fig.align_ylabels() 
    fig.subplots_adjust(wspace=1.9, hspace=0.3, bottom=0.2, top=0.8)
    fig.patch.set_alpha(0.)
    if file is not None:
        fig.savefig(file, transparent=True)

##########################
## TL specific routines ##
##########################
import time
def prepare_dataframe(TL_data, scaling_data, decimate_dist=10,):
    
    t0 = time.time()

    scaler = lambda TL: 20*np.log10(TL)
    
    lats = TL_data['lat'].values
    lons = TL_data['lon'].values
    azs = TL_data['az'].values
    dists = TL_data['rbin'].values
    idx = np.arange(0, dists.size, decimate_dist)
    receiver_height_km = TL_data['receiver_height_km'].values
    
    LONS, LATS, AZS, DISTS, HEIGHT = np.meshgrid(lons, lats, azs, dists[idx], receiver_height_km)

    #t1 = time.time()

    #print('t1=', t1-t0)

    scaling_vals =  scaling_data['sqrt_rho'].values.transpose(1, 0, 2)[:,:,None,None,:]
    
    TL_loc = TL_data['TL'].values[:,:,:,idx,:] * scaling_vals
    TL_loc /= TL_data['TL'].values[:,:,:,:,:].max(axis=3, keepdims=True)
    TL_loc = scaler(TL_loc)

    #t2 = time.time()

    #print('t2=', t2-t1)

    pd_amplitudes = pd.DataFrame(np.c_[LATS.ravel(), LONS.ravel(), AZS.ravel(), DISTS.ravel(), HEIGHT.ravel(), TL_loc.ravel()], columns=['lat', 'lon', 'az', 'dist', 'height', 'TL'])

    #t3 = time.time()

    #print('t3=', t3-t2)

    return pd_amplitudes

def get_fit():

    VEIs = np.arange(7)
    probas = np.array([9.6, 36.8, 37.3, 13.6, 2.3, 0.3, 0.1]) # Table 1 in 10.1029/2021JE007040
    
    def gaussian(x, mu, sigma):
        """A * exp(-((x - mu)^2) / (2 * sigma^2))"""
        return (1/(sigma*np.sqrt(2*np.pi))) * np.exp(-((x - mu)**2) / (2 * sigma**2))

    popt, _ = curve_fit(gaussian, VEIs, probas*1e-2,)
    #x = np.linspace(VEIs.min(), VEIs.max(), 100)
    mu_fit, sigma_fit = popt
    
    #p = gaussian(x, mu_fit, sigma_fit)
    #cdf_probas = np.cumsum(probas*1e-2)
    #cdf_p = 0.5*(1+erf((x-mu_fit)/(sigma_fit*np.sqrt(2))))

    return mu_fit, sigma_fit

def get_volcano_stats(file_volcanoes, type_v=None, n_eruptions_year=4.14*6, std_n_eruptions_year=2.13*6):

    mu_fit, sigma_fit = get_fit()

    pd_volcanoes = pd.read_csv(file_volcanoes, header=[0])

    pd_volcanoes['mean_gaussian'] = mu_fit
    pd_volcanoes['std_gaussian'] = sigma_fit
    pd_volcanoes['n_eruptions_year'] = n_eruptions_year
    pd_volcanoes['std_n_eruptions_year'] = std_n_eruptions_year
    if 'type_v' not in pd_volcanoes.columns and type_v is None: 
        pd_volcanoes['type_v'] = 'large'
    else:
        pd_volcanoes['type_v'] = type_v

    if not 'Height_km' in pd_volcanoes.columns: # For intermediate volcanoes > 5 km, we lack height information
        pd_volcanoes['Height_km'] = 0.

    if 'Ellipse_Max_Altitude_km' in pd_volcanoes.columns: # DEM derived
        pd_volcanoes['Height_km'] = pd_volcanoes['Ellipse_Max_Altitude_km']

    if 'Topo' in pd_volcanoes.columns: # Sophs
        pd_volcanoes['Height_km'] = pd_volcanoes['Topo']/1e3

    return pd_volcanoes

def compute_traj(time_max=3600*24*30*2, start_location=[-45.,0.], alt_balloon=60., file_atmos='/staff/quentin/Documents/Projects/2024_Venus_Detectability/Venus_Detectability/data/VCD_atmos_globe_new.dat'):

    altitude = alt_balloon*1e3
    winds = VCD.get_winds(file_atmos, altitude)

    opt_trajectory = dict(
        time_max=time_max,
        save_trajectory=False,
        folder = './data/',
    )
    trajectory = VCD.compute_trajectory(winds, start_location, **opt_trajectory)

    return trajectory

def compute_proba_trajectory(proba_model_allUST, times_TL, trajectory, dropoff_hour_UST):

    snrs = proba_model_allUST[0].SNR_thresholds
    lats, lons = proba_model_allUST[0].all_lats, proba_model_allUST[0].all_lons
    probas = [proba_model.proba_all.copy() for proba_model in proba_model_allUST] # SNR x lats x lons
    print('Running compute_proba_one_trajectory')
    new_trajectories = compute_proba_one_trajectory(trajectory, snrs, lats, lons, probas, times_TL, dropoff_hour_UST=dropoff_hour_UST, norm_factor_time=3600.)

    return new_trajectories

def get_all_probas(freqs, pattern_TL='balloon_tl_map_ust{}h_freq{}Hz_rangedep.nc', times_TL=np.arange(0,24), dropoff_hour_UST=1, folder_TL='/staff/quentin/Documents/Projects/2025_Sophus_MSc/data_TL/', files_volcanoes=['/staff/quentin/Documents/Projects/2024_Venus_Detectability/Venus_Detectability/data/volcanoes/05_large_greaterthan100.csv'], n_eruptions_year=4.14*6, std_n_eruptions_year=2.13*6, max_dist_linearfit = 19000., min_dist_threshold_linearfit=0., dist_threshold_linearfit = 500., delta_dist = 1000., perc_uncertainty=1e-2, T_eruptions=[100.], r_to_vent=1000., rho_surface=65., perc_rise_to_duration=0.05, use_eruption_magnitudes=False, use_stf_amplitudes=False, M0s = np.linspace(1., 7., 30), SNR_thresholds = np.linspace(0.1, 10., 50), noise_level = 1e-2, duration = 1./(365.*24.), m_min = 1., r_venus = 6052, which_TL_distribution='normal', s_batch_volcanoes=5, time_max=3600*24*30*2, start_locations=[[-45.,0.]], alt_balloon=60., file_atmos='/staff/quentin/Documents/Projects/2024_Venus_Detectability/Venus_Detectability/data/VCD_atmos_globe_new.dat', folder_save_proba_map=None, trajectory_df=None, run_name=None, skip_ivolcano=None, overwrite=False):

    opt_amplitudes = dict(
        max_dist_linearfit = max_dist_linearfit, 
        min_dist_threshold_linearfit = min_dist_threshold_linearfit, 
        dist_threshold_linearfit = dist_threshold_linearfit, 
        delta_dist = delta_dist, 
        perc_uncertainty=perc_uncertainty,
        r_to_vent=r_to_vent, 
        rho_surface=rho_surface, 
        perc_rise_to_duration=perc_rise_to_duration,
        use_eruption_magnitudes=use_eruption_magnitudes,
    )

    opt_model = dict(
        M0s = M0s, # Low discretization will lead to terrible not unit integrals
        SNR_thresholds = SNR_thresholds,
        noise_level = noise_level, # noise level in Pa
        duration = duration, # (1/mission_duration)
        m_min = m_min,
        r_venus = r_venus,
        which_TL_distribution=which_TL_distribution,
        s_batch_volcanoes=s_batch_volcanoes
    )

    trajectory = []
    if trajectory_df is None:
        for istart, start_location in enumerate(start_locations):
            opt_traj = dict(
                time_max=time_max, 
                start_location=start_location, 
                alt_balloon=alt_balloon, 
                file_atmos=file_atmos
            )
            trajectory.append( compute_traj(**opt_traj) )
    else:
        print('Overwriting drop off locations with trajectory_df DataFrame')
        start_locations = []
        for itraj, traj in trajectory_df.groupby('drop_off_idx'):
            lat, lon = traj.iloc[0].drop_off_lat, traj.iloc[0].drop_off_lon
            trajectory.append(traj)
            start_locations.append( [lat, lon] )

    """
    trajectory_df = pd.DataFrame();
    for itraj, traj in enumerate(trajectory): traj['drop_off_idx'] = itraj
    for traj in trajectory: traj['drop_off_lat'] = traj.iloc[0].lat
    for traj in trajectory: traj['drop_off_lon'] = traj.iloc[0].lon
    for traj in trajectory: trajectory_df = pd.concat([trajectory_df, traj])
    trajectory_df.to_csv('./trajectories/trajectory_paths.csv', header=True, index=False)
    """

    proba_models = dict()
    trajectories = pd.DataFrame()
    for freq in freqs:

        """
        import time
        t0 = time.time()
        """

        pd_amplitudes_allUST = []
        for time_TL in times_TL:
            #pattern_TL = 'balloon_tl_map_ust12h_freq{}Hz_rangedep.nc'
            pattern_loc = pattern_TL.format(f'{time_TL}', f'{freq:.3f}')
            file_TL = f'{folder_TL}{pattern_loc}'
            TL_data = xr.open_dataset(file_TL)
            file_scaling = f'{folder_TL}sqrt_rho.nc'
            scaling_data = xr.open_dataset(file_scaling)
            pd_amplitudes = prepare_dataframe(TL_data, scaling_data)
            pd_amplitudes_allUST.append( pd_amplitudes )

        """
        print(times_TL)
        print(len(pd_amplitudes_allUST))
        t1 = time.time()
        from pdb import set_trace as bp
        print(t1-t0)
        bp()
        """

        proba_models_teruption = dict()
        for T_eruption in T_eruptions:
            opt_amplitudes['T_eruption'] = T_eruption

            #TL_array_provided = None
            proba_models_teruption[T_eruption] = []
            for ivolcano, file_volcanoes in enumerate(files_volcanoes):

                #import time
                #t0 = time.time()
                if skip_ivolcano is not None:
                    if ivolcano in skip_ivolcano:
                        continue

                if run_name is not None:
                    import shutil
                    add_str = f'_{freq}Hz_{T_eruption}s_v{ivolcano}.csv'
                    run_name_loc = f'{run_name}{add_str}'

                    if os.path.exists(run_name_loc) and not overwrite:
                        continue

                pd_volcanoes = get_volcano_stats(file_volcanoes, n_eruptions_year=n_eruptions_year, std_n_eruptions_year=std_n_eruptions_year)
                
                probas_allUST = []
                cpt_amp = -1
                for pd_amplitudes in pd_amplitudes_allUST:
                    cpt_amp += 1

                    model_name = f'probas_{freq}Hz_{T_eruption}s_v{ivolcano}_{times_TL[cpt_amp]}h.h5'
                    #from pdb import set_trace as bp
                    #bp()
                    if folder_save_proba_map is not None and os.path.exists(f"{folder_save_proba_map}{model_name}"):
                        print(f'Loading {folder_save_proba_map}{model_name}')
                        probas = proba_model_stf.load_h5(f"{folder_save_proba_map}{model_name}")
                    else:
                        if use_stf_amplitudes:
                            probas = proba_model_stf(pd_volcanoes, pd_amplitudes, **opt_amplitudes, freq=freq)
                        else:
                            probas = proba_model(pd_volcanoes, pd_amplitudes, **opt_amplitudes,)
                        probas.compute_proba_map(**opt_model)
                        if folder_save_proba_map is not None:
                            probas.save_h5(f"{folder_save_proba_map}{model_name}")
                    probas_allUST.append(probas)

                for istart, start_location in enumerate(start_locations):
                    new_trajectories = compute_proba_trajectory(probas_allUST, times_TL, trajectory[istart], dropoff_hour_UST)
                    new_trajectories['freq'] = freq
                    new_trajectories['ivolcano'] = ivolcano
                    new_trajectories['T_eruption'] = T_eruption
                    new_trajectories['istart'] = istart
                    new_trajectories['start_lat'] = start_location[0]
                    new_trajectories['start_lon'] = start_location[1]
                    trajectories = pd.concat([trajectories, new_trajectories])

                #TL_array_provided = probas.TL_array, probas.lat_balloons_default, probas.lon_balloons_default 

                #t1 = time.time()
                #from pdb import set_trace as bp
                #print(t1-t0)
                if run_name is not None:
                    trajectories.to_csv(f'{run_name_loc}_temp', header=True, index=False)
                    shutil.move(f'{run_name_loc}_temp', f'{run_name_loc}')

                #bp()

                proba_models_teruption[T_eruption].append( probas_allUST )
        proba_models[freq] = proba_models_teruption
    
    trajectories.reset_index(drop=True, inplace=True)

    return proba_models, trajectories

import matplotlib.colors as colors
def plot_Figure_summary(trajectories, start_locations, n_months, snr_chosen, levels = np.arange(0.2, 1.05, 0.1), figsize=(8,4), file=None):

    alphabet = string.ascii_lowercase

    #start_locations = opt_traj['start_locations']
    #n_months = 2
    time_target = 3600*24*30*n_months
    diff = abs(trajectories['time']-time_target)
    trajectories_last = trajectories.loc[diff<1e4].groupby(['snr', 'freq', 'ivolcano', 'T_eruption', 'istart']).last().reset_index()

    fig = plt.figure(figsize=figsize)
    grid = fig.add_gridspec(trajectories_last.ivolcano.unique().size, trajectories_last.freq.unique().size)

    
    labels_volcanoes = ['all', '>4 km',]

    # Define discrete levels
    
    #levels = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9,1 .]
    cmap = plt.get_cmap('Reds', len(levels) - 1)
    norm = colors.BoundaryNorm(levels, cmap.N)

    T_eruptions = trajectories_last.T_eruption.unique()
    n_T_eruption = T_eruptions.size
    n_freqs = trajectories_last.freq.unique().size
    istarts = trajectories_last.istart.unique()
    lats = trajectories_last.start_lat.unique()
    n_istart = istarts.size
    lats_starts = [start_locations[istart][0] for istart in range(n_istart)]
    cpt_ivolc = -1
    for ivolcano, traj_volcano in trajectories_last.loc[trajectories_last.snr==snr_chosen].groupby('ivolcano'):
        ifreq = -1
        cpt_ivolc += 1
        for freq, traj_volc_freq in traj_volcano.groupby('freq'):
            ifreq += 1
            ax = fig.add_subplot(grid[cpt_ivolc, ifreq])
            if ifreq == 0:
                ax_first = ax

            #print(freq, ivolcano, traj_volc_freq.shape[0])
            #print(traj_volcano)

            proba_volc_freq = traj_volc_freq.proba.values.reshape((n_T_eruption, n_istart,))
            sc = ax.pcolormesh(lats_starts, T_eruptions, proba_volc_freq, cmap=cmap, norm=norm)
            ax.set_xticks(lats)

            #proba_volc_freq = traj_volc_freq.istart.values.reshape((n_T_eruption, n_istart,))
            #sc = ax.pcolormesh(lats_starts, T_eruptions, proba_volc_freq, cmap=cmap)
           
            ax.set_yscale('log')
            ax.set_ylim([1., 100.])
            ilabel = cpt_ivolc*n_freqs + ifreq
            ax.text(-0., 1.04, f'{alphabet[ilabel]})', fontsize=20., ha='right', va='bottom', transform=ax.transAxes)
            if cpt_ivolc == 0:
                ax.set_title(f'{freq:.2f} Hz')
            if cpt_ivolc == 1:
                ax.set_xlabel('Drop-off latitude')
        ax_first.set_ylabel(f'{labels_volcanoes[cpt_ivolc]}'+'\nEruption time (s)')
        
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    axins = inset_axes(ax, width="6%", height="150%", loc='lower left', bbox_to_anchor=(1.05, 0.25, 1, 1.), bbox_transform=ax.transAxes, borderpad=0)
    axins.tick_params(axis='both', which='both', labelbottom=False, labelleft=False, bottom=False, left=False)
    cbar = fig.colorbar(sc, cax=axins, orientation='vertical', extend='both',)
    cbar.set_label(f'Detection probability after {n_months} months', rotation=270, labelpad=10.5,)

    fig.subplots_adjust(left=0.1, hspace=0.4)

    if file is not None:
        #plt.savefig(f'./figures/summary_Figure_noscaling_{n_months}months.pdf')
        plt.savefig(file)

from matplotlib.colors import LogNorm
def plot_Figure_mass_rates(coef_rise_time=5e-2, rho_exit=300, radius=10, n_eruptions_year=24, T_chosen=10., file=None):

    alphabet = string.ascii_lowercase

    a_mu, b_mu = line_through_two_points(1., 1.5, 5., 5.) # in eruption mags, Figure 2 https://doi.org/10.1029/2022RG000783
    mu_vei = lambda VEI: a_mu*VEI + b_mu
    sigma_vei = lambda VEI: 0.35 + 0.*VEI # in eruption mags, Figure 2 https://doi.org/10.1029/2022RG000783

    def gaussian(x, mu, sigma):
        """A * exp(-((x - mu)^2) / (2 * sigma^2))"""
        return (1/(sigma*np.sqrt(2*np.pi))) * np.exp(-((x - mu)**2) / (2 * sigma**2))
    mu_fit, sigma_fit = get_fit()
    x = np.linspace(0, 7, 100)
    p = gaussian(x, mu_fit, sigma_fit)

    Mg = np.linspace(1., 7., 31)
    VEIs = np.linspace(1., 7., 30)
    m = 10**(Mg+4)
    T_durs = np.logspace(0, 2, 20)
    fs = np.logspace(-2, 0, 20)

    m_Mg, m_T_durs = np.meshgrid(Mg, T_durs)
    m_Mg_2, m_VEIs = np.meshgrid(Mg, VEIs)
    m_Mg_3, m_f = np.meshgrid(Mg, fs)
    m_t_rise = coef_rise_time*m_T_durs
    mass_rate_max = m/(m_T_durs+m_t_rise)
    exit_vel = mass_rate_max/((radius**2)*np.pi*rho_exit)
    p_peak = (1/(4*np.pi*1000.))*(10**(m_Mg_3+4)/(T_chosen))*(2./np.sqrt(1+np.pi*m_f*coef_rise_time*T_chosen))
    
    fig = plt.figure(figsize=(8,6))

    grid = fig.add_gridspec(2, 2)

    ax = fig.add_subplot(grid[0, 0])
    m_distrib_Mg = np.zeros_like(m_Mg_2)
    for iVEI, VEI in enumerate(VEIs):
        #print(gaussian(Mg, mu_vei, sigma_vei))
        m_distrib_Mg[iVEI,:] = gaussian(VEI, mu_fit, sigma_fit)*gaussian(Mg, mu_vei(VEI), sigma_vei(VEI))*n_eruptions_year
    sc = ax.pcolormesh(m_Mg_2, m_VEIs, m_distrib_Mg, cmap='Greens', norm=LogNorm(vmin=1e-2, vmax=5))
    ax.set_ylabel('VEI')
    plt.colorbar(sc)
    ax.set_title('Number of Eruptions')
    mg_1d = m_distrib_Mg.sum(axis=1)
    ax.plot(Mg.max()-mg_1d/mg_1d.max(), VEIs, color='black', lw=2)
    mg_1d = m_distrib_Mg.sum(axis=0)
    ax.plot(Mg, VEIs.max()-mg_1d/mg_1d.max(), color='black', ls='--', lw=2)
    ax.text(-0.2, 1.1, f'{alphabet[0]})', fontsize=20., ha='right', va='bottom', transform=ax.transAxes)
    
    ax = fig.add_subplot(grid[1, 0])
    sc = ax.pcolormesh(m_Mg, m_T_durs, mass_rate_max, norm=LogNorm(vmin=1e5, vmax=1e10), cmap='Blues')
    cs = ax.contour(m_Mg, m_T_durs, mass_rate_max, levels=[1e6, 1e7, 1e8, 1e9, 1e10], colors="black", linewidths=2)
    ax.clabel(cs, fmt=lambda v: f"{v:.0e}", fontsize=8)

    #mass_rate_realistic = mass_rate_max.copy()
    #mass_rate_realistic[(mass_rate_realistic<5e6)|(mass_rate_realistic>5e10)] = np.nan
    #ax.pcolormesh(m_Mg, m_T_durs, mass_rate_realistic, norm=LogNorm(vmin=1e5, vmax=1e10), cmap='Reds', alpha=0.5, label='Lefevere, 2025')
    #ax.legend()

    ax.set_yscale('log')
    #ax.set_xlabel('Eruption Magnitude')
    plt.colorbar(sc)
    str_model = f'$rho=${rho_exit} kg/m$^3$ r={radius} m'
    ax.set_title(f'Peak Mass Rate (kg/s)')
    ax.text(-0.2, 1.1, f'{alphabet[1]})', fontsize=20., ha='right', va='bottom', transform=ax.transAxes)
    ax.set_xlabel('Eruption Magnitude')
    ax.set_ylabel('Eruption Time (s)')
    
    ax = fig.add_subplot(grid[1, 1])
    sc = ax.pcolormesh(m_Mg, m_T_durs, exit_vel, norm=LogNorm(vmin=1e-1, vmax=1e4), cmap='Reds')
    cs = ax.contour(m_Mg, m_T_durs, exit_vel, levels=[1, 10, 100, 1000], colors="black", linewidths=2)
    #exit_vel_realistic = exit_vel.copy()
    #exit_vel_realistic[(exit_vel_realistic<100.)|(exit_vel_realistic>350.)] = np.nan
    #ax.pcolormesh(m_Mg, m_T_durs, exit_vel_realistic, norm=LogNorm(vmin=1e-1, vmax=1e4), cmap='Greys', alpha=0.5)
    ax.clabel(cs, fmt=lambda v: f"{v:.0e}", fontsize=8)

    plt.colorbar(sc)
    ax.set_yscale('log')
    #ax.set_xlabel('Eruption Magnitude')
    #ax.set_ylabel('Eruption Time (s)')
    ax.set_title(f'Exit Velocity (m/s)\n{str_model}')
    ax.text(-0.2, 1.1, f'{alphabet[2]})', fontsize=20., ha='right', va='bottom', transform=ax.transAxes)
    ax.set_xlabel('Eruption Magnitude')
    ax.set_ylabel('Eruption Time (s)')

    ax = fig.add_subplot(grid[0, 1])
    sc = ax.pcolormesh(m_Mg_3, m_f, p_peak, norm=LogNorm(), cmap='Oranges')
    cs = ax.contour(m_Mg_3, m_f, p_peak, levels=[10, 1e3, 1e4, 1e5], colors="black", linewidths=2)
    ax.clabel(cs, fmt=lambda v: f"{v:.0e}", fontsize=8)

    plt.colorbar(sc)
    ax.set_yscale('log')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title(f'Peak Pressure (Pa) with T$_{{dur}}$={T_chosen} s')
    ax.text(-0.2, 1.1, f'{alphabet[1]})', fontsize=20., ha='right', va='bottom', transform=ax.transAxes)

    fig.subplots_adjust(left=0.1, right=0.97, hspace=0.4, wspace=0.3)

    

    if file is not None: 
        print(file)
        fig.savefig(file)

def find_night_and_day(lst_hours, ref_lon = 0.0, subsolar_lat = 0.0, retrograde = True):

    # --------------------------------------------------
    # Convert local solar time -> subsolar longitude
    # --------------------------------------------------
    if retrograde:
        # Venus: local solar time increases westward
        subsolar_lon = ref_lon + 15.0 * (lst_hours - 12.0)
    else:
        # Earth-like case
        subsolar_lon = ref_lon + 15.0 * (12.0 - lst_hours)

    # wrap to [-180, 180]
    subsolar_lon = ((subsolar_lon + 180) % 360) - 180

    print("Subsolar longitude:", subsolar_lon)

    # --------------------------------------------------
    # Lat/lon grid
    # --------------------------------------------------
    lon = np.linspace(-180, 180, 721)
    lat = np.linspace(-90, 90, 361)
    LON, LAT = np.meshgrid(lon, lat)

    # radians
    lam = np.radians(LON)
    phi = np.radians(LAT)
    lam0 = np.radians(subsolar_lon)
    phi0 = np.radians(subsolar_lat)

    # --------------------------------------------------
    # Solar zenith angle cosine
    # --------------------------------------------------
    coschi = (
        np.sin(phi) * np.sin(phi0)
        + np.cos(phi) * np.cos(phi0) * np.cos(lam - lam0)
    )

    # Masks
    day = np.where(coschi > 0, 1.0, np.nan)
    night = np.where(coschi < 0, 1.0, np.nan)

    return LON, LAT, night, day

def plot_Figure_probas(proba_models, trajectories, files_volcanoes, lst=0, snr_chosen=2., other_snr_chosen=[5.], cmap='Reds', n_hours=1, vmax=2e-3, vmin=0, labels_volcanoes = ['>50 km', '5-100 km'], mapping_ax = {0.05: 0, 0.1: 1, 0.5: 2}, file=None, earth_days_UST=116.75):

    #snr_chosens = [1,2,5]
    #opt_volcanoes['files_volcanoes'][ivolcano]
    alphabet = string.ascii_lowercase
    SNR_thresholds = proba_models[0.05][0].SNR_thresholds#np.linspace(0.1, 10., 50)

    fig = plt.figure(figsize=(8,8))
    
    #for ii, snr_chosen in enumerate(snr_chosens):
    freqs_keys = [key for key in proba_models.keys()]
    n_freq = len(freqs_keys)
    n_volcanoes = len(proba_models[freqs_keys[0]])
    isnr = np.argmin(abs(SNR_thresholds-snr_chosen))
    grid = fig.add_gridspec(n_volcanoes+1+1, n_freq)
    offset_maps_proba = 1

    ifreq_count_base = -1
    ax_traj = None

    all_freqs = [key for key in proba_models.keys()]
    isorted_freq = np.argsort(all_freqs)

    c_type = {labels_volcanoes[0]:'gold', labels_volcanoes[1]:'tab:green'}
    s_type = {labels_volcanoes[0]:25, labels_volcanoes[1]:25}
    ivolcano_map = [0,2]
    for ivolcano_scatter in range(len(proba_models[all_freqs[0]])):

        type_v = labels_volcanoes[ivolcano_scatter]
        pd_volcanoes_type = get_volcano_stats(files_volcanoes[ivolcano_scatter], )
    
        ax = fig.add_subplot(grid[0, ivolcano_map[ivolcano_scatter]])
        if ivolcano_scatter == 0:
            ax_init = ax
        m = Basemap(projection='robin', lon_0=0, ax=ax)
        m.drawmeridians(np.linspace(-180., 180., 5), labels=[0, 0, 0, 1], fontsize=10,)
        m.drawparallels(np.linspace(-90., 90., 5), labels=[1, 0, 0, 0], fontsize=10,)
        m.scatter(pd_volcanoes_type.Lon_Center, pd_volcanoes_type.Lat_Center, marker='^', edgecolor=c_type[type_v], color=None, latlon=True, s=s_type[type_v], alpha=0.5, label=type_v)

    trajectory_loc = trajectories.loc[(trajectories.ivolcano==0)&(trajectories.freq==all_freqs[0])&(trajectories.snr==snr_chosen)]

    def forward(x):
        return (24*x/earth_days_UST)%24      # map x → new physical value

    def inverse(x):
        return [earth_days_UST*(x + ii*24)/24. for ii in range(10)]

    sc = m.scatter(trajectory_loc.lon, trajectory_loc.lat, c=forward(trajectory_loc.time/3600.), label='trajectory', s=5, latlon=True)

    axins = inset_axes(ax, width="50%", height="5%", loc='lower left', bbox_to_anchor=(0.25, 1.4, 1, 1.), bbox_transform=ax.transAxes, borderpad=0)
    cbar = fig.colorbar(sc, cax=axins, orientation='horizontal', )
    cbar.set_label(f'UST (h)', rotation=0, labelpad=-2, fontsize=11)

    bbox_to_anchor=(1., 0.75)
    ax_init.text(-0., 1., f'{alphabet[0]})', fontsize=20., ha='right', va='bottom', transform=ax_init.transAxes)
    ax_init.legend(frameon=False, bbox_to_anchor=bbox_to_anchor, ncol=2, bbox_transform=ax_init.transAxes, columnspacing=0.2, handletextpad=0.5)

    bbox_to_anchor=(2.2, 0.75)
    ax.text(-0., 1., f'{alphabet[1]})', fontsize=20., ha='right', va='bottom', transform=ax.transAxes)
    ax.legend(frameon=False, bbox_to_anchor=bbox_to_anchor, ncol=1, bbox_transform=ax_init.transAxes, columnspacing=0.2, handletextpad=0.5)

    LON, LAT, night, day = find_night_and_day(lst, ref_lon = 0.0, subsolar_lat = 0.0, retrograde = True)
    
    for ifreq, (freq_base, proba_model_volcanoes) in enumerate(proba_models.items()):
        
        #if freq == 0.25: ## Skip certain frequencies
        #    continue
            
        ifreq_count_base += 1
        ifreq_count = isorted_freq[ifreq_count_base]
        freq = all_freqs[ifreq_count_base]

        ax_opt = dict()
        if ax_traj is not None:
            ax_opt = dict(sharex=ax_traj, sharey=ax_traj)
        ax_traj = fig.add_subplot(grid[-1, mapping_ax[freq]], **ax_opt)
        ilabel = ifreq_count + 8
        ax_traj.text(-0., 1.1, f'{alphabet[ilabel]})', fontsize=20., ha='right', va='bottom', transform=ax_traj.transAxes)
        for ivolcano, proba_model in enumerate(proba_model_volcanoes):
            
            #print(ivolcano, ifreq_count, n_freq,)
            ax = fig.add_subplot(grid[ivolcano+offset_maps_proba, ifreq_count])
            ilabel = ivolcano*n_freq + ifreq_count + 2
            ax.text(-0., 1., f'{alphabet[ilabel]})', fontsize=20., ha='right', va='bottom', transform=ax.transAxes)
            if ivolcano == 0:
                if ifreq == 1:
                    ax.set_title(f'UST {lst} h\n{freq:.2f} Hz')
                else:
                    ax.set_title(f'{freq} Hz')
            if ifreq_count == 0:
                ax.set_ylabel(labels_volcanoes[ivolcano], labelpad=20)
            
            #proba_model = proba_model_volcanoes[0]
            proba = 1-(1.-proba_model.proba_all)**n_hours
            lons = proba_model.all_lons.copy()
            lons[lons>180] -= 360
            LONS, LATS = np.meshgrid(lons, proba_model.all_lats)
            idx = lons.argsort()
            LONS = LONS[:,idx]
            LATS = LATS[:,idx]

            proba_test = proba[isnr,:,idx].T
            m = Basemap(projection='robin', lon_0=0, ax=ax)
            m.drawmeridians(np.linspace(-180., 180., 5), labels=[0, 0, 0, 1], fontsize=10,)
            m.drawparallels(np.linspace(-90., 90., 5), labels=[1, 0, 0, 0], fontsize=10,)
            x, y = m(LONS.ravel(), LATS.ravel())
            x, y = x.reshape(LONS.shape), y.reshape(LONS.shape)
            sc = m.pcolormesh(x, y, proba_test, cmap=cmap, vmax=vmax, vmin=vmin)
            #sc = m.pcolormesh(x, y, proba_test, cmap=cmap,)

            x, y = m(LON, LAT)
            m.contourf(x, y, night, levels=[0.5, 1.5], colors=['navy'], alpha=0.25)

            display_traj_on_map = False
            if display_traj_on_map:
                if ifreq == 1 and ivolcano == 0:
                    trajectory_loc = trajectories.loc[(trajectories.ivolcano==ivolcano)&(trajectories.freq==freq)&(trajectories.snr==snr_chosen)]
                    m.scatter(trajectory_loc.lon, trajectory_loc.lat, c=trajectory_loc.time, label='trajectory', s=5, latlon=True)
                    ax.legend(frameon=False, bbox_to_anchor=bbox_to_anchor, ncol=2, bbox_transform=ax.transAxes, columnspacing=0.2)

            #c_type = dict(large='gold', intermediate='tab:green')
            #s_type = dict(large=25, intermediate=25)
            display_volcanoes_on_map = False
            if display_volcanoes_on_map:
                c_type = {labels_volcanoes[0]:'gold', labels_volcanoes[1]:'tab:green'}
                s_type = {labels_volcanoes[0]:25, labels_volcanoes[1]:25}
                if ifreq == 0 and ivolcano == 0:
                    for ivolcano_scatter in range(len(proba_model_volcanoes)):
                        type_v = labels_volcanoes[ivolcano_scatter]
                        pd_volcanoes_type = get_volcano_stats(files_volcanoes[ivolcano_scatter], )
                        m.scatter(pd_volcanoes_type.Lon_Center, pd_volcanoes_type.Lat_Center, marker='^', edgecolor=c_type[type_v], color=None, latlon=True, s=s_type[type_v], alpha=0.5, label=type_v)

                    if ivolcano == 0:
                        #ax.legend(frameon=False, ncol=2)
                        bbox_to_anchor=(1., -0.1)
                        ax.legend(frameon=False, bbox_to_anchor=bbox_to_anchor, ncol=2, bbox_transform=ax.transAxes, columnspacing=0.2, handletextpad=0.5)

            c_type = {0:'gold', 1:'tab:green'}
            ls_snr = ['-', ':', '.-']
            #ivolcano_selected =
            for isnr_loc, snr in enumerate([snr_chosen,] + other_snr_chosen):
                trajectory_loc = trajectories.loc[(trajectories.ivolcano==ivolcano)&(trajectories.freq==freq)&(trajectories.snr==snr)]
                trajectory_loc.sort_values(by='time', inplace=True)
                ax_traj.plot(trajectory_loc.time/(3600*24), trajectory_loc.proba, color=c_type[ivolcano], label=f'{labels_volcanoes[ivolcano]} - SNR {snr:.0f}', ls=ls_snr[isnr_loc], lw=4.)
            ax_traj.set_xlabel('Time (days)')
            ax_traj.grid(alpha=0.5)
            ax_traj.set_ylim([0., 1.])

            if ifreq_count == 0:
                ax_traj.set_ylabel('Detection Probability')
                bbox_to_anchor=(0., -0.6)
                ax_traj.legend(frameon=False, loc="lower left", bbox_to_anchor=bbox_to_anchor, ncol=4, bbox_transform=ax_traj.transAxes, columnspacing=1, handletextpad=0.5)
            else:
                ax_traj.tick_params(axis='both', which='both', labelleft=False)
            
            ax_traj_lst = ax_traj.twiny()
            ticks = ax_traj.get_xticks()
            ax_traj_lst.set_xticks(ticks)
            ax_traj_lst.set_xticklabels([f"{forward(t):.0f}" for t in ticks])
            ax_traj_lst.set_xlim(ax_traj.get_xlim())
            ax_traj_lst.set_xlabel('UST (h)')
            for lst_plot in inverse(lst):
                ax_traj_lst.axvline(lst_plot, color='black', ls=':', alpha=0.5)
                
            if ifreq_count == n_freq-1 and ivolcano == 0:
                fmt = lambda x, pos: '{:.1f} %'.format(x*1e2) # 
                axins = inset_axes(ax, width="50%", height="5%", loc='lower left', bbox_to_anchor=(0.25, -0.32, 1, 1.), bbox_transform=ax.transAxes, borderpad=0)
                cbar = fig.colorbar(sc, format=FuncFormatter(fmt), cax=axins, orientation='horizontal', )
                cbar.set_label(f'Hourly proba.', rotation=0, labelpad=2.5, fontsize=11)

                
            
    fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.95)
    if file is not None: 
        print(file)
        fig.savefig(file)

##########################
if __name__ == '__main__':

    """
    PATH_VENUS_DATA = os.path.join("../../../Venus_data/")
    PATH_VENUS = os.path.join(f"{PATH_VENUS_DATA}tectonic_settings_Venus")
    VENUS = {
        'corona': gpd.read_file(f"{PATH_VENUS}/corona.shp"),
        'rift': gpd.read_file(f"{PATH_VENUS}/rifts.shp"),
        'ridge': gpd.read_file(f"{PATH_VENUS}/ridges.shp")
    }

    ## Below to create surface ratios
    output_file = './test_data_Venus/surface_ratios.csv'
    l_lon = np.arange(-179, -150, 1)
    l_lat = np.arange(-89, 90, 1)
    compute_ratios(VENUS, l_lon, l_lat, output_file, ratio_df=pd.DataFrame())
    """

    file_slopes = '../../../Venus_data/distribution_venus_per_mw.csv'
    pd_slopes = get_slopes(file_slopes)

    file_curve = './test_data_Venus/GF_reverse_fault_1Hz_c15km.csv'
    TL_new, TL_new_qmin, TL_new_qmax = get_TL_curves(file_curve, dist_min = 100., plot=False)

    file_ratio = './test_data_Venus/surface_ratios_fixed.csv'
    surface_ratios = get_surface_ratios(file_ratio)

    dlat = 5.
    r_venus = 6052
    opt_model = dict(
        scenario = 'active_high_min', # Iris' seismicity scenario
        dists = np.arange(10., np.pi*r_venus, 200), # Low discretization will lead to terrible not unit integrals
        M0s = np.linspace(3., 8., 30), # Low discretization will lead to terrible not unit integrals
        SNR_thresholds = np.linspace(0.1, 10., 50),
        noise_level = 5e-2, # noise level in Pa
        duration = 1./(365.*24), # (1/mission_duration)
        all_lats = np.arange(-90., 90.+dlat, dlat),
        all_lons = np.arange(-180, 180+dlat*2, dlat*2),
        homogeneous_ratios = False,
        m_min = 3.,
        r_venus = r_venus,
        
    )

    proba_all_other_high = compute_proba_map(pd_slopes, surface_ratios, TL_new, TL_new_qmin, TL_new_qmax, **opt_model)

    ## Visualization
    # low_cmap, high_cmap = np.arange(1e-2, 5e-2, 5e-3), np.arange(5e-2, 1.25e-1, 1e-2) # 1 hour
    low_cmap, high_cmap = np.arange(5e-1, 1, 1e-1), np.arange(2, 3, 0.25) # 1 day RW
    low_cmap, high_cmap = np.arange(2.5e-1, 1, 1e-1), np.arange(2, 3, 0.25) # 1 day RW low activity
    #low_cmap, high_cmap = np.arange(3e-1, 1, 1e-1), np.arange(1, 3, 0.25) # 1 day body

    plot_map(all_lats, all_lons, proba_all_other_high, SNR_thresholds, VENUS, c_cbar='black', l_snr_to_plot=[1.,5.], n_colors=10, low_cmap=low_cmap, high_cmap=high_cmap,)#low_cmap, high_cmap = np.arange(20, 60, 10), np.arange(60, 80, 10) # 1 day ratio over homogeneous
    #plot_map(all_lats, all_lons, proba_all_other, SNR_thresholds, duration, VENUS, l_snr_to_plot=[1.,5.], n_colors=10, low_cmap=low_cmap, high_cmap=high_cmap, proba_all_homo=proba_all_homo)


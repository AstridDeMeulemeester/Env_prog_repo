import os
import pandas as pd
import numpy as np
import scipy
import scipy.stats as stats
import random
import xarray as xr
import matplotlib.pyplot as plt
import glob
import rasterio
import rasterio.features
import rasterio.warp
import netCDF4 as netcdf
from PET_estimation import ET0
from rasterio.enums import Resampling
from rasterio.merge import merge
from tqdm import tqdm
from scipy.stats import beta, gamma, gumbel_r, gumbel_l, lognorm, logistic, norm, weibull_min 
from scipy.stats import norm,genextreme,genlogistic,pearson3,genpareto,exponnorm,kappa3
from Prog_task import *
from tqdm.contrib.itertools import product

from Drought_index_characterization import *

distribution_dict_spi = {"beta": beta,
                         "gamma": gamma,
                         "gumbel_r":gumbel_r,
                         "gumbel_l": gumbel_l,
                         "lognorm": lognorm,
                         "logistic": logistic,
                         "norm": norm,
                         "weibull_min":weibull_min
                         }
distribution_dict_spei = {"norm": norm,
                          "genextreme": genextreme,
                          "genlogistic": genlogistic,
                          "pearson": pearson3,
                          "genpareto": genpareto,
                          "exponnorm": exponnorm,
                          "kappa3": kappa3
                         }




### RUN THESE LINES:

TIMESTEPS = [1, 3, 6, 9, 12, 24]

DEBUG=True


dataset = load_dataset("./Data/ssp585", gcms=["ukesm1"], save_rescaled_datasets=True, rescaled_dir="./output")




for var in VARIABLES:
    plot_annual_average(dataset, var, dstdir="./output")

rescale_and_save_dem_file("./DEM/DEM_Ethiopia/DEM_Ethiopia.tif", "./output/dem_ethiopia_rescaled.tif")

with rasterio.open("./output/dem_ethiopia_rescaled.tif") as fdem:
    dem = np.array(fdem.read())
print(dem.shape)
compute_PET(dataset, dem, "./output")
compute_water_balance(dataset, "./output")

print("Calculating SPI")
calculate_spi_dataset(dataset, [norm], TIMESTEPS)



# dataset = load_dataset(["tasmin"]) #CHOOSE THE VARIABLE TO BE PLOTTED
# plot_mean_value(dataset, "tasmin") #CHOOSE THE VARIABLE TO BE PLOTTED
# plt.show()

# make_dem_mosaic()
# open_rescaled_dem_file() 
# monthly_average_precipitation()
# split_monthly_average_pr()
# #load_monthly_precipitation_for_era()
# #calculate_spi()
# compute_and_save_rejection_frequency_for_SPI(distribution_dict_spi)       
# load_and_draw_figures_for_distribution_choice_SPI(distribution_dict_spi)
# compute_PET()
# monthly_average_water_balance()
# split_monthly_water_balance()
# #calculate_spei_for_dataset()
# compute_and_save_rejection_frequency_for_SPEI(distribution_dict_spei)       
# load_and_draw_figures_for_distribution_choice_SPEI(distribution_dict_spei)
# #estimate_spei()
# #estimate_spei_for_dataset()
# split_monthly_water_balance_by_season()
# #calculate_drought_characteristics()
# #plot_drought_difference()
# #classify_drought()
# #count_drought_pixels()


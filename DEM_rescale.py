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
from scipy.stats import norm, genextreme, genlogistic, pearson3, genpareto, exponnorm, kappa3

new_directory = "C:/Users/farah/Desktop/Masters/environmental_programming"
os.chdir(new_directory)

parent_dir = os.getcwd()

# Path to the clipped NetCDF data
path_to_data = fr"{parent_dir}\DEM_clipped"

def rescale_and_save_dem_files():
    subdirectories = ['DEM_clipped']
    dem_folder = os.path.join(os.getcwd(), *subdirectories)
    parent_dir = os.path.dirname(dem_folder)
    output_folder = 'DEM_Ethiopia_reassembledV3'
    DEM_Ethiopia_reassembledV3 = os.path.join(parent_dir, output_folder)
    
    # Create the output folder if it doesn't exist
    os.makedirs(DEM_Ethiopia_reassembledV3, exist_ok=True)
    
    # Get a list of all DEM files in the input folder
    dem_files = glob.glob(os.path.join(dem_folder, '*.tif'))
    for dem_file in dem_files:
        output_dem_file = os.path.join(DEM_Ethiopia_reassembledV3, os.path.basename(dem_file))
        with rasterio.open(dem_file) as src:
            
            # Calculate the resampled dimensions based on the target resolution
            target_resolution = 0.5  # this is in degrees!!!
            resampled_height = int(src.height * src.res[0] / target_resolution)
            resampled_width = int(src.width * src.res[0] / target_resolution)
            
            # Perform the resampling using bilinear interpolation
            resampled_data = src.read(
                out_shape=(src.count, resampled_height, resampled_width),
                resampling=Resampling.bilinear )
            
            #transform = rasterio.Affine(scale_x, rotation_x, translation_x, scale_y, rotation_y, translation_y)
            transform = rasterio.Affine(target_resolution, 0.0, src.bounds.left, 0.0, -target_resolution, src.bounds.top)
            
            # Update metadata for the resampled DEM
            resampled_metadata = {
                'driver': 'GTiff',
                'count': 1,
                'dtype': src.dtypes[0],
                'height': resampled_height,
                'width': resampled_width,
                'transform': transform }
            
            with rasterio.open(output_dem_file, 'w', **resampled_metadata) as dst:
                if not np.isnan(np.sum(resampled_data)):
                    dst.write(resampled_data[0, :, :], 1)
                    print(f"File {output_dem_file} successfully written.")
                else:
                    print(f"Warning: File {output_dem_file} has empty or NaN values and won't be written.")
                    
rescale_and_save_dem_files('Clipped_Eth_DEM.tif')
    

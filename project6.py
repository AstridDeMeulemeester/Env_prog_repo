import os
import netCDF4
import h5netcdf
from matplotlib import pyplot as plt

import xarray as xr

# Define the folder containing the NetCDF files
#folder_path = 'C:/Users/farah/Desktop/Masters/environmental_programming/ssp585/ssp585/CanESM5'
folder_path = "C:/Users/astri/Documents/KUL/Eerste Master/Environmental Programming/AA_Taak/ssp585"

models = [
    "canesm5_r1i1p1f1_ssp585", 
    "cnrm-cm6-1_r1i1p1f2_ssp585", 
    "cnrm-esm2-1_r1i1p1f2_ssp585", 
    "ec-earth3_r1i1p1f1_ssp585", 
    "gfdl-esm4_r1i1p1f1_ssp585",
    "ipsl-cm6a-lr_r1i1p1f1_ssp585",
    "miroc6_r1i1p1f1_ssp585",
    "mpi-esm1-2-hr_r1i1p1f1_ssp585",
    "mri-esm2-0_r1i1p1f1_ssp585",
    "ukesm1-0-ll_r1i1p1f2_ssp585"]

variables = [
    "hurs",
    "pr",
    "rlds",
    "rsds",
    "sfcwind",
    "tasmax",
    "tasmin"
]

datasets = {}
titels = {
    "hurs": "Annual Average Relative Humidity",
    "pr": "Annual Average Percipitation",
    "rlds": "Annual Average Longwave Radiation",
    "rsds": "Annual Average Shortwave Radiation",
    "sfcwind": "Annual Average Wind Speed",
    "tasmax": "Annual Average Maximum Temperature",
    "tasmin": "Annual Average Minimum Temperature",
    }

variables = ["hurs", "pr", "rlds", "rsds", "sfcwind", "tasmax", "tasmin"]


def plot_var_ssb(var: str):
    # Calculate the average variable over the 10 different GCM's
    yearly_vdat = []
    for model in models:
        ds_name = f"{model}_{var}_daily_1971_2100.nc"
        ds = datasets[ds_name]
        vdat = ds[var]

        vdat = vdat.resample(time='YE').sum()  # Resample on yearly frequency
        vdat = vdat.mean(dim=['time'])         # Average across all years

        yearly_vdat.append(vdat)

    res = xr.concat(yearly_vdat, dim="model")
    res = res.mean(dim="model")

    
    
    plt.figure()
    res.plot()
    plt.title(titels[var])
    plt.ylabel('Lattitude')
    plt.xlabel('Longitude')
    plt.savefig(f'{var}.png')
 

# Iterate through each NetCDF file
for root, subdirs, files in os.walk(folder_path):
    for file in files:
        if file.endswith(".nc"):
            file_path = os.path.join(root, file)

            print(f"Processing file: {file}")
            data = xr.open_dataset(file_path)

            # Save the dataset in the dictionary
            datasets[file] = data
    
print("Datasets loaded:", datasets.keys())

for var in variables:
    plot_var_ssb(var)



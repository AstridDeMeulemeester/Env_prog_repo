# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 15:53:29 2024

@author: farah
"""

import os
import xarray as xr

# Define the folder containing the NetCDF files
folder_path = 'C:/Users/farah/Desktop/Masters/environmental_programming/ssp585/ssp585/CanESM5_clipped'

# List all files in the folder
nc_files = [f for f in os.listdir(folder_path) if f.endswith('.nc')]
datasets = {}

# Iterate through each NetCDF file
for nc_file in nc_files:
    file_path = os.path.join(folder_path, nc_file)
    print(f"Processing file: {nc_file}")
    
    # Open the dataset with a context manager
    with xr.open_dataset(file_path) as data:
        # Load data into memory and save in the dictionary
        datasets[nc_file] = data.load()

print("Datasets loaded:", datasets.keys())


import matplotlib.pyplot as plt

# Output folder for plots
plot_output_folder = "C:/Users/farah/Desktop/Masters/environmental_programming/plots"
os.makedirs(plot_output_folder, exist_ok=True)

# Process each dataset in the dictionary
for file_name, ds in datasets.items():
    print(f"Processing file: {file_name}")
    
    # Identify the main variable in the dataset (skip 'spatial_ref')
    variable = [var for var in ds.data_vars if var != 'spatial_ref'][0]
    print(f"Found variable: {variable}")
    
    # Step 1: Compute annual averages
    annual_avg = ds[variable].groupby('time.year').mean('time')
    print(f"Computed annual averages for variable: {variable}")
    
    # Step 2: Plot spatial maps for each year
    for year in annual_avg['year']:
        annual_avg_year = annual_avg.sel(year=year)
        
        # Generate the plot
        plt.figure(figsize=(10, 8))
        annual_avg_year.plot(
            cmap='viridis',  # Colormap
            cbar_kwargs={'label': f'Annual Average {variable} (units)'}
        )
        plt.title(f"{variable} Annual Average for {int(year.values)}", fontsize=18)
        plt.xlabel("Longitude", fontsize=14)
        plt.ylabel("Latitude", fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        
        # Save the plot
        plot_file_name = f"{os.path.splitext(file_name)[0]}_{variable}_{int(year.values)}.png"
        plot_path = os.path.join(plot_output_folder, plot_file_name)
        print(f"Saving plot: {plot_path}")
        plt.savefig(plot_path, dpi=300)
        plt.close()  # Close the figure to save memory

print("Task 2 completed!")


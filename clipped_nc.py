# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 14:28:02 2024

@author: limai
"""

import os
import geopandas as gpd
import xarray as xr
import rioxarray

ethiopia_shapefile = "C:/Users/limai/Desktop/KU LEUVEN/02_Environmental Programming/Assignment/Env_prog_repo/world-administrative-boundaries/world-administrative-boundaries.shp"
ethiopia_shape = gpd.read_file(ethiopia_shapefile)


print(f"Original Shapefile CRS: {ethiopia_shape.crs}")

input_folder = "C:/Users/limai/Desktop/KU LEUVEN/02_Environmental Programming/Assignment/ssp585/UKESM1-0-LL"
output_folder = "C:/Users/limai/Desktop/KU LEUVEN/02_Environmental Programming/Assignment/ssp585_clipped/UKESM1-0-LL_clipped"
os.makedirs(output_folder, exist_ok=True)  # Create output folder if it doesn't exist

# Step 4: Iterate through all NetCDF files in the folder
for nc_file in os.listdir(input_folder):
    if nc_file.endswith('.nc'):
        input_path = os.path.join(input_folder, nc_file)
        output_path = os.path.join(output_folder, f"{os.path.splitext(nc_file)[0]}_clipped.nc")
        
        print(f"Processing file: {nc_file}")
        
        try:
            # Step 5: Load the NetCDF file and explicitly set the backend engine
            ds = xr.open_dataset(input_path, engine="netcdf4")  # Specify the backend explicitly
            print(f"Dataset dimensions: {ds.dims}, variables: {list(ds.data_vars)}")

            # Step 6: Check or assign CRS for the NetCDF file
            if not hasattr(ds, 'rio') or not ds.rio.crs:
                print("No CRS found in NetCDF; assuming EPSG:4326.")
                ds = ds.rio.write_crs("EPSG:4326")  # Assign CRS if missing

            # Step 7: Clip the dataset using the Ethiopia shapefile
            clipped_ds = ds.rio.clip(ethiopia_shape.geometry, ethiopia_shape.crs, drop=True)

            # Step 8: Save the clipped dataset
            clipped_ds.to_netcdf(output_path)
            print(f"Saved clipped file: {output_path}")
        except Exception as e:
            print(f"Error processing {nc_file}: {e}")

print("All files processed.")

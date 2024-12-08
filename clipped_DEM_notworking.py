# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 21:21:53 2024

@author: farah
"""
import os
import geopandas as gpd
import rioxarray
from rasterio.merge import merge
from rasterio.io import MemoryFile
import rasterio

# Paths
dem_folder = "C:/Users/farah/Desktop/Masters/environmental_programming/OneDrive_1_12-8-2024"  # Folder containing the DEM files
ethiopia_shapefile = "C:/Users/farah/Desktop/Masters/environmental_programming/reprojected_shp_eth/reprojected_ethiopia_shapefile.shp"  # Path to the Ethiopia shapefile
output_clipped_dem = "C:/Users/farah/Desktop/Masters/environmental_programming/DEM_clipped/ethiopia_clipped_dem.tif"  # Output path for the clipped DEM

# Step 1: Load the Ethiopia shapefile
ethiopia_shape = gpd.read_file(ethiopia_shapefile).to_crs("EPSG:4326")  # Ensure CRS matches DEMs

# Step 2: Merge the DEM files
# List all .tif files in the DEM folder
dem_files = [os.path.join(dem_folder, f) for f in os.listdir(dem_folder) if f.endswith('.tif')]

# Open the DEM files with rasterio
src_files = [rasterio.open(dem) for dem in dem_files]

# Merge the DEMs into a single raster
merged_dem, merged_transform = merge(src_files)

# Close the source files
for src in src_files:
    src.close()

# Check the shape of the merged DEM
print(f"Merged DEM shape: {merged_dem.shape}")  # Should be (bands, height, width)

# Step 3: Clip the merged DEM to Ethiopia's shapefile
with MemoryFile() as memfile:
    with memfile.open(
        driver="GTiff",
        height=merged_dem.shape[1],  # Height of the merged DEM
        width=merged_dem.shape[2],   # Width of the merged DEM
        count=merged_dem.shape[0],   # Number of bands
        dtype=merged_dem.dtype.name,
        transform=merged_transform,
        crs="EPSG:4326"
    ) as dataset:
        # Write the merged DEM data to the dataset
        dataset.write(merged_dem)

        # Clip the dataset using Ethiopia's geometry
        clipped_dem = rioxarray.open_rasterio(memfile.name).rio.clip(
            ethiopia_shape.geometry, ethiopia_shape.crs, drop=True
        )

# Step 4: Save the clipped DEM
clipped_dem.rio.to_raster(output_clipped_dem)
print(f"Clipped DEM saved to {output_clipped_dem}")


# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 13:55:35 2024

@author: farah
"""

import geopandas as gpd

# File path to the shapefile
ethiopia_shapefile = "C:/Users/farah/Desktop/Masters/environmental_programming/Ethiopia_All/Eth_Zone_2013.shp"  # Replace with your shapefile path
reprojected_shapefile = "reprojected_ethiopia_shapefile.shp"  # Output reprojected shapefile

# Target CRS (EPSG:4326)
target_crs = "EPSG:4326"

# Step 1: Load the Shapefile
ethiopia_shape = gpd.read_file(ethiopia_shapefile)
print(f"Original Shapefile CRS: {ethiopia_shape.crs}")

# Step 2: Reproject if Needed
if ethiopia_shape.crs != target_crs:
    print(f"Reprojecting shapefile to {target_crs}...")
    ethiopia_shape = ethiopia_shape.to_crs(target_crs)
    ethiopia_shape.to_file(reprojected_shapefile)
    print(f"Reprojected shapefile saved to: {reprojected_shapefile}")
else:
    print("Shapefile already in target CRS.")

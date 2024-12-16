# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 00:20:11 2024

@author: farah
"""
import os
import xarray as xr
import matplotlib.pyplot as plt

# Set working directory
new_directory = "C:/Users/farah/Desktop/Masters/environmental_programming"
os.chdir(new_directory)

parent_dir = os.getcwd()

# Path to the clipped NetCDF data
path_to_data = fr"{parent_dir}\Data\ssp585\ssp585\clipped_nc"

# Load datasets function
def load_dataset_by_model(load_variables="all"):
    """
    Parameters
    ----------
    load_variables : TYPE, optional
        DESCRIPTION. The default is "all".

    Returns dataset
    -------
    dataset : dict
        Dictionary of datasets organized by model and variable.
        Example:
        {
            'Model1': {'tasmin': <xarray.Dataset>, 'tasmax': <xarray.Dataset>, ...},
            'Model2': {'tasmin': <xarray.Dataset>, 'tasmax': <xarray.Dataset>, ...},
            ...
        }
    """
    dataset = {}
    if load_variables == "all":
        load_variables = ['tasmin', 'tasmax', 'sfcwind', 'rsds', 'pr', 'hurs']
    
    for model_directory in os.scandir(path_to_data):
        model_name = model_directory.name  # Use folder name as model name
        dataset[model_name] = {}
        
        for climate_var in load_variables:
            for filename in os.listdir(model_directory.path):
                if climate_var in filename and filename.endswith('.nc'):
                    file_path = os.path.join(model_directory.path, filename)
                    print(f"Model: {model_name}, Variable: {climate_var}, File: {file_path}")
                    dataset[model_name][climate_var] = xr.open_dataset(file_path)
                    break
    return dataset

# Load datasets
datasets_by_model = load_dataset_by_model(load_variables="all")

def compute_multi_model_mean(datasets_by_model, variable_to_be_averaged):
    """
    Computes the multi-model mean for a given variable across all models.

    Parameters:
    ----------
    datasets_by_model : dict
        Dictionary containing datasets for all models.
    variable_to_be_averaged : str
        Variable name (e.g., 'tasmin', 'pr').

    Returns:
    --------
    multi_model_mean : xarray.DataArray
        Multi-model mean of the variable.
    """
    datasets = []
    for model_name, variables in datasets_by_model.items():
        if variable_to_be_averaged in variables:
            datasets.append(variables[variable_to_be_averaged][variable_to_be_averaged])
    
    # Combine datasets across a new 'model' dimension
    combined = xr.concat(datasets, dim="model")
    
    # Compute mean across the 'model' dimension
    multi_model_mean = combined.mean(dim="model")
    return multi_model_mean

def plot_multi_model_mean(multi_model_mean, variable_name):
    """
    Plots the spatial distribution of the multi-model mean for a variable.

    Parameters:
    ----------
    multi_model_mean : xarray.DataArray
        Multi-model mean to be plotted.
    variable_name : str
        Name of the variable (e.g., 'tasmin').
    """
    # Ensure the data is 2D by averaging over 'time' if necessary
    if 'time' in multi_model_mean.dims:
        multi_model_mean = multi_model_mean.mean(dim="time")  # Take mean over time

    # Plotting the multi-model mean
    plt.figure(figsize=(10, 8), dpi=300)
    multi_model_mean.plot(
        cmap="viridis",  # Colormap for the plot
        cbar_kwargs={"label": f"{variable_name} (units)"}  # Colorbar label
    )
    
    # Adding labels and titles
    plt.xlabel('Longitude', fontsize=18)
    plt.ylabel('Latitude', fontsize=18)
    plt.title(f"Multi-Model Annual Average {variable_name}", fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    # Save the plot
    output_folder = os.path.join(parent_dir, "plots1", variable_name)
    os.makedirs(output_folder, exist_ok=True)
    plot_filename = f"{variable_name}_multi_model_mean.png"
    plt.savefig(os.path.join(output_folder, plot_filename))
    plt.close()
    print(f"Saved plot for {variable_name}: {plot_filename}")

# Example: Compute and plot multi-model mean for tasmin
multi_model_mean_tasmin = compute_multi_model_mean(datasets_by_model, "tasmin")
plot_multi_model_mean(multi_model_mean_tasmin, "tasmin")


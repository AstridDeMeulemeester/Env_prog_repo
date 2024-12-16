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


parent_dir = os.getcwd()
path_to_data = fr"{parent_dir}\Data"


#Functie zoekt klimaatdata in bestanden en maakt dictionary van xarray-datasets
def load_dataset(load_variables="all"):
    """
    Parameters
    ----------
    load_variables : TYPE, optional
        DESCRIPTION. The default is "all".

    Returns dataset
    -------
    dataset : TYPE
        DESCRIPTION.

    """

    filelist = []
    if load_variables == "all":
        load_variables = ['tasmin', 'tasmax', 'sfcwind', 'rsds', 'pr', 'hurs']
    for climate_var in load_variables:
        for model_directory in os.scandir(os.path.join(parent_dir, 'Data', 'ssp585', 'ssp585', "clipped_nc")):
            for filename in os.listdir(model_directory):             
                if climate_var in filename and filename.endswith('.nc'):
                    file_path = os.path.join(model_directory, filename)
                    filelist.append(file_path)
                    print(climate_var, "Found file:", file_path)
    dataset = {climate_var: xr.open_dataset(file_path) for climate_var, file_path in zip(load_variables, filelist)}
    return dataset


#Functie berekent jaarlijkse gemiddelde meteorologische waardes van een variabele uit een dataset en plot de resultaten
def plot_mean_value(dataset, variable_to_be_plotted):
    variable_ds = dataset[variable_to_be_plotted]
    meaned_variable = variable_ds.mean(dim="time")
    
    # Plotting annual mean of variable
    plt.figure(figsize=(10, 8), dpi=300)
    meaned_variable[variable_to_be_plotted].plot()
    
    # Adding labels and titles
    plt.xlabel('Longitude', fontsize=18)
    plt.ylabel('Latitude', fontsize=18)
    plt.title("Annual Average {}".format(variable_to_be_plotted), fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True)
    
    
#functie herschaalt DEM-bestanden naar een doelresolutie van 0.5 graden en slaat ze op in een nieuwe map
def rescale_and_save_dem_files():
    subdirectories = ['Data', 'DEM_EthiopiaV3']
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
    
#functie opent oude en nieuwe DEM's    
def open_rescaled_dem_file():
    old_dem = fr"{parent_dir}\Data\DEM_EthiopiaV3\n15_e037_3arc_v2.tif"
    new_dem = fr"{parent_dir}\Data\DEM_Ethiopia_reassembledV3\n15_e037_3arc_v2.tif"
    with rasterio.open(old_dem) as oldsrc:
        print("old height", oldsrc.height)
        print("old resolution", oldsrc.res)
    with rasterio.open(new_dem) as newsrc:
        print("new height", newsrc.height)
        print("new resolution", newsrc.res)
        
        
#functie maakt hoogtemozaÃ¯ek, slaat het op en toont kaart        
def make_dem_mosaic():
    path_to_reassembled_dem = fr"{parent_dir}\Data\DEM_Ethiopia_reassembledV3/"
    list_of_files = os.listdir(path_to_reassembled_dem)
    raster_to_mosaic = []
    for file in list_of_files:
        if file.endswith('.tif'):
            raster = rasterio.open(path_to_reassembled_dem+file)
            raster_to_mosaic.append(raster)
    mosaic, output = merge(raster_to_mosaic)
    output_mosaic = "merged_mosaic_height"
    np.savetxt(path_to_reassembled_dem+output_mosaic, np.array(mosaic).reshape((26,38)))
    mosaic = np.array(mosaic).reshape((26,38))
    plt.imshow(mosaic) 
    plt.Figure(figsize=(10, 8))
    plt.xlabel('Longitude', fontsize=18)
    plt.ylabel('Latitude', fontsize=18)
    plt.title("Height", fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True)
    plt.show()
    

#SPI berekeningen .......................


#berekenen van maandelijkse gemiddelde neerslag
def monthly_average_precipitation():
    dataset = load_dataset(load_variables=["pr"])
    precipitation = dataset['pr'].to_array()
    
    #Calculate montly average precipitation
    monthly_average_pr = precipitation.resample(time='M').sum()
    
    #Save as netCDF file in new precipitation folder & otherwise create new folder
    path_to_monthly_average_pr_folder = path_to_data + r"\precipitation"
    if not os.path.exists(path_to_monthly_average_pr_folder):
        os.makedirs(path_to_monthly_average_pr_folder)
    monthly_average_pr.to_netcdf(path_to_monthly_average_pr_folder+"\monthly_average_pr")


#neerslag (precipitatie) opgesplitst in historisch, nabije toekomst en verre toekomst
def split_monthly_average_pr():
    path_to_monthly_average_pr = path_to_data + r"\precipitation" + "\monthly_average_pr"
    monthly_average_pr_xr = xr.open_dataarray(path_to_monthly_average_pr)
    
    #Split monthly precipitation in three time windows
    monthly_precipitation_hist = monthly_average_pr_xr.sel(time=slice('1971-01-01', '2000-12-31'))
    monthly_precipitation_nearfut = monthly_average_pr_xr.sel(time=slice('2040-01-01', '2069-12-31'))
    monthly_precipitation_farfut = monthly_average_pr_xr.sel(time=slice('2070-01-01', '2099-12-31'))
    
    #Save results as netCDF files
    path_to_monthly_average_pr_folder = path_to_data + r"\precipitation"
    monthly_precipitation_hist.to_netcdf(path_to_monthly_average_pr_folder+"\monthly_precipitation_hist")
    monthly_precipitation_nearfut.to_netcdf(path_to_monthly_average_pr_folder+"\monthly_precipitation_nearfut")
    monthly_precipitation_farfut.to_netcdf(path_to_monthly_average_pr_folder+"\monthly_precipitation_farfut")


#maandelijkse neerslag laden voor elke tijdsperiode
def load_monthly_precipitation_for_era(era, accumulation_period):
    path_to_precipitation = path_to_data + r"\precipitation"
    
    #Opening netcdf file depending on era input
    if era == "hist":
        precipitation = xr.open_dataset(path_to_precipitation+"\monthly_precipitation_hist", engine='netcdf4')
    elif era == "nearfut":
        precipitation = xr.open_dataset(path_to_precipitation+"\monthly_precipitation_nearfut", engine='netcdf4')
    elif era == "farfut":
        precipitation = xr.open_dataset(path_to_precipitation+"\monthly_precipitation_farfut", engine='netcdf4')
    else:
        raise ValueError("Invalid era specified")   #Gives error in case of typo in era
    sum = precipitation.coarsen(time=accumulation_period).sum()
    return sum


#berekening SPI (standarized precipitation index), anders gestandaardiseerde neerslag gebruikt
def calculate_spi(precipitation, distribution, computing_SPI=False):
    try:
        fit_status = True
        params = distribution.fit(precipitation)
        prob = distribution.cdf(precipitation, *params)
        spi = stats.norm.ppf(prob)
    except:
        fit_status = False
        spi = False
        if computing_SPI:
            standardized_precipitation = stats.zscore(precipitation)
            params = distribution.fit(standardized_precipitation)
            prob = distribution.cdf(standardized_precipitation, *params)
            spi = stats.norm.ppf(prob)
    return spi, fit_status


#SPI berekenen voor verschillende tijdschalen
def calculate_spi_for_dataset(dataset, distribution, accumulation_periods):
    results = np.empty((len(accumulation_periods), len(dataset['time']), ), dtype=float)
    for i, time_step in enumerate(accumulation_periods):
        precipitation = dataset.to_array()
        results[:,i, ] = calculate_spi(precipitation, distribution, time_step)
    return results


#rejection frequency berekenen
def compute_rejection_frequency_for_SPI(n_success, acc_period, n_pixels):
    rej_freq = (1-n_success/(acc_period*n_pixels)) * 100
    return rej_freq
    

#rejection frequency voor SPI berekenen
def compute_and_save_rejection_frequency_for_SPI(distribution_dict_spi):
    results_dictionnary = {}
    for era in ["hist", "nearfut", "farfut"]:
        results_dictionnary[era] = {}
        for dist_key in distribution_dict_spi.keys():
            rejection_freq = []
            print("Now trying to fit" + dist_key + " distribution for accumulation periods..")
            accumulation_periods = [1, 3, 6, 9, 12, 24]
            for accumulation_period in tqdm(accumulation_periods):
                n_success_fit = 0
                pr_era_period = load_monthly_precipitation_for_era(era, accumulation_period)
                # Get rid of Xarray and get some nice numpy arrays
                pr_era_array = np.array(pr_era_period.to_array())
                shape = pr_era_array.shape
                # flattening array
                pr_era_array  = pr_era_array.reshape(shape[2:])
                # looping over every pixel
                pr_pixel_list = np.ravel(pr_era_array)
                n_pixels =len(pr_pixel_list)
                for pixel in tqdm(pr_pixel_list, colour="red"):
                    spi, status = calculate_spi(pixel, distribution_dict_spi[dist_key])
                    if status == True:
                        n_success_fit += 1
                rejection_freq.append(compute_rejection_frequency_for_SPI(n_success_fit, accumulation_period, n_pixels))
            results_dictionnary[dist_key] = rejection_freq
            path_to_rej_freq = "Data/rejection_frequency_SPI/rejection_frequency_{}.txt".format(dist_key)
            header = "Computed rejection frequency for {} distribution, for different accumulation periods".format(dist_key) + "\n Acumulationperiod\t Rejfreq"
            np.savetxt(path_to_rej_freq, np.array([accumulation_periods, rejection_freq]).T, header=header, delimiter="\t")
                       
            
#draw figures,... voor SPI                      
def load_and_draw_figures_for_distribution_choice_SPI(distribution_dict_spi):
    fig, ax = plt.subplots(3, figsize=(12, 8))
    axes = np.ravel(ax)
    colors = ["firebrick", "teal", "darkorchid", "darkorange", "pink", "brown", "blue", "black"]
    for era, ax in zip(["hist", "nearfut", "farfut"], axes):
        for dist_key, color in zip(distribution_dict_spi.keys(), colors):
            path_to_results =  "Data/rejection_frequency_SPI/rejection_frequency_{}.txt".format(dist_key)
            accumulation, rej_freq = np.loadtxt(path_to_results).T
            ax.plot(accumulation, rej_freq, label=dist_key, color=color)
            ax.scatter(accumulation, rej_freq, color=color)
            ax.set_ylim(0,105)
            ax.legend()
            ax.set_title(era)
    fig.supxlabel('Accumulation Period [months]')
    fig.supylabel('Rejection Frequency [-]')
    plt.legend()
    plt.tight_layout()
    plt.savefig("Data/rejection_frequency_SPI/rejection_figure.png", dpi=300)
    plt.show()

           
#SPEI.............

#PET berekenen met meteorologische data en hoogte    
def compute_PET():
    path_to_reassembled_dem = path_to_data + r"\DEM_Ethiopia_reassembledV3/"

    height_array = "merged_mosaic_height"
    height = np.loadtxt(path_to_reassembled_dem+height_array)
    
    dataset = load_dataset()

    PET = ET0(tasmax=np.array(dataset["tasmax"].to_array()),
              tasmin=np.array(dataset["tasmin"].to_array()),
              hurs=np.array(dataset["hurs"].to_array()),
              sfcWind=np.array(dataset["sfcwind"].to_array()),
              rsds=np.array(dataset["rsds"].to_array()),
              rsdt=np.array(dataset["rsds"].to_array()),
              height=height)
    PET_dataarray = dataset["sfcwind"].to_array().copy(data=PET)
    path_to_PET_folder = path_to_data + r"\PET"
    if not os.path.exists(path_to_PET_folder):
        os.makedirs(path_to_PET_folder)
    PET_dataarray.to_netcdf(path_to_PET_folder+"\computed_PET")
    #rsdt is not given in our data
    
    
#berekenen waterbalans: precipitatie - PET  
def compute_water_balance():
    #Path to PET data
    path_to_PET = path_to_data + r"\PET"+"\computed_PET"
    pet_xr = xr.open_dataarray(path_to_PET)
    dataset = load_dataset(load_variables=["pr"])
    
    #Computing water balance
    water_balance = np.array(dataset["pr"].to_array()) - np.array(pet_xr)
    
    #Creating a folder for the new water balance data (if not yet existing)
    path_to_water_balance_folder = path_to_data + r"\water_balance"
    if not os.path.exists(path_to_water_balance_folder):
        os.makedirs(path_to_water_balance_folder)
        
    #Copying precipitation data array and replace data with computed water balance
    water_balance_dataarray = dataset["pr"].to_array().copy(data=water_balance)
    #Saving water balance array to new netCDF file in the (new) folder
    water_balance_dataarray.to_netcdf(path_to_water_balance_folder+"\water_balance")
    
    
#maandelijks gemiddelde waterbalans berekenen    
def monthly_average_water_balance():
    path_to_water_balance = path_to_data + r"\water_balance" + "\water_balance"
    water_balance_xr = xr.open_dataarray(path_to_water_balance)
    
    #Computing monthly water balance
    monthly_water_balance = water_balance_xr.resample(time='M').sum()
    
    #Saving monthly water balance as netCDF in the water balance folder
    path_to_monthly_water_balance_folder = path_to_data + r"\water_balance"
    monthly_water_balance.to_netcdf(path_to_monthly_water_balance_folder+"\monthly_water_balance")


#waterbalans opgesplitst in historisch, nabije toekomst en verre toekomst
def split_monthly_water_balance():
    path_to_monthly_water_balance = path_to_data + r"\water_balance" + "\monthly_water_balance"
    monthly_water_balance_xr = xr.open_dataarray(path_to_monthly_water_balance)
    
    #Split the monthly sum data into three time windows
    monthly_water_balance_hist = monthly_water_balance_xr.sel(time=slice('1971-01-01', '2000-12-31'))
    monthly_water_balance_nearfut = monthly_water_balance_xr.sel(time=slice('2040-01-01', '2069-12-31'))
    monthly_water_balance_farfut = monthly_water_balance_xr.sel(time=slice('2070-01-01', '2099-12-31'))
    
    #Saving the monthly sum data for 3 windows in 3 netCDF files
    path_to_monthly_water_balance_folder = path_to_data + r"\water_balance"
    monthly_water_balance_hist.to_netcdf(path_to_monthly_water_balance_folder+"\monthly_water_balance_hist")
    monthly_water_balance_nearfut.to_netcdf(path_to_monthly_water_balance_folder+"\monthly_water_balance_nearfut")
    monthly_water_balance_farfut.to_netcdf(path_to_monthly_water_balance_folder+"\monthly_water_balance_farfut")


def load_monthly_water_balance_for_era(era, accumulation_period):
    path_to_water_balance = path_to_data + r"\water_balance"
    
    #Opening netcdf file depending on era input
    if era == "hist":
        water_balance = xr.open_dataset(path_to_water_balance +"\monthly_water_balance_hist", engine='netcdf4')
    elif era == "nearfut":
        water_balance = xr.open_dataset(path_to_water_balance +"\monthly_water_balance_nearfut", engine='netcdf4')
    elif era == "farfut":
        water_balance = xr.open_dataset(path_to_water_balance +"\monthly_water_balance_farfut", engine='netcdf4')
    else:
        raise ValueError("Invalid era specified")   #Gives error in case of typo in era
    sum = water_balance.coarsen(time=accumulation_period).sum()
    return sum


def calculate_spei(water_balance, distribution, computing_SPEI=True):
    try:
        fit_status = True
        params = distribution.fit(water_balance)
        prob = distribution.cdf(water_balance, *params)
        spei = stats.norm.ppf(prob)
    except:
        fit_status = False
        spei = False
        if computing_SPEI:
            standardized_water_balance = stats.zscore(water_balance)
            params = distribution.fit(standardized_water_balance)
            prob = distribution.cdf(standardized_water_balance, *params)
            spei = stats.norm.ppf(prob)
    return spei, fit_status

def calculate_spei_for_dataset(dataset, distribution, accumulation_periods):
    results = np.empty((len(accumulation_periods), len(dataset['time']), ), dtype=float)
    for i, time_step in enumerate(accumulation_periods):
        water_balance = dataset.to_array()
        results[:,i, ] = calculate_spei(water_balance, distribution, time_step)
    return results

def compute_rejection_frequency_for_SPEI(n_success, acc_period, n_pixels):
    rej_freq = (1-n_success/(acc_period*n_pixels)) * 100
    return rej_freq

def compute_and_save_rejection_frequency_for_SPEI(distribution_dict_spei):
    results_dictionnary = {}
    for era in ["hist", "nearfut", "farfut"]:
        results_dictionnary[era] = {}
        for dist_key in distribution_dict_spei.keys():
            rejection_freq = []
            print("Now trying to fit" + dist_key + " distribution for accumulation periods..")
            accumulation_periods = [1, 3, 6, 9, 12, 24]
            for accumulation_period in tqdm(accumulation_periods):
                n_success_fit = 0
                wb_era_period = load_monthly_water_balance_for_era(era, accumulation_period)
                # Get rid of Xarray and get some nice numpy arrays
                wb_era_array = np.array(wb_era_period.to_array())
                shape = wb_era_array.shape
                # flattening array
                wb_era_array  = wb_era_array.reshape(shape[2:])
                # looping over every pixel
                wb_pixel_list = np.ravel(wb_era_array)
                n_pixels =len(wb_pixel_list)
                for pixel in tqdm(wb_pixel_list, colour="red"): 
                    spei, status = calculate_spei(pixel, distribution_dict_spei[dist_key])
                    if status == True:
                        n_success_fit += 1
                rejection_freq.append(compute_rejection_frequency_for_SPEI(n_success_fit, accumulation_period, n_pixels))
            results_dictionnary[dist_key] = rejection_freq
            path_to_rej_freq = "Data/rejection_frequency_SPEI/rejection_frequency_{}.txt".format(dist_key)
            header = "Computed rejection frequency for {} distribution, for different accumulation periods".format(dist_key) + "\n Acumulationperiod\t Rejfreq"
            np.savetxt(path_to_rej_freq, np.array([accumulation_periods, rejection_freq]).T, header=header, delimiter="\t")
            
def load_and_draw_figures_for_distribution_choice_SPEI(distribution_dict_spei):
    fig, ax = plt.subplots(3, figsize=(12, 8))
    axes = np.ravel(ax)
    colors = ["firebrick", "teal", "darkorchid", "darkorange", "pink", "brown", "blue"]
    for era, ax in zip(["hist", "nearfut", "farfut"], axes):
        for dist_key, color in zip(distribution_dict_spei.keys(), colors):
            path_to_results =  "Data/rejection_frequency_SPEI/rejection_frequency_{}.txt".format(dist_key)
            accumulation, rej_freq = np.loadtxt(path_to_results).T
            ax.plot(accumulation, rej_freq, label=dist_key, color=color)
            ax.scatter(accumulation, rej_freq, color=color)
            ax.set_ylim(0,105)
            ax.legend()
            ax.set_title(era)
    fig.supxlabel('Accumulation Period [months]')
    fig.supylabel('Rejection Frequency [-]')
    plt.legend()
    plt.tight_layout()
    plt.savefig("Data/rejection_frequency_SPEI/rejection_figure.png", dpi=300)
    plt.show()
            

def estimate_spei(dataset, distribution_dict_spei, accumulation_periods): 
    results_dictionary = {}

    for era in ["hist", "nearfut", "farfut"]:
        results_dictionary[era] = {}
        
        for dist_key in distribution_dict_spei.keys():
            spei_values = []
            print("Now trying to fit " + dist_key + " distribution for accumulation periods..")

            for accumulation_period in tqdm(accumulation_periods):
                wb_era_period = load_monthly_water_balance_for_era(era, accumulation_period)
                wb_era_array = np.array(wb_era_period.to_array())
                shape = wb_era_array.shape
                wb_era_array = wb_era_array.reshape(shape[2:])
                wb_pixel_list = np.ravel(wb_era_array)

                spei_pixels = []
                for pixel in tqdm(wb_pixel_list, colour="red"):
                    spei, status = calculate_spei(pixel, distribution_dict_spei[dist_key])
                    if status:
                        spei_pixels.append(spei)

                spei_values.append(np.mean(spei_pixels))

            results_dictionary[era][dist_key] = spei_values
            path_to_spei_values = f"Data/spei_values/spei_values_{era}_{dist_key}.txt"

            # Check if the directory exists, create it if not
            directory = os.path.dirname(path_to_spei_values)
            if not os.path.exists(directory):
                os.makedirs(directory)

            # Convert accumulation_periods and spei_values to numpy arrays
            accumulation_periods_array = np.array(accumulation_periods)
            spei_values_array = np.array(spei_values)

            # Check if spei_values_array has zero dimensions (i.e., an empty array)
            if spei_values_array.ndim == 0:
                print(f"No SPEI values to save for {era} and {dist_key}.")
            else:
                header = f"Computed SPEI values for {dist_key} distribution, for different accumulation periods\n Accumulation period\t SPEI values"
                np.savetxt(path_to_spei_values, np.column_stack((accumulation_periods_array, spei_values_array)), header=header, delimiter="\t")

    return results_dictionary

def estimate_spei_for_dataset(path_to_water_balance, distribution_dict_spei):
    dataset = xr.open_dataarray(path_to_water_balance)
    accumulation_periods = [1, 3, 6, 9, 12, 24]
    results = estimate_spei(dataset, distribution_dict_spei, accumulation_periods)
    return results

path_to_water_balance = os.path.join(path_to_data, "water_balance", "water_balance")
distribution_dict_spei = {"norm": norm,
                          "genextreme": genextreme,
                          "genlogistic": genlogistic,
                          "pearson": pearson3,
                          "genpareto": genpareto,
                          "exponnorm": exponnorm,
                          "kappa3": kappa3
                         } 


def split_monthly_water_balance_by_season():
    path_to_monthly_water_balance_folder = path_to_data + r"\water_balance"
    monthly_water_balance_hist = os.path.join(path_to_monthly_water_balance_folder, "monthly_water_balance_hist")
    monthly_water_balance_nearfut = os.path.join(path_to_monthly_water_balance_folder, "monthly_water_balance_nearfut")
    monthly_water_balance_farfut = os.path.join(path_to_monthly_water_balance_folder, "monthly_water_balance_farfut")

    # Define seasons
    seasons = {
        'Winter': [12, 1, 2],
        'Spring': [3, 4, 5],
        'Summer': [6, 7, 8],
        'Autumn': [9, 10, 11]
    }

    # Load data for each era
    hist_data = xr.open_dataset(monthly_water_balance_hist)
    nearfut_data = xr.open_dataset(monthly_water_balance_nearfut)
    farfut_data = xr.open_dataset(monthly_water_balance_farfut)

    # Define a function to calculate seasonal water balance
    def calculate_seasonal_balance(data):
        return data.groupby('time.month').mean(dim='time')

    # Convert 'time' to datetime if it's not already
    for era_data in [hist_data, nearfut_data, farfut_data]:
        if not isinstance(era_data['time'].values[0], (pd.Timestamp, np.datetime64)):
            era_data['time'] = pd.to_datetime(era_data['time'].values)

    # Create a folder to store seasonal files
    output_folder = os.path.join(path_to_monthly_water_balance_folder, "seasonal_data")
    os.makedirs(output_folder, exist_ok=True)

    # Split each era into seasons and save to separate files
    for era_data, era_name in zip([hist_data, nearfut_data, farfut_data], ['historical', 'near_future', 'far_future']):
        print(f"Processing era: {era_name}")
        for season, months in seasons.items():
            print(f"  Processing season: {season}")
            
            # Filter data based on month values
            seasonal_data = era_data.where(era_data['time.month'].isin(months), drop=True)
            
            # Ensure 'time' dimension is not empty
            if len(seasonal_data['time']) == 0:
                print("    Skipping season, time dimension is empty.")
                continue

            seasonal_balance = calculate_seasonal_balance(seasonal_data)

            # Save to NetCDF file with ".nc" extension
            output_file = os.path.join(output_folder, f"{era_name}_seasonal_{season}.nc")
            seasonal_balance.to_netcdf(output_file)
            print(f"    Seasonal balance saved to: {output_file}")



#this part isn't done because we didn't get the drought index value per pixel (SPEI) => so this is written how it could be
    #drought index = SPEI:
    # Function to calculate drought characteristics
def calculate_drought_characteristics(SPEI_data, threshold=-1, consecutive_months=2):
    path_to_SPEI_data = path_to_data + r'\spei_values'  
    SPEI_data = xr.open_dataarray(path_to_SPEI_data)

    number_of_events = np.zeros_like(SPEI_data[0])
    event_duration = np.zeros_like(SPEI_data[0])
    event_severity = np.zeros_like(SPEI_data[0])

    #de huidige maand moet een droogte-indexwaarde onder de drempel hebben
    #de opeenvolgende maanden moeten allemaal droogte-indexwaarden onder de drempel hebben
    for i in range(consecutive_months - 1, len(SPEI_data)):
        is_drought = SPEI_data[i] < threshold
        is_consecutive_drought = np.all(SPEI_data[i - consecutive_months + 1:i + 1] < threshold, axis=0)

        number_of_events += is_consecutive_drought
        event_duration[is_consecutive_drought] += 1
        event_severity[is_consecutive_drought] += np.sum(SPEI_data[i, is_consecutive_drought])
    print (number_of_events, event_duration, event_severity)
    return number_of_events, event_duration, event_severity

# Comparison as example between different time windows
def plot_drought_difference(data, hist, nearfut, title):
    path_to_SPEI_data = path_to_data + r'\spei_values'  
    SPEI_data = xr.open_dataarray(path_to_SPEI_data)
    avg_hist = data.sel(time= hist).mean(dim='GCM')
    avg_nearfut = data.sel(time= nearfut).mean(dim='GCM')
    difference = avg_hist - avg_nearfut

    plt.figure(figsize=(8, 6))
    difference.plot(cmap="coolwarm", vmin=-1, vmax=1)
    plt.title(title)
    plt.show()

    # Call the function to calculate drought characteristics
    num_events, event_duration, event_severity = calculate_drought_characteristics(SPEI_data)

    #different time windows
    hist = slice('1971-01-01', '2000-12-31')
    nearfut = slice('2040-01-01', '2069-12-31')
    farfut = slice('2070-01-01', '2099-12-31')

    # Plot the spatial differences
    plot_drought_difference(num_events, hist, nearfut, 'Spatial Differences - 1971-2000 vs. Near Future')
    plot_drought_difference(num_events, hist, farfut, 'Spatial Differences - 1971-2000 vs. Far Future')


def classify_drought(SPEI_values):
    if SPEI_values >= 2:
        return 'extremely_wet'
    elif 1.5 <= SPEI_values < 2:
        return 'very_wet'
    elif 1 <= SPEI_values < 1.5:
        return 'moderately_wet'
    elif -0.99 <= SPEI_values <= 0.99:
        return 'near_normal'
    elif -1 <= SPEI_values < -1.49:
        return 'moderately_dry'
    elif -1.5 <= SPEI_values < -1.99:
        return 'severely_dry'
    elif -2 <= SPEI_values:
        return 'extremely_dry'

def count_drought_pixels(data, SPEI_thresholds):
    pixel_counts = {category: 0 for category in SPEI_thresholds}

    for time_index in data['time']:
        selected_month = data.sel(time=time_index)
        SPEI_value = selected_month['SPEI'].values
        
        # Count when SPEI_value is =< the threshold 
        for category, threshold in SPEI_thresholds.items():
            if SPEI_value <= threshold:
               pixel_counts[category] += 1

    return pixel_counts

#this still needed to be done for the 10 GCM's and the three time windows


if __name__ == "__main__":
   #dataset = load_dataset(["all"])
   #plot_mean_value(dataset, "all")
   #plt.show()
   #rescale_and_save_dem_files()
   #make_dem_mosaic()
   #open_rescaled_dem_file()  
   
   #### SPI
   #monthly_average_precipitation()
   #split_monthly_average_pr()
   #load_monthly_precipitation_for_era()
   #calculate_spi_for_dataset()
   #calculate_spi()
   #compute_rejection_frequency_for_SPI
   distribution_dict_spi = {"beta": beta,
                            "gamma": gamma,
                            "gumbel_r":gumbel_r,
                            "gumbel_l": gumbel_l,
                            "lognorm": lognorm,
                            "logistic": logistic,
                            "norm": norm,
                            "weibull_min":weibull_min
                            }
   #compute_and_save_rejection_frequency_for_SPI(distribution_dict_spi)       
   #load_and_draw_figures_for_distribution_choice_SPI(distribution_dict_spi)
   
   #### SPEI
   #compute_PET()
   #monthly_average_water_balance()
   #split_monthly_water_balance()
   #load_monthly_water_balance_for_era()
   #calculate_spei_for_dataset()
   #calculate_spei()
   #compute_rejection_frequency_for_SPEI
   distribution_dict_spei = {"norm": norm,
                             "genextreme": genextreme,
                             "genlogistic": genlogistic,
                             "pearson": pearson3,
                             "genpareto": genpareto,
                             "exponnorm": exponnorm,
                             "kappa3": kappa3
                            }
   #compute_and_save_rejection_frequency_for_SPEI(distribution_dict_spei)       
   #load_and_draw_figures_for_distribution_choice_SPEI(distribution_dict_spei)
   
   #estimate_spei()
   #estimate_spei_for_dataset()
   #results = estimate_spei_for_dataset(path_to_water_balance, distribution_dict_spei)
   #split_monthly_water_balance_by_season()
   #calculate_drought_characteristics()
   #plot_drought_difference()
   #classify_drought
   #count_drought_pixels
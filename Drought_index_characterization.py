import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import shapiro

import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings

####### lets calculate drought index based on 

from scipy.stats import beta, gamma, gumbel_r, gumbel_l, lognorm, logistic, norm, weibull_min 

# for SPEI and SPETWUI use, normal, generalized extreme value, generalized logistics, Pearson type 3, 
# generalized pareto, exponentially modified Normal and kappa 3 distributions

from scipy.stats import norm,genextreme,genlogistic,pearson3,genpareto,exponnorm,kappa3

# drought index estimation function

def calculate_spi(precipitation, distribution, time_step):
    #warnings.filterwarnings("ignore")
    params = distribution.fit(precipitation)
    #warnings.filterwarnings("ignore")
    prob = distribution.cdf(precipitation, *params)
    #warnings.filterwarnings("ignore")
    spi = stats.norm.ppf(prob, loc=0, scale=1)
    return spi

# drought index estimation for each month and Shapiro Wilk test for each month drought index

def calculate_monthly_spi_opt(data, distribution, time_step):  
    """
    
    Note: Assumes data starts with month January
    """
    # Calculate the cumulative precipitation using a moving window sum on the whole time series
    window = np.ones(time_step)
    cum_precip = np.convolve(data, window, mode='valid')

    shapiro_results = []
    spi_result = np.zeros(len(data))

    for month in range(12):
        # Extract the precipitation data for this month.
        # Note that due to the convolution, we lose the original `time_step` data points
        if month < (time_step - 1) % 12:
            monthly_precip = cum_precip[month + 12 - (time_step - 1)%12::12]
        else:
            monthly_precip = cum_precip[month - (time_step - 1)%12::12]

        # Calculate SPI for the month
        spi_values = calculate_spi(monthly_precip, distribution, time_step)

        # Re-insert the computed SPI values in the correct order.
        if month < (time_step - 1) % 12:
            spi_result[month + ((time_step-1)//12 + 1)*12::12] = spi_values
        else:
            spi_result[month + ((time_step-1)//12)*12::12] = spi_values

        #W, p_value = shapiro(spi_values.tolist())
        #shapiro_results.append({'Month': month+1, 'W': W, 'p-value': p_value})

    #shapiro_results = pd.DataFrame(shapiro_results)

    return spi_result, shapiro_results

def calculate_monthly_spi(df, distribution, time_step):
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    
    # Calculate the cumulative precipitation using a moving window sum on the whole time series
    window = np.ones(time_step)
    cum_precip = np.convolve(df['divided'].values, window, mode='valid')
    df['Cumulative'] = np.concatenate((np.full(time_step - 1, None), cum_precip))
    print(df['Cumulative'])
    print(len(df['Cumulative']))
    
    # Remove rows with None in the 'Cumulative Precipitation' column
    df = df.dropna(subset=['Cumulative'])
    print(df['Cumulative'])
    print(len(df['Cumulative']))
    
    df['SPI'] = None  # Initialize SPI column

    shapiro_results = []

    for month in range(1, 13):
        monthly_precip = df[df.index.month == month]['Cumulative'].values

        # Convert the array to a list so that the fit function will not have any problem
        monthly_precip = monthly_precip.tolist()
        print(f"{time_step} -- {len(monthly_precip)}")

        # Calculate SPI for the month
        spi_values = calculate_spi(monthly_precip, distribution, time_step)

        # Find the index positions corresponding to the current month
        month_indexes = df.index[df.index.month == month]
        
        # Update the 'SPI' column for the found index positions
        df.loc[month_indexes, 'SPI'] = spi_values

        W, p_value = shapiro(spi_values.tolist())
        shapiro_results.append({'Month': month, 'W': W, 'p-value': p_value})

    shapiro_results = pd.DataFrame(shapiro_results)

    return df, shapiro_results

################### drought characterization function using standardized index values ################

def estimate_drought_events(dataframe):
    drought_events = []
    for col in dataframe.columns[1:]:  # Iterate over each column except the first (date) column
        drought_start = None
        drought_duration = 0
        drought_severity = 0

        for index, row in dataframe.iterrows():
            spei_value = row[col]
            if spei_value < -1:  # If SPEI value is below -1, it's a drought event
                if drought_start is None:
                    drought_start = row['Date']
                drought_duration += 1
                drought_severity += spei_value
            else:
                if drought_duration > 2:  # If drought duration is above 2 months, consider it a drought event
                    drought_events.append({
                        'start_date': drought_start,
                        'end_date': row['Date'],
                        'duration': drought_duration,
                        'severity': drought_severity,
                        'column': col  # Add column name to identify the column
                    })
                drought_start = None
                drought_duration = 0
                drought_severity = 0

    return drought_events

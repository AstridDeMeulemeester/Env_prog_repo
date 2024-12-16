import numpy as np

def ET0(tasmax, tasmin, hurs, sfcWind, rsds, rsdt, height, ps=None):
    # Constants
    zero_kelvin = 273.15
    a_tmp = 0.6108
    b_tmp = 17.27
    c_tmp = 237.3

    # Unit conversions
    if np.nanmean(tasmax) > 200:
        tasmax = tasmax - zero_kelvin  # Conversion from K to °C
    if np.nanmean(tasmin) > 200:
        tasmin = tasmin - zero_kelvin  # Conversion from K to °C
    hurs = hurs
    rsds = rsds * 0.0864  # Conversion from W/m^2 to MJ/m^2/day
    rsdt = rsdt * 0.0864  # Conversion from W/m^2 to MJ/m^2/day

    if ps is None and not np.isnan(np.min(height)):
        ps = 101.3 * ((293 - 0.0065 * height) / 293) ** 5.26
    ps = ps / 1000.0  # Conversion from Pa to kPa

    # Variable definitions
    Tmean = (tasmax + tasmin) / 2
    lmbda = 2.501 - 0.002361 * Tmean

    gamma = 0.00163 * ps / lmbda
    etasmax = a_tmp * np.exp((b_tmp * tasmax) / (tasmax + c_tmp))
    etasmin = a_tmp * np.exp((b_tmp * tasmin) / (tasmin + c_tmp))
    eTmean = a_tmp * np.exp((b_tmp * Tmean) / (Tmean + c_tmp))
    es = (etasmax + etasmin) / 2
    Delta = 4098 * eTmean / (Tmean + c_tmp) ** 2
    U2 = sfcWind * 4.87 / (np.log(67.8 * 10 - 5.42))
    ea = hurs / 100 * (etasmin + etasmax) / 2
    Rso = (0.75 + 2e-05 * height) * rsdt
    Rn = (1 - 0.23) * rsds - (1.35 * rsds / Rso - 0.35) * (0.34 - 0.14 * np.sqrt(ea)) * 4.903e-09 * (
            (zero_kelvin + tasmax) ** 4 + (zero_kelvin + tasmin) ** 4) / 2
    Rn[np.isnan(rsds)] = 0
    G = 0
    ET0 = (0.408 * Delta * (Rn - G) + gamma * (900 / (Tmean + zero_kelvin)) * U2 * (es - ea)) / (
            Delta + gamma * (1 + 0.34 * U2))
    ET0[ET0 < 0] = 0

    return ET0

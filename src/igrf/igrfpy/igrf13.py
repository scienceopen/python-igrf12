import numpy as np
import pandas as pd
import json
import os
import datetime
from dateutil.parser import parse


def load_dict_from_json(filename):
    with open(filename, 'r') as json_file:
        return json.load(json_file)
    

def IGRF13(lat, lon, alt, dates, isv=0, itype=1):
    """
    This is a Python-reconstructed function by uho-33, based on the original Fortran code:
    https://github.com/space-physics/igrf/blob/main/src/igrf/fortran/igrf13.f.
    
    The model is valid from 1900.0 to 2025.0. Values from 1945.0 to 2015.0 are definitive,
    while those outside this range are non-definitive.

    Parameters
    ----------
    lat : float, list, or numpy.ndarray
        Latitude(s) of the point(s) in degrees. Positive values indicate north latitude.
    
    lon : float, list, or numpy.ndarray
        Longitude(s) of the point(s) in degrees. Positive values indicate east longitude.

    alt : float, list, or numpy.ndarray
        Altitude(s) in kilometers:
        - If `itype = 1`: Altitude above sea level.
        - If `itype = 2`: Distance from Earth's center (> 3485 km).
    
    dates : float, str, int, datetime, list, or numpy.ndarray
        Date(s) for the IGRF calculation. Supported formats include:
        - `float`: Decimal year (e.g., 2016.5).
        - `str`: Date in 'YYYYMMDD' format (e.g., '20160708').
        - `int`: Year as an integer (e.g., 2016).
        - `datetime`: `datetime.datetime` or `datetime.date`.
        - `list` or `numpy.ndarray`: A collection of the above types.

    isv : int, optional (default=0)
        Type of values to compute:
        - `0`: Main-field values.
        - `1`: Secular variation values.

    itype : int, optional (default=1)
        Coordinate system of the input:
        - `1`: Geodetic (spheroid).
        - `2`: Geocentric (sphere).

    Returns
    -------
    numpy.ndarray
        A 2D array of shape `(4, N)`, where:
        - The first dimension (size 4) represents magnetic field components:
            - `[0]`: `B_north` – North component (nT) if `isv = 0`, nT/year if `isv = 1`.
            - `[1]`: `B_east` – East component (nT) if `isv = 0`, nT/year if `isv = 1`.
            - `[2]`: `B_vert` – Vertical (earthward) component (nT) if `isv = 0`, nT/year if `isv = 1`.
            - `[3]`: `B_total` – Total intensity (nT) if `isv = 0`, nT/year if `isv = 1`.
        - The second dimension (size N) corresponds to the number of input points.

    Examples
    --------
    Single point calculation for a specific date:
    >>> IGRF13(30.0, 232.0, 35786, 2016.5)
    Output:
    [[ 85.76211027]
    [ 17.57562939]
    [123.91279388]
    [151.71823487]]

    Multiple points and dates:
    >>> IGRF13([30.0, 40.0], [232.0, 250.0], [35786, 40000], ['20160708', '20170809'])
    Output:
    [[ 81.3243117   52.36266943]
    [  9.52994813   6.89129736]
    [124.99133933 113.81947292]
    [149.42321939 125.47594011]]
    
    
    """
    # Import gh data
    script_path = os.path.realpath(__file__)
    script_directory = os.path.dirname(script_path)
    gh = load_dict_from_json(os.path.join(script_directory, 'data/gh_data.json'))
    
    # Ensure the variables are 1-dimensional arrays
    if not isinstance(dates, (list, np.ndarray)):
        dates = [dates]
        lat =[lat]
        lon = [lon]
        alt = [alt]
    lat, lon, alt, dates = map(np.array,[lat, lon, alt, dates])

    # Check if all input arrays have the same shape
    if not (lat.shape == lon.shape == alt.shape == dates.shape):
        raise ValueError(f"Input arrays must have the same shape. "
                         f"Received shapes: lat={lat.shape}, lon={lon.shape}, alt={alt.shape}, dates={dates.shape}")
    
    # Convert dates to decimal years
    dates = datetime2yeardec(dates)

    # Initialize output arrays
    B_north = np.zeros((len(dates)))
    B_east = np.zeros((len(dates)))
    B_vert = np.zeros((len(dates)))
    B_total = np.zeros((len(dates)))
    
    colat = 90 - lat
    elong = lon

    out_of_bounds = (dates < 1900.0) | (dates > 2030.0)
    if np.any(out_of_bounds):
        raise ValueError("Error: Some dates are out of bounds (1900-2030):{}".format(set(map(int, dates[out_of_bounds]))))

    # Vectorized warning for dates after 2025
    accuracy_warning = dates > 2025.0
    if np.any(accuracy_warning):
        print("Warning: Accuracy may be reduced for year: {}".format(set(map(int, dates[accuracy_warning]))))

    # Calculate time-related variables
    t = np.zeros_like(dates)
    tc = np.zeros_like(dates)
    mask_2020 = dates >= 2020.0
    t = np.where(mask_2020, 
                 1.0 if isv==1 else dates-2020,
                 0.2 if isv==1 else (dates - 1900)%5/5 )
    tc = np.where(mask_2020,
                  0.0 if isv==1 else 1.0,
                  -0.2 if isv==1 else 1.0-t)

    nmx = np.where(dates<2000, 10, 13)
    kmx   = (nmx+1)*(nmx+2)/2
    

    # Initial trigonometric calculations
    cl = np.zeros((len(dates),13))
    sl = np.zeros((len(dates),13))
    colat_rad, elong_rad = np.radians(colat), np.radians(elong)
    ct, st = np.cos(colat_rad), np.sin(colat_rad) 
    cl[:,0] = np.cos(elong_rad)
    sl[:,0] =  np.sin(elong_rad)
    cd = 1.
    sd = 0.
    
    # Convert geocentric to geographic coordinates if necessary
    if itype != 2:
        a2 = 40680631.6
        b2 = 40408296.0
        one   = a2*st*st
        two   = b2*ct*ct
        three = one + two
        rho = np.sqrt(three)
        r = np.sqrt(alt * (alt + 2.0 * rho) + (a2 * one + b2 * two) / (three))
        cd = (alt + rho) / r
        sd = ((a2 - b2) / rho) * ct * st / r
        ct, st = ct * cd - st * sd, st * cd + ct * sd
    else:
        r = alt
    ratio = 6371.2/r
    rr = ratio**2
    
    max_kmx = int(np.max(kmx))
    p = np.zeros((len(dates), max_kmx))
    q = np.zeros((len(dates), max_kmx))
    p[:,0] = 1. 
    p[:,2] = st
    q[:,0] = 0.
    q[:,2] = ct
    fn, gn = np.zeros(len(dates)), np.zeros(len(dates))
    l, m, n = np.ones(len(dates)), np.ones(len(dates)), np.zeros(len(dates))
    fm = m
    year = np.where(mask_2020, 2020, dates//5*5)
    year_next = np.where(mask_2020, 2022, year)
    gh_values_before = np.array([gh[str(int(key))] for key in year])
    gh_values_after = np.array([gh[str(int(key))] for key in year_next])
    

    for k in range(1, max_kmx):
        # one, two, and three are used to store intermediate values for calculations
        one, two, three = np.zeros(len(dates)), np.zeros(len(dates)), np.zeros(len(dates)) 

        valid_mask = k <= kmx
        reset_mask = (n < m) & valid_mask
        m = np.int32(np.where(reset_mask, 0, m))
        n = np.int32(np.where(reset_mask, n+1, n))
        rr = np.where(reset_mask, rr*ratio, rr)
        fn = np.where(reset_mask, n, fn)
        gn = np.where(reset_mask, n-1, gn)
        
        fm = np.where(valid_mask, m, fm)
        
        equal_mask = (m == n) & valid_mask
        equal_and_not2_mask = equal_mask & (k != 2) & valid_mask
        if np.any(equal_and_not2_mask):
            one[equal_and_not2_mask] = np.sqrt(1.0 - 0.5 / fm[equal_and_not2_mask])
        j = np.int32(np.where(equal_and_not2_mask, k-n-1, 0))
        p[:,k] = np.where(equal_and_not2_mask, 
                          one * st *p[np.arange(len(j)),j], 
                          p[:,k])
        q[:,k] = np.where(equal_and_not2_mask, 
                          one * (st * q[np.arange(len(j)),j] + ct * p[np.arange(len(j)),j]), 
                          q[:,k])
        cl[:,m-1] = np.where(equal_and_not2_mask, 
                             cl[np.arange(len(m)),m-2] * cl[:,0] - sl[np.arange(len(m)),m-2] * sl[:,0], 
                             cl[np.arange(len(m)),m-1])
        sl[:,m-1] = np.where(equal_and_not2_mask, 
                             sl[np.arange(len(m)),m-2] * cl[:,0] + cl[np.arange(len(m)),m-2] * sl[:,0], 
                             sl[np.arange(len(m)),m-1])

        notequal_mask = (m != n) & valid_mask
        gmm = np.where(notequal_mask, m**2, 0)
        one = np.where(notequal_mask, np.sqrt(fn**2 - gmm), one)
        
        if np.any(notequal_mask): 
            two[notequal_mask] = np.sqrt(gn**2 - gmm) / one[notequal_mask]
            three[notequal_mask] = (fn + gn) / one[notequal_mask]

        i = np.int32(np.where(notequal_mask, k - n, 0)) 
        j = np.where(notequal_mask, i - n + 1, j)
        p[:,k] = np.where(notequal_mask, 
                          three * ct * p[np.arange(len(i)),i] - two * p[np.arange(len(i)),j], 
                          p[:,k])
        q[:,k] = np.where(notequal_mask, 
                          three * (ct * q[np.arange(len(i)),i] - st * p[np.arange(len(i)),i]) - two * q[np.arange(len(j)),j], 
                          q[:,k])
        
        lm = np.int32(l-1)
        one = (tc * gh_values_before[np.arange(len(lm)),lm]+ t * gh_values_after[np.arange(len(lm)),lm]) * rr
        
        noteuqal0_mask = m != 0
        two = np.where(noteuqal0_mask, 
                       (tc *gh_values_before[np.arange(len(lm)),lm+1] + t * gh_values_after[np.arange(len(lm)),lm+1]) * rr, 
                       two)
        three = np.where(noteuqal0_mask, 
                         one * cl[np.arange(len(lm)),m-1] + two * sl[np.arange(len(lm)),m-1], 
                         three)
        B_north += np.where(noteuqal0_mask, 
                            np.squeeze(three * q[:,k]), 
                            np.squeeze(one * q[:,k]))
        B_vert -= np.where(noteuqal0_mask, 
                           np.squeeze((fn + 1.0) * three * p[:,k]), 
                           np.squeeze((fn + 1.0) * one * p[:,k]))

        less_mask = st-0 < 10e-8
        noteuqal0_and_less_mask = noteuqal0_mask & less_mask
        noteuqal0_and_greater_mask = noteuqal0_mask & ~less_mask
        if np.any(noteuqal0_and_less_mask):
            B_east += np.where(noteuqal0_mask & less_mask, 
                           np.squeeze((one * sl[np.arange(len(m)),m-1] - two * cl[np.arange(len(m)),m-1]) * q[:,k] * ct), 
                           0)
        if np.any(noteuqal0_and_greater_mask):
            B_east += np.where(noteuqal0_mask & ~less_mask, 
                               (one * sl[np.arange(len(m)),m-1] - two * cl[np.arange(len(m)),m-1]) * fm * p[:,k] / st,
                               0)
        l += np.int32(np.where(noteuqal0_mask, 2, 1))
        m += 1

    one = B_north
    B_north = B_north*cd + B_vert*sd
    B_vert = B_vert*cd - one*sd
    B_total = np.sqrt(B_north**2 + B_east**2 + B_vert**2)
    
    return np.array([B_north, B_east, B_vert, B_total])


def IGRF13_v(data_matrix, isv=0, itype=1):
    """
    Vectorized wrapper function for the function IGRF13.

    Parameters
    ----------
    data_matrix : numpy.ndarray
        A 2D array where each row contains input data in the following columns:
        - `[:,0]`: Latitude(s) in degrees.
        - `[:,1]`: Longitude(s) in degrees.
        - `[:,2]`: Altitude(s) in kilometers.
        - `[:,3]`: Date(s) in a format supported by IGRF13 (e.g., float, str, int).
    
    isv : int, optional (default=0)
        Type of values to compute:
        - `0`: Main-field values.
        - `1`: Secular variation values.

    itype : int, optional (default=1)
        Coordinate system of the input:
        - `1`: Geodetic (spheroid).
        - `2`: Geocentric (sphere).

    Returns
    -------
    numpy.ndarray
        A 2D array of shape `(4, N)`, where:
        - The first dimension (size 4) represents magnetic field components:
            - `[0]`: `B_north` – North component (nT) if `isv = 0`, nT/year if `isv = 1`.
            - `[1]`: `B_east` – East component (nT) if `isv = 0`, nT/year if `isv = 1`.
            - `[2]`: `B_vert` – Vertical (downward) component (nT) if `isv = 0`, nT/year if `isv = 1`.
            - `[3]`: `B_total` – Total intensity (nT) if `isv = 0`, nT/year if `isv = 1`.
        - The second dimension (size N) corresponds to the number of input dates/points.

    Examples
    -------
    >>> IGRF13_v([[30.0, 232.0, 35786, '20160708'], [40.0, 250.0, 40000, '20170809']])
    Output:
    [[ 81.3243117   52.36266943]
    [  9.52994813   6.89129736]
    [124.99133933 113.81947292]
    [149.42321939 125.47594011]]
    """
    data_matrix = np.array(data_matrix)
    lat = data_matrix[:, 0].astype(float)
    lon = data_matrix[:, 1].astype(float)
    alt = data_matrix[:, 2].astype(float)
    dates = data_matrix[:, 3]

    result = IGRF13(lat, lon, alt, dates, isv, itype)
    return result


def datetime2yeardec(times):
    times = pd.to_datetime(times)
    year = times.year
    start_of_year = pd.to_datetime(year.astype(str) + '-01-01')
    next_year = start_of_year + pd.offsets.YearEnd()
    year_fraction = (times - start_of_year) / (next_year - start_of_year)
    return year + year_fraction

# def datetime2yeardec(time: str | int | np.integer |datetime.datetime | datetime.date) -> float:
#     """
#     Convert a datetime into a float. The integer part of the float should
#     represent the year.
#     Order should be preserved. If adate<bdate, then d2t(adate)<d2t(bdate)
#     time distances should be preserved: If bdate-adate=ddate-cdate then
#     dt2t(bdate)-dt2t(adate) = dt2t(ddate)-dt2t(cdate)
    
#     """

#     if isinstance(time, float):
#         # assume already year_dec
#         return time
#     if isinstance(time, str):
#         t = parse(time)
#     elif isinstance(time, (int, np.integer)):
#         t = float(time)
#         return t
#     elif isinstance(time, datetime.datetime):
#         t = time
#     elif isinstance(time, datetime.date):
#         t = datetime.datetime.combine(time, datetime.datetime.min.time())
#     elif isinstance(time, np.datetime64):
#         # Handle a single np.datetime64 element
#         t = pd.Timestamp(time.astype('datetime64[s]')).to_pydatetime()
#     elif isinstance(time, (tuple, list, np.ndarray)):
#         return np.asarray([datetime2yeardec(t) for t in time])
#     else:
#         raise TypeError("unknown input type {}".format(type(time)))

#     year = t.year

#     boy = datetime.datetime(year, 1, 1)
#     eoy = datetime.datetime(year + 1, 1, 1)

#     return year + ((t - boy).total_seconds() / ((eoy - boy).total_seconds()))

import statsmodels.api as sm


def check_stationarity(time_series):
    """
    Check the stationarity of a time series using the Dickey-Fuller test.

    :param time_series: time series data
    :return: Output of the Dickey-Fuller test
    """
    result = sm.tsa.adfuller(time_series)

    print('p-value:', result[1])
    print('Critical Values:', result[4])

    if result[1] <= 0.05:
        print("The time series is stationary.")
        return True
    print("The time series is non-stationary.")
    return False

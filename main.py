import numpy as np
import pandas as pd
from filterpy import common
from matplotlib import pyplot as plt
import pmdarima as pm
from ARIMA import ARIMA
from ExponentialSmoothing import ExponentialSmoothing
from Holt import Holt
from KalmanFilter import KalmanFilter
from stationarity_check import check_stationarity

import warnings

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    data = [446.6565, 454.4733, 455.663, 423.6322, 456.2713, 440.5881, 425.3325, 485.1494, 506.0482, 526.792, 514.2689,
            494.211]
    index = pd.date_range(start="1996", end="2008", freq="A")
    oil_data = pd.Series(data, index)

    #check_stationarity(oil_data)
    #check_stationarity(oil_data.diff().dropna())

    fig, axs = plt.subplots(4, 2, figsize=(12, 10))

    # EXPONENTIAL SMOOTHING

    exp_sm1 = ExponentialSmoothing(endog=oil_data[:-1], initialization_method="heuristic")
    exp_sm1.fit(smoothing_level=0.2, optimized=False)
    fcast1 = exp_sm1.predict(1).rename(r"$\alpha=0.2$")

    exp_sm2 = ExponentialSmoothing(endog=oil_data[:-1], initialization_method="heuristic")
    exp_sm2.fit(smoothing_level=0.6, optimized=False)
    fcast2 = exp_sm2.predict(1).rename(r"$\alpha=0.6$")

    exp_sm3 = ExponentialSmoothing(endog=oil_data[:-1], initialization_method="estimated")
    exp_sm3.fit()
    fcast3 = exp_sm3.predict(1).rename(f"alpha={exp_sm3.model.params['smoothing_level']}")

    lines = []

    exp_sm1.plot_results(axs[0][0], oil_data, fcast1, np.random.rand(3, ), lines)
    exp_sm2.plot_results(axs[0][0], oil_data, fcast2, np.random.rand(3, ), lines)
    exp_sm3.plot_results(axs[0][0], oil_data, fcast3, np.random.rand(3, ), lines)

    axs[0][0].legend(lines, [fcast1.name, fcast2.name, fcast3.name], loc='center left', bbox_to_anchor=(1, 0.5))
    axs[0][0].set_title("ExpSmothing")
    # HOLT

    holt1 = Holt(endog=oil_data[:-1], initialization_method="heuristic")
    holt1.fit(smoothing_trend=0.8, smoothing_level=0.2, optimized=False)
    fcast1 = holt1.predict(1).rename(r"$\alpha=0.2$  $\beta = 0.8$")

    holt2 = Holt(endog=oil_data[:-1], initialization_method="heuristic")
    holt2.fit(smoothing_trend=0.6, smoothing_level=0.6, optimized=False)
    fcast2 = holt2.predict(1).rename(r"$\alpha=0.6$ $\beta = 0.6$")

    holt3 = Holt(endog=oil_data[:-1], initialization_method="estimated")
    holt3.fit()
    fcast3 = holt3.predict(1).rename(r"$\alpha=%s$" % holt3.model.params["smoothing_level"])

    lines = []

    holt1.plot_results(axs[1][0], oil_data, fcast1, np.random.rand(3, ), lines)
    holt2.plot_results(axs[1][0], oil_data, fcast2, np.random.rand(3, ), lines)
    holt3.plot_results(axs[1][0], oil_data, fcast3, np.random.rand(3, ), lines)

    axs[1][0].legend(lines, [fcast1.name, fcast2.name, fcast3.name], loc='center left', bbox_to_anchor=(1, 0.5))
    axs[1][0].set_title("Holt")
    # MOOVING AVERAGE

    ma1 = ARIMA(endog=oil_data[:-3], order=(0, 0, 10))
    ma1.fit()
    fcast1 = ma1.predict(5).rename(r"$order=(0, 0, 10)$")

    ma2 = ARIMA(endog=oil_data[:-3], order=(0, 0, 2))
    ma2.fit()
    fcast2 = ma2.predict(5).rename(r"$order=(0, 0, 2)$")

    ma3 = ARIMA(endog=oil_data[:-3], order=(0, 0, 5))
    ma3.fit()
    fcast3 = ma3.predict(5).rename(r"$order=(0, 0, 5)$")

    lines = []

    ma1.plot_results(axs[2][0], oil_data, fcast1, np.random.rand(3, ), lines)
    ma2.plot_results(axs[2][0], oil_data, fcast2, np.random.rand(3, ), lines)
    ma3.plot_results(axs[2][0], oil_data, fcast3, np.random.rand(3, ), lines)

    axs[2][0].legend(lines, [fcast1.name, fcast2.name, fcast3.name], loc='center left', bbox_to_anchor=(1, 0.5))

    axs[2][0].set_title("MA")

    # AUTOREGRESSION

    ar1 = ARIMA(endog=oil_data[:-3], order=(10, 0, 0))
    ar1.fit()
    fcast1 = ar1.predict(5).rename(r"$order=(10, 0, 0)$")
    ar2 = ARIMA(endog=oil_data[:-3], order=(2, 0, 0))
    ar2.fit()
    fcast2 = ar2.predict(5).rename(r"$order=(2, 0, 0)$")
    ar3 = ARIMA(endog=oil_data[:-3], order=(5, 0, 0))
    ar3.fit()
    fcast3 = ar3.predict(5).rename(r"$order=(5, 0, 0)$")

    lines = []

    ar1.plot_results(axs[3][0], oil_data, fcast1, np.random.rand(3, ), lines)
    ar2.plot_results(axs[3][0], oil_data, fcast2, np.random.rand(3, ), lines)
    ar3.plot_results(axs[3][0], oil_data, fcast3, np.random.rand(3, ), lines)

    axs[3][0].legend(lines, [fcast1.name, fcast2.name, fcast3.name], loc='center left', bbox_to_anchor=(1, 0.5))
    axs[3][0].set_title("AR")

    # ARMA

    arima2 = ARIMA(endog=oil_data[:-5], order=(2, 0, 3))
    arima2.fit()
    fcast2 = arima2.predict(5).rename(r"$order=(2, 0, 3)$")

    lines = []

    arima2.plot_results(axs[0][1], oil_data, fcast2, np.random.rand(3, ), lines)

    axs[0][1].legend(lines, [fcast2.name], loc='center left', bbox_to_anchor=(1, 0.5))

    axs[0][1].set_title("ARMA")

    ################################

    data = [40959, 39052, 35257, 30981, 26167, 23643, 22507, 21650, 21118, 21260, 24110, 24278, 21749, 21597, 22534]
    index = pd.date_range(start="4/19/17 19:00", end="4/20/17 9:00", freq="H")
    data = pd.Series(data, index)


    # ARIMA

    auto_arima = pm.auto_arima(data[:-2], stepwise=False, seasonal=False)
    fcast_auto_arima = auto_arima.predict(n_periods=5)

    arima3 = ARIMA(endog=data[:-2], order=(2, 1, 3))
    arima3.fit()
    fcast3 = arima3.predict(5).rename(r"$order=(2, 1, 3)$")

    lines = []

    axs[1][1].plot(data, marker="o", color="black")
    axs[1][1].plot(fcast_auto_arima, marker="o", color="blue")
    (line1,) = axs[1][1].plot(fcast_auto_arima, marker="o", color="blue")
    lines.append(line1)

    arima3.plot_results(axs[1][1], data, fcast3, np.random.rand(3, ), lines)

    axs[1][1].legend(lines, [f'order={auto_arima.order}', fcast3.name], loc='center left', bbox_to_anchor=(1, 0.5))
    axs[1][1].set_title("ARIMA")
    # KALMAN

    transition_matrix = [[1, 1], [0, 1]]
    observation_matrix = [[1, 0]]
    process_covariance = [[1e-5, 0], [0, 1e-5]]
    observation_covariance = [[1e-3]]
    initial_state_mean = [0, 0]
    initial_state_covariance = [[1, 0], [0, 1]]


    dt = 0.01
    noiseSigma = 0.5
    samplesCount = 1000
    noise = np.random.normal(loc=0.0, scale=noiseSigma, size=samplesCount)

    trajectory = np.zeros((3, samplesCount))

    position = 0
    velocity = 1.0
    acceleration = 0.0

    for i in range(1, samplesCount):
        position = position + velocity * dt + (acceleration * dt ** 2) / 2.0
        velocity = velocity + acceleration * dt
        acceleration = acceleration

        trajectory[0][i] = position
        trajectory[1][i] = velocity
        trajectory[2][i] = acceleration

    measurement = trajectory[0] + noise
    processNoise = 1e-4


    # F - матрица процесса
    F = np.array([[1, dt, (dt ** 2) / 2],
                  [0, 1.0, dt],
                  [0, 0, 1.0]])

    # Матрица наблюдения
    H = np.array([[1.0, 0.0, 0.0]])

    # Ковариационная матрица ошибки модели
    Q = common.Q_discrete_white_noise(dim=3, dt=dt, var=processNoise)

    measurementSigma = 0.5
    # Ковариационная матрица ошибки измерения
    R = np.array([[measurementSigma * measurementSigma]])

    # Начальное состояние.
    x = np.array([0.0, 0.0, 0.0])

    # Ковариационная матрица для начального состояния
    P = np.array([[10.0, 0.0, 0.0],
                  [0.0, 10.0, 0.0],
                  [0.0, 0.0, 10.0]])

    k = KalmanFilter()
    k.fit(dim_x=3,  # Размер вектора стостояния
          dim_z=1,
          transition_matrix=F,
          observation_matrix=H,
          initial_state=x,
          initial_covariance=P,
          observation_covariance=R,
          process_covariance=Q
          )

    filteredState, stateCovarianceHistory = k.predict(measurement)

    axs[2][1].set_title("Kalman filter (3rd order)")
    axs[2][1].plot(measurement, label="Измерение", color="#99AAFF")
    axs[2][1].plot(trajectory[0], label="Истинное значение", color="#FF6633")
    axs[2][1].plot(filteredState[:, 0], label="Оценка фильтра", color="#224411")
    axs[2][1].legend()

    plt.tight_layout()
    plt.show()

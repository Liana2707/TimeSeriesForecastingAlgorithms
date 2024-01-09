import numpy as np

from Algorithm import Algorithm
from filterpy.kalman import KalmanFilter as kf


class KalmanFilter(Algorithm):

    def fit(self, dim_x, dim_z,
            transition_matrix,
            observation_matrix,
            process_covariance,
            observation_covariance,
            initial_state, initial_covariance
            ):
        self.model = kf(dim_x, dim_z)

        # F - матрица процесса
        self.model.F = transition_matrix

        # Матрица наблюдения
        self.model.H = observation_matrix

        # Ковариационная матрица ошибки модели
        self.model.Q = process_covariance

        measurementSigma = 0.5
        # Ковариационная матрица ошибки измерения
        self.model.R = observation_covariance

        # Начальное состояние.
        self.model.x = initial_state

        # Ковариационная матрица для начального состояния
        self.model.P = initial_covariance

    def predict(self, measurement):
        filtered_state = []
        state_covariance_history = []

        for i in range(0, len(measurement)):
            z = [measurement[i]]  # Вектор измерений
            self.model.predict()  # Этап предсказания
            self.model.update(z)  # Этап коррекции

            filtered_state.append(self.model.x)
            state_covariance_history.append(self.model.P)

        filtered_state = np.array(filtered_state)
        state_covariance_history = np.array(state_covariance_history)

        return filtered_state, state_covariance_history

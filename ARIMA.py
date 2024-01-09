import statsmodels.tsa.api as sm


from Algorithm import Algorithm


class ARIMA(Algorithm):

    def fit(self, **kwargs):
        self.data = self.options['endog']
        self.model = sm.ARIMA(**self.options)
        self.results = self.model.fit(**kwargs)

    '''def plot_results(self, actual_data, predicted_data):
        """
        Plot the actual vs. predicted values.

        :param actual_data: actual time series data
        :param predicted_data: predicted time series data
        """
        plt.figure(figsize=(10, 6))
        plt.plot(actual_data, label='Actual Data', color='blue')
        plt.plot(predicted_data, label='Predicted Data', color='red')
        plt.title('ARIMA Model - Actual vs. Predicted')
        plt.xlabel('Time Steps')
        plt.ylabel('Values')
        plt.legend()
        plt.show()
    '''

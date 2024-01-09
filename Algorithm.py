from matplotlib import pyplot as plt


class Algorithm:
    def __init__(self, **kwargs):
        """
        Initialize the Algorithm.

        :param data: time series data (pandas Series or numpy array)
        :param **kwargs: options for algorithm
        """
        self.options = kwargs
        self.data = None
        self.results = None
        self.model = None

    def fit(self, **kwargs):
        """
        Train the model on the provided data.
        """
        pass

    def predict(self, steps=1):
        """
        Make predictions for future values using the trained model.

        :param steps: number of future steps to forecast
        :return: array of predictions
        """

        if not self.model:
            raise ValueError("Model is not trained. Call the fit() method before making predictions.")
        return self.results.predict(start=len(self.data), end=len(self.data) + steps - 1)

    def get_options(self):
        """

        :return: dict of options
        """
        return self.options

    def summary(self):
        """
        Print a statistical summary of the model.
        """
        if not self.model:
            raise ValueError("Model is not trained. Call the fit() method before printing the summary.")
        print(self.results)
        print(self.results.summary())

    def plot_results(self, ax, actual_data, predicted_data, color, lines):
        """
        Plot the actual vs. predicted values.

        :param actual_data: actual time series data
        :param predicted_data: predicted time series data
        """
        ax.plot(actual_data, marker="o", color="black")
        ax.plot(self.results.fittedvalues, marker="o", color=color)
        (line1,) = ax.plot(predicted_data, marker="o", color=color)
        lines.append(line1)

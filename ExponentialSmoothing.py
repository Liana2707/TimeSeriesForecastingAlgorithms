import statsmodels.api as sm
from Algorithm import Algorithm


class ExponentialSmoothing(Algorithm):

    def fit(self, **kwargs):
        self.data = self.options['endog']
        self.model = sm.tsa.ExponentialSmoothing(**self.options)
        self.results = self.model.fit(**kwargs)




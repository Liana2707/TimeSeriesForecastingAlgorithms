import statsmodels.api as sm

from Algorithm import Algorithm


class Holt(Algorithm):

    def fit(self, **kwargs):
        self.data = self.options['endog']
        self.model = sm.tsa.Holt(**self.options)
        self.results = self.model.fit(**kwargs)

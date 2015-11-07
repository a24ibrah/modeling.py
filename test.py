# -*- coding: utf-8 -*-

__all__ = ["test_gradient"]

import numpy as np

from modeling import ModelMixin, Parameter, parameter_sort


class LinearModel1(ModelMixin):

    m = Parameter()
    b = Parameter()

    def get_value(self, x):
        return self.m * x + self.b

    @parameter_sort
    def get_gradient(self, x):
        return dict(m=x, b=np.ones_like(x))


class LinearModel2(ModelMixin):

    m = Parameter()
    b = Parameter()

    def get_value(self, x):
        return self.m * x + self.b

    @parameter_sort
    def get_value_and_gradient(self, x):
        return self.get_value(x), dict(m=x, b=np.ones_like(x))


def test_gradient(seed=12345):
    np.random.seed(seed)
    x = np.random.randn(5)

    model = LinearModel1(m=0.5, b=10.0)
    assert model.test_gradient(x)

    model = LinearModel2(m=0.5, b=10.0)
    assert model.test_gradient(x)

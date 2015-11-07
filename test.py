# -*- coding: utf-8 -*-

__all__ = ["test_gradient"]

import numpy as np

from modeling import ModelMixin, Parameter, parameter_sort


class LinearModel1(ModelMixin):

    m = Parameter()
    b = Parameter()
    value_count = 0
    grad_count = 0

    def get_value(self, x):
        self.value_count += 1
        return self.m * x + self.b

    @parameter_sort
    def get_gradient(self, x):
        self.grad_count += 1
        return dict(m=x, b=np.ones_like(x))


class LinearModel2(ModelMixin):

    m = Parameter()
    b = Parameter()

    value_count = 0
    grad_count = 0

    def get_value(self, x):
        self.value_count += 1
        return self.m * x + self.b

    @parameter_sort
    def get_value_and_gradient(self, x):
        self.grad_count += 1
        return self.m * x + self.b, dict(m=x, b=np.ones_like(x))


def test_gradient(seed=12345):
    np.random.seed(seed)
    x = np.random.randn(5)

    model = LinearModel1(m=0.5, b=10.0)
    assert np.allclose(model.get_value(x), model.get_value_and_gradient(x)[0])
    assert model.value_count == 2
    assert model.test_gradient(x)
    assert model.grad_count == 3

    model = LinearModel2(m=0.5, b=10.0)
    assert np.allclose(model.get_value(x), model.get_value_and_gradient(x)[0])
    assert model.value_count == 1
    assert model.test_gradient(x)
    assert model.grad_count == 3

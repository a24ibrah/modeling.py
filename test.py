# -*- coding: utf-8 -*-

__all__ = [
    "test_gradient", "test_value_and_gradient",
]

import numpy as np

from modeling import ModelMixin, Parameter, Relationship, parameter_sort


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


class LikelihoodModel(ModelMixin):

    log_sigma2 = Parameter()
    sigma2 = Parameter(depends=[log_sigma2])
    mean_model = Relationship(LinearModel2, scalar=False)

    def __init__(self, y, **kwargs):
        super(LikelihoodModel, self).__init__(**kwargs)
        self.y = y

    def get_value(self, *args, **kwargs):
        mean = self.mean_model[0].get_value(*args, **kwargs)
        s2 = self.sigma2
        return -0.5 * (np.sum((self.y - mean)**2)/s2 + np.log(s2))

    @sigma2.getter
    def get_sigma2(self):
        return np.exp(self.log_sigma2)


def test_gradient(seed=12345):
    np.random.seed(seed)
    x = np.random.randn(5)

    model = LinearModel1(m=0.5, b=10.0)
    assert np.allclose(model.get_value(x), model.get_value_and_gradient(x)[0])
    assert model.value_count == 2
    assert model.test_gradient(x)
    assert model.grad_count == 3


def test_value_and_gradient(seed=12345):
    np.random.seed(seed)
    x = np.random.randn(5)
    model = LinearModel2(m=0.5, b=10.0)

    assert np.allclose(model.get_value(x), model.get_value_and_gradient(x)[0])
    assert model.value_count == 1
    assert model.test_gradient(x)
    assert model.grad_count == 3


def test_relationship(seed=12345):
    np.random.seed(seed)
    x = np.random.randn(5)
    y = 0.5 * x + 1.0
    mean_model = LinearModel2(m=0.5, b=10.0)
    model = LikelihoodModel(y, mean_model=mean_model, log_sigma2=-0.5)
    # model.freeze_parameter("log_sigma2")
    print(model.mean_model)
    print(model.get_value(x))
    print(model.get_vector())
    print(model.get_parameter_names())
    assert 0

    # assert np.allclose(model.get_value(x), model.get_value_and_gradient(x)[0])
    # assert model.value_count == 1
    # assert model.test_gradient(x)
    # assert model.grad_count == 3

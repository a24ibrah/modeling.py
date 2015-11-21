# -*- coding: utf-8 -*-

__all__ = [
    "test_gradient", "test_value_and_gradient", "test_repr", "test_pickle",
]

import pickle
import numpy as np
from collections import Iterable

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


class LikelihoodModel1(ModelMixin):

    log_sigma2 = Parameter()
    sigma2 = Parameter(depends=[log_sigma2])
    mean_model = Relationship(LinearModel2)

    def __init__(self, y, **kwargs):
        super(LikelihoodModel1, self).__init__(**kwargs)
        self.y = y

    def get_value(self, *args):
        mean = self.mean_model.get_value(*args)
        s2 = self.sigma2
        return -0.5 * (np.sum((self.y - mean)**2)/s2 + np.log(s2))

    @parameter_sort
    def get_gradient(self, *args):
        mean, mean_grad = self.mean_model.get_value_and_gradient(*args)
        delta = self.y - mean
        s2 = self.sigma2
        result = dict(zip(map("mean_model:{0}".format,
                          self.mean_model.get_parameter_names()),
                          np.dot(mean_grad, delta) / s2))
        result["log_sigma2"] = 0.5*s2*(np.sum(delta**2)/s2**2 - 1.0/s2)
        return result

    @sigma2.getter
    def get_sigma2(self):
        return np.exp(self.log_sigma2)


class LikelihoodModel2(ModelMixin):

    log_sigma2 = Parameter()
    sigma2 = Parameter(depends=[log_sigma2])
    mean_model = Relationship(LinearModel2, scalar=False)

    def __init__(self, y, **kwargs):
        super(LikelihoodModel2, self).__init__(**kwargs)
        self.y = y

    def get_value(self, *args):
        mean = self.mean_model[0].get_value(*args)
        s2 = self.sigma2
        return -0.5 * (np.sum((self.y - mean)**2)/s2 + np.log(s2))

    @parameter_sort
    def get_gradient(self, *args):
        mean, mean_grad = self.mean_model[0].get_value_and_gradient(*args)
        delta = self.y - mean
        s2 = self.sigma2
        result = dict(zip(map("mean_model(0):{0}".format,
                          self.mean_model[0].get_parameter_names()),
                          np.dot(mean_grad, delta) / s2))
        result["log_sigma2"] = 0.5*s2*(np.sum(delta**2)/s2**2 - 1.0/s2)
        return result

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

    model1 = LikelihoodModel1(y, mean_model=LinearModel2(m=0.5, b=10.0),
                              log_sigma2=-0.5)
    model2 = LikelihoodModel2(y, mean_model=LinearModel2(m=0.5, b=10.0),
                              log_sigma2=-0.5)

    for model in (model1, model2):
        names = model.get_parameter_names()
        model.freeze_parameter("mean_model*")
        assert len(model.get_parameter_names()) == len(names) - 2
        model.thaw_parameter("mean_model*")
        assert model.get_parameter_names() == names

        vector = np.array(model.get_vector())
        if isinstance(model.mean_model, Iterable):
            model.mean_model[0].m = 100.0
        else:
            model.mean_model.m = 100.0
        vector[1] = 100.0
        assert np.allclose(vector, model.get_vector())

        assert model.test_gradient(x)


def test_repr():
    model = LinearModel2(m=0.5, b=10.0)
    assert model.__class__.__name__ == "LinearModel2"


def test_pickle(seed=12340):
    np.random.seed(seed)
    x = np.random.randn(5)
    y = 0.5 * x + 1.0

    mean_model = LinearModel2(m=0.5, b=10.0)
    model = LikelihoodModel1(y, mean_model=mean_model, log_sigma2=-0.5)
    model.freeze_parameter("mean_model:b")
    pkl = pickle.dumps(model)

    loaded = pickle.loads(pkl)
    assert np.allclose(loaded.get_vector(), model.get_vector())
    assert np.allclose(loaded.get_vector(full=True),
                       model.get_vector(full=True))
    assert all(a == b for a, b in zip(loaded.get_parameter_names(),
                                      model.get_parameter_names()))
    assert all(a == b for a, b in zip(loaded.get_parameter_names(full=True),
                                      model.get_parameter_names(full=True)))

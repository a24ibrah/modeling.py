# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = []

from functools import partial
from collections import OrderedDict

from six import add_metaclass
from six import iteritems, itervalues


class Parameter(object):

    def __init__(self,
                 baseclass=None,
                 default=None,
                 required=None,
                 backref=None):
        self.baseclass = baseclass
        self.default = default
        self.required = default is None if required is None else required
        self.backref = backref

        self.model = None
        self.value = None
        self.update = None

        self._getting_function = None
        self._setting_function = None

    def getter(self, f):
        self._getting_function = f

    def setter(self, f):
        self._setting_function = f

    def get(self):
        if self.model is None:
            raise ValueError("we need a model")

        if self.update is not None:
            self._get_update()

        if self.value is None:
            raise ValueError("required parameter not specified")

        return self.value

    def set(self, value):
        self.update = self.default if value is None else value

    def _get_update(self):
        if hasattr(self.update, "__call__"):
            self.value = self.update(self.model)
        else:
            self.value = self.update
        self.update = None

        if self.baseclass is None:
            self.value = float(self.value)
        elif not isinstance(self.value, self.baseclass):
            raise TypeError("the type must be {0}"
                            .format(self.baseclass.__name__))


class ModelMeta(type):

    def __new__(cls, name, parents, dct):
        parameters = OrderedDict()
        for key in sorted(dct.keys()):
            value = dct[key]
            if isinstance(value, Parameter):
                parameters[key] = value
                dct[key] = property(partial(getter, value),
                                    partial(setter, value),
                                    None,
                                    key)
        dct["_parameters"] = parameters
        mdl = type.__new__(cls, name, parents, dct)

        for p in itervalues(parameters):
            p.model = mdl

        return mdl


def getter(parameter, model):
    return parameter.get()


def setter(parameter, model, value):
    return parameter.set(value)


@add_metaclass(ModelMeta)
class Model(object):

    dude = Parameter()
    blah = Parameter()

    def __init__(self, **kwargs):
        for k, v in iteritems(kwargs):
            setattr(self, k, v)

    @property
    def parameters(self):
        return self._parameters


class Body(Model):

    # Base parameters.
    radius = Parameter()
    t0 = Parameter()
    a = Parameter()
    di = Parameter()
    e = Parameter()
    pomega = Parameter()

    # Re-parameterization.
    period = Parameter()
    b = Parameter()

    @period.getter
    def period_getter(self):
        return 505.

    @period.setter
    def period_setter(self, value):
        print("setter", self, value)


class Star(Model):

    bodies = ParameterVector(Body, default=[], backref="star")
    mass = Parameter()
    radius = Parameter()


m = Body()
print(m.parameters)

# -*- coding: utf-8 -*-
"""
A flexible framework for building models for all your data analysis needs.

"""

__all__ = ["Parameter", "ModelMixin"]
__version__ = "0.0.1.dev0"
__author__ = "Daniel Foreman-Mackey (foreman.mackey@gmail.com)"
__copyright__ = "Copyright 2015 Daniel Foreman-Mackey"
__contributors__ = [
    # Alphabetical by first name.
]

import copy
import logging
import fnmatch
from collections import OrderedDict

import numpy as np
from six import with_metaclass, iteritems


class Parameter(object):
    """
    Instances of this class provide the parameter specification to custom
    models.

    :param size: (optional)
        For vector parameters, ``size`` is the length of the vector. For
        scalar parameters, keep the default value of ``size=0``.

    :param default: (optional)
        The default value that the parameter should take if it isn't
        specified. This can be a float or an array of size ``size``.

    :param bounds: (optional)
        A tuple or list of tuples specifying the bounds of the parameter. For
        a vector parameter, a single tuple can be given and the bounds will
        be assumed to be equivalent along each axis. Otherwise, the list of
        bounds must include ``size`` tuples. To indicate no limit in one
        direction, use ``None``. For example, for a strictly positive
        parameter, you could use ``bounds=(0, None)``.

    """

    # This global counter keeps track of the order that parameters are added
    # to models. The point of this is to force a model to always have a
    # deterministic parameter order. In practice, the ordering is enforced by
    # the ``ModelMeta`` metaclass.
    _creation_counter = 0

    def __init__(self, size=0, default=None, bounds=None, frozen=False):
        self._creation_order = Parameter._creation_counter
        Parameter._creation_counter += 1

        self.default = default
        self.bounds = bounds
        self.size = size
        self.frozen = frozen

        # Check the bounds.
        if bounds is None:
            self.bounds = [(None, None) for i in range(max(1, size))]
        else:
            shape = np.shape(bounds)
            if shape == (2, ):
                self.bounds = [bounds for i in range(max(1, size))]
            elif shape != (max(1, size), 2):
                raise ValueError("invalid bounds; must have shape (size, 2) "
                                 "or (2,) not {0}".format(shape))

        # Check the default vector.
        if size > 0 and default is not None:
            try:
                float(default)
            except TypeError:
                if len(default) != size:
                    raise ValueError("invalid dimensions for default vector")

    def __len__(self):
        return self.size

    def __repr__(self):
        args = ", ".join(map("{0}={{0.{0}}}".format,
                             ("default", "bounds", "size")))
        return "Parameter({0})".format(args.format(self))

    def get_default(self):
        """
        Get the default value for the parameter. If no default was defined,
        this will return ``np.nan`` or an array of ``np.nan`` and with
        correct shape.

        """
        if len(self):
            if self.default is None:
                return np.nan + np.zeros(len(self))
            return self.default + np.zeros(len(self))
        if self.default is None:
            return np.nan
        return self.default


class ModelMeta(type):
    """
    This metaclass is used to unpack and parse the parameter specification of
    a new model on creation. The main thing that it does is pop the
    ``Parameter`` attributes from the object's ``__dict__`` and converts them
    into an ``OrderedDict`` (ordered by ``Parameter._creation_order``) that
    is subsequently added to the namespace as ``__parameters__``. This allows
    the model to have a well-defined parameter order.

    They always say: "if you need metaprogramming, you'll know"...

    """

    def __new__(cls, name, parents, orig_dct):
        dct = dict(orig_dct)

        # Loop over the members of the class and find all the `Parameter`s.
        # These will form the basis of the modeling protocol by exposing the
        # parameter names and other available settings.
        parameters = []
        for name, obj in iteritems(dct):
            if isinstance(obj, Parameter):
                parameters.append((name, copy.deepcopy(obj)))

        # The parameters are ordered by their creation order (i.e. the order
        # that they appear in the class definition) so that the `vector` will
        # be deterministic.
        dct["__parameters__"] = parameters = OrderedDict(
            sorted(parameters, key=lambda o: o[1]._creation_order)
        )

        # Remove the parameter definitions from the namespace so they can be
        # overwritten.
        for k in parameters:
            dct.pop(k)

        return super(ModelMeta, cls).__new__(cls, name, parents, dct)


class ModelMixin(with_metaclass(ModelMeta, object)):
    """
    An abstract base class implementing most of the modeling functionality.
    To implement a custom model, inherit from this import class, list
    parameters using ``Parameter`` objects and implement the ``get_value``
    method. When initializing the custom model, the default ``__init__``
    takes your parameter values as keyword arguments and it will check to
    make sure that you completely specified the model.

    """

    def __new__(cls, *args, **kwargs):
        # On creation of a new model, we use this method to make sure that
        # all the necessary memory is allocated and all the bookkeeping
        # arrays are in place.
        self = super(ModelMixin, cls).__new__(cls)

        # Preallocate space for the vector and compute how each parameter
        # will index into the vector.
        count = 0
        vector = []
        for k, o in iteritems(self.__parameters__):
            if len(o):
                o.index = slice(count, count + len(o))
                count += len(o)
            else:
                o.index = count
                count += 1
            vector.append(np.atleast_1d(o.get_default()))
        self._vector = np.concatenate(vector)

        # Keep track of which parameters are frozen or thawed and save the
        # parameter bounds.
        self._frozen = np.zeros(count, dtype=bool)
        self._bounds = []
        for k, o in iteritems(self.__parameters__):
            self._frozen[o.index] = o.frozen
            self._bounds += o.bounds

        return self

    def __init__(self, **kwargs):
        for k, v in iteritems(kwargs):
            if k not in self.__parameters__:
                raise ValueError("unrecognized parameter '{0}'".format(k))
            setattr(self, k, v)

        # Check to make sure that all the parameters were given either with
        # a default value or in this initialization.
        if np.any(np.isnan(self._vector)):
            pars = []
            for k, o in iteritems(self.__parameters__):
                if np.any(np.isnan(self._vector[o.index])):
                    pars.append(k)
            raise ValueError(
                ("missing values for parameters: {0}. Use a default parameter "
                 "value or specify a value in the initialization of the model")
                .format(pars)
            )

    def __len__(self):
        return len(self._frozen) - sum(self._frozen)

    def __repr__(self):
        args = ", ".join(map("{0}={{0.{0}}}".format,
                             self.__parameters__.keys()))
        return "{0}({1})".format(self.__class__.__name__, args.format(self))

    def __getattr__(self, name):
        if self.__parameters__ is None or name not in self.__parameters__:
            raise AttributeError(name)

        o = self.__parameters__.get(name, None)
        return self._vector[o.index]

    def __setattr__(self, name, value):
        if self.__parameters__ is None or name not in self.__parameters__:
            return super(ModelMixin, self).__setattr__(name, value)

        o = self.__parameters__.get(name, None)
        self._vector[o.index] = value

    def __getitem__(self, k):
        return self.get_parameter(k)

    def __setitem__(self, k, v):
        return self.set_parameter(k, v)

    def __call__(self, vector, *args, **kwargs):
        """
        Evaluate the model at a given parameter vector. Any other arguments
        are passed directly to ``get_value``.

        :param vector:
            The vector of parameters where you would like to evaluate the
            model.

        """
        vector0 = np.array(self._vector)
        self.set_vector(vector)
        value = self.get_value(*args, **kwargs)
        self._vector = vector0
        return value

    def get_vector(self, full=False):
        if full:
            return self._vector
        return self._vector[~self._frozen]

    def set_vector(self, value, full=False):
        if full:
            self._vector[:] = value
        else:
            self._vector[~self._frozen] = value

    @property
    def vector(self):
        return self.get_vector()

    @vector.setter
    def vector(self, value):
        self.set_vector(value)

    def check_vector(self, vector, full=False):
        bounds = self.get_bounds()
        if len(bounds) != len(vector):
            raise ValueError("dimension mismatch")
        for i, (a, b) in enumerate(bounds):
            v = vector[i]
            if (a is not None and v < a) or (b is not None and b < v):
                return False
        return True

    def get_bounds(self, full=False):
        if full:
            return self._bounds
        return [b for i, b in enumerate(self._bounds) if not self._frozen[i]]

    def get_parameter_names(self, full=False):
        names = []
        for k, o in iteritems(self.__parameters__):
            if len(o):
                names += ["{0}({1})".format(k, i) for i in range(len(o))]
            else:
                names.append(k)
        if full:
            return names
        return [n for i, n in enumerate(names) if not self._frozen[i]]

    @property
    def parameter_names(self):
        return self.get_parameter_names()

    def match_parameter(self, name):
        # Return fast for exact matches.
        if name in self.__parameters__:
            return self.__parameters__[name].index

        # Search the parameter name list for matches.
        inds = []
        for i, n in enumerate(self.get_parameter_names(full=True)):
            if n == name:
                return i
            if not fnmatch.fnmatch(n, name):
                continue
            inds.append(i)
        if not len(inds):
            raise KeyError("unknown parameter '{0}'".format(name))
        return inds

    def get_parameter(self, name):
        i = self.match_parameter(name)
        v = self._vector[i]
        try:
            return float(v)
        except TypeError:
            return v

    def set_parameter(self, name, value):
        i = self.match_parameter(name)
        self._vector[i] = value

    def freeze_parameter(self, name):
        i = self.match_parameter(name)
        self._frozen[i] = True

    def thaw_parameter(self, name):
        i = self.match_parameter(name)
        self._frozen[i] = False

    @staticmethod
    def parameter_sort(f):
        if f.__name__ == "get_value_and_gradient":
            def func(self, *args, **kwargs):
                value, gradient = f(self, *args, **kwargs)
                return value, self._sort(gradient)
        elif f.__name__ == "get_gradient":
            def func(self, *args, **kwargs):
                return self._sort(f(self, *args, **kwargs))
        return func

    def _sort(self, values):
        ret = [values[k] for k in self.get_parameter_names()]
        # Horrible hack to only return numpy array if that's what was
        # given by the wrapped function.
        if len(ret) and type(ret[0]).__module__ == np.__name__:
            return np.vstack(ret)
        return ret

    def get_value(self, *args, **kwargs):
        raise NotImplementedError("sublasses must implement the 'get_value' "
                                  "method")

    def get_gradient(self, *args, **kwargs):
        if hasattr(self, "get_value_and_gradient"):
            return self.get_value_and_gradient(*args, **kwargs)[1]

        logging.warn("using default numerical gradients; this might be slow "
                     "and numerically unstable")
        return self.get_numerical_gradient(*args, **kwargs)

    def get_numerical_gradient(self, *args, **kwargs):
        eps = kwargs.pop("_modeling_eps", 1.245e-5)
        vector = self.get_vector()
        value0 = self.get_value(*args, **kwargs)
        grad = np.empty([len(vector)] + list(value0.shape), dtype=np.float64)
        for i, v in enumerate(vector):
            vector[i] = v + eps
            self.set_vector(vector)
            value = self.get_value(*args, **kwargs)
            vector[i] = v
            self.set_vector(vector)
            grad[i] = (value - value0) / eps
        return grad


class Model(ModelMixin):

    amp = Parameter(frozen=True, default=0.1, bounds=(-10, 10))
    mu = Parameter(size=3)
    log_sigma = Parameter()

    def get_value(self, x):
        inv_sig = np.exp(-self.log_sigma)
        return self.amp*np.exp(-np.sum((x-self.mu)**2)*inv_sig)

    @ModelMixin.parameter_sort
    def get_value_and_gradient(self, x):
        value = self.get_value(x)
        return value, dict(
            amp=value / self.amp,
            mu=value
        )


if __name__ == "__main__":
    m = Model(log_sigma=1.0, mu=0.0)
    print(m)
    print(m.get_bounds())
    print(m.get_vector())
    print(m.get_parameter_names(full=True))
    m.thaw_parameter("amp")

    print(m.get_value(0.1))

    m2 = Model(log_sigma=10.0, mu=10.0)
    print(m2._vector)
    print(m2.log_sigma)

    print(m._vector)
    print(m.log_sigma)

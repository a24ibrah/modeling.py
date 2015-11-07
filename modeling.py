# -*- coding: utf-8 -*-
"""
A flexible framework for building models for all your data analysis needs.

"""

__all__ = ["Parameter", "ModelMixin", "parameter_sort"]
__version__ = "0.0.1.dev0"
__author__ = "Daniel Foreman-Mackey (foreman.mackey@gmail.com)"
__copyright__ = "Copyright 2015 Daniel Foreman-Mackey"
__contributors__ = [
    # Alphabetical by first name.
]

import copy
import logging
import fnmatch
from collections import OrderedDict, Iterable

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
    __creation_counter__ = 0

    def __init__(self, name=None, size=0, default=None, bounds=None,
                 frozen=False, depends=None):
        self._creation_order = Parameter.__creation_counter__
        Parameter.__creation_counter__ += 1

        self.name = name
        self.default = default
        self.bounds = bounds
        self.size = size
        self.frozen = frozen

        # Build the dependency tree.
        self.dependents = []
        if depends is not None:
            if not isinstance(depends, Iterable):
                depends = [depends]
        self.depends = depends

        # Caching.
        self.value = None

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
                             ("name", "default", "bounds", "size", "frozen",
                              "depends")))
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

    def add_dependent(self, param):
        self.dependents.append(param)

    def getter(self, func):
        self._getter = func

    def setter(self, func):
        self._setter = func

    def reset(self):
        self.value = None
        [p.reset() for p in self.dependents]

    @property
    def has_getter(self):
        return hasattr(self, "_getter")

    def get_value(self, model):
        if self.depends is None or self.value is None:
            self.value = self._getter(model)
        return self.value

    @property
    def has_setter(self):
        return hasattr(self, "_setter")

    def set_value(self, model, value):
        self.value = value
        self._setter(model, value)


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
                obj.name = name
                if not obj.has_getter:
                    parameters.append((name, obj))

        # This copy must be after going through the full dict so that we can
        # reconstruct the dependency tree. There must be a better way but
        # this works!
        parameters = [(name, copy.deepcopy(obj)) for name, obj in parameters]

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

        # Now we'll do the same and loop through the reparameterizations. For
        # each reparameterization, update the parent parameters with a link
        # to the correct (copied) parameter.
        reparams = []
        for name, obj in iteritems(dct):
            if isinstance(obj, Parameter):
                reparams.append((name, obj))
        reparams = [(name, copy.deepcopy(obj)) for name, obj in reparams]
        dct["__reparams__"] = reparams = OrderedDict(
            sorted(reparams, key=lambda o: o[1]._creation_order)
        )
        for k in reparams:
            dct.pop(k)

        # Build the dependency tree linking the parents to their children to
        # keep track of everything.
        for name, obj in iteritems(reparams):
            if obj.depends is None:
                continue
            for o in obj.depends:
                parent = parameters.get(o.name)
                parent = reparams[o.name] if parent is None else parent
                parent.add_dependent(obj)

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
            if k not in self.__parameters__ and k not in self.__reparams__:
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
        if self.__parameters__ is None:
            raise AttributeError(name)

        # This is a hack to make sure that a user implemented gradient will
        # be preferred to the default numerical gradients. The reason why
        # this is needed is that I want to allow the user to implement
        # 'get_gradient_and_value' and/or 'get_gradient'.
        if name == "get_gradient":
            if hasattr(self, "get_value_and_gradient"):
                def f(*args, **kwargs):
                    return self.get_value_and_gradient(*args, **kwargs)[1]
            else:
                f = self._get_gradient
            return f
        if name == "get_value_and_gradient":
            return self._get_value_and_gradient

        # Return the correct parameter or reparameterization.
        o = self.__parameters__.get(name, None)
        if o is None:
            if name not in self.__reparams__:
                raise AttributeError(name)
            return self.__reparams__[name].get_value(self)
        return self._vector[o.index]

    def __setattr__(self, name, value):
        if self.__parameters__ is None:
            return super(ModelMixin, self).__setattr__(name, value)

        o = self.__parameters__.get(name, None)
        if o is None:
            o = self.__reparams__.get(name, None)
            if o is None:
                return super(ModelMixin, self).__setattr__(name, value)
            if not o.has_setter:
                raise AttributeError("'{0}' is read-only".format(name))
            o.set_value(self, value)
        else:
            self._vector[o.index] = value

        # Make sure that any dependent parameters get reset.
        [p.reset() for p in o.dependents]

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
        self.set_vector(vector0)
        return value

    def get_vector(self, full=False):
        """
        Get the current parameter vector.

        :param full: (optional)
            If ``True``, return the "full" vector, not just the currently
            thawed parameters. (default: ``False``)

        """
        if full:
            return self._vector
        return self._vector[~self._frozen]

    def set_vector(self, value, full=False):
        """
        Set the current parameter vector.

        :param full: (optional)
            If ``True``, the "full" vector will be updated, not just the
            currently thawed parameters. (default: ``False``)

        """
        [p.reset() for p in self.__reparams__.values()]
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

    def get_bounds(self, full=False):
        """
        Get the bounds of the parameter space. This will be a list of tuples
        in a format compatible with ``scipy.optimize.minimize``.

        :param full: (optional)
            If ``True``, the "full" vector will be checked, not just the
            currently thawed parameters. (default: ``False``)

        """
        if full:
            return self._bounds
        return [b for i, b in enumerate(self._bounds) if not self._frozen[i]]

    def check_vector(self, vector, full=False):
        """
        Returns ``True`` if a proposed vector is within the allowed bounds of
        parameter space. If a ``validate`` method is implemented, it will
        also be tested. Note: if you implement a validate method, the
        parameters will have to be updated (i.e. you'll lose any cached
        values) but then reset to the current value.

        :param vector:
            The proposed parameter vector.

        :param full: (optional)
            If ``True``, the "full" vector will be checked, not just the
            currently thawed parameters. (default: ``False``)

        """
        # Check the bounds first.
        bounds = self.get_bounds(full=full)
        if len(bounds) != len(vector):
            raise ValueError("dimension mismatch")
        for i, (a, b) in enumerate(bounds):
            v = vector[i]
            if (a is not None and v < a) or (b is not None and b < v):
                return False

        # If implemented, call the 'validate' method and check the output.
        flag = True
        if hasattr(self, "validate"):
            vector0 = np.array(self.get_vector(full=full))
            self.set_vector(vector, full=full)
            flag = self.validate()
            self.set_vector(vector0, full=full)

        return flag

    def get_parameter_names(self, full=False):
        """
        Get the list of parameter names. This only includes base parameters
        (not reparameterizations) and it will be in the same order as the
        parameter vector.

        :param full: (optional)
            If ``True``, get all the parameter names, not just the
            currently thawed parameters. (default: ``False``)

        """
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
            p = self.__parameters__[name]
            return p.index, [p]

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
        return inds, [self.__parameters__[i] for i in inds]

    def get_parameter(self, name):
        i, _ = self.match_parameter(name)
        v = self._vector[i]
        try:
            return float(v)
        except TypeError:
            return v

    def set_parameter(self, name, value):
        i, params = self.match_parameter(name)
        self._vector[i] = value
        [p.reset() for p in params]

    def freeze_parameter(self, name):
        i, _ = self.match_parameter(name)
        self._frozen[i] = True

    def thaw_parameter(self, name):
        i, _ = self.match_parameter(name)
        self._frozen[i] = False

    def get_value(self, *args, **kwargs):
        raise NotImplementedError("sublasses must implement the 'get_value' "
                                  "method")

    def _get_value_and_gradient(self, *args, **kwargs):
        return (
            self.get_value(*args, **kwargs),
            self.get_gradient(*args, **kwargs)
        )

    def _get_gradient(self, *args, **kwargs):
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

    def test_gradient(self, *args, **kwargs):
        return (np.allclose(
            self.get_gradient(*args, **kwargs),
            self.get_numerical_gradient(*args, **kwargs),
        ) and np.allclose(
            self.get_value_and_gradient(*args, **kwargs)[1],
            self.get_numerical_gradient(*args, **kwargs),
        ))

    def _sort(self, values):
        return np.array([values[k] for k in self.get_parameter_names()])


def parameter_sort(f):
    """
    A decorator used to sort a dictionary of outputs from a model method into
    the correct parameter order. This will generally but useful for sorting
    the output from a method computing model gradients.

    Here's an example for how you might use it:

    .. code-block:: python

        class LinearModel(ModelMixin):

            m = Parameter()
            b = Parameter()

            def get_value(self, x):
                return self.m * x + self.b

            @parameter_sort
            def get_gradient(self, x):
                return dict(m=x, b=np.ones_like(x))

    """

    def func(self, *args, **kwargs):
        # Call the method.
        values = f(self, *args, **kwargs)

        # Try to sort the output. This will work if the method just outputs
        # a single dictionary.
        try:
            return self._sort(values)
        except KeyError:
            raise ValueError("a method wrapped by 'parameter_sort' must "
                             "return a dictionary with a key for each "
                             "parameter in: {0}"
                             .format(self.get_parameter_names()))
        except TypeError:
            pass

        # If we get here, the output is more complicated. If it is iterable,
        # we'll check each element to see if it has the right format of
        # dictionary.
        if not isinstance(values, Iterable):
            raise ValueError("invalid output for a method wrapped by "
                             "'parameter_sort'")
        any_ = False
        ret = []
        for v in values:
            # For each element in the output, try to sort it.
            try:
                ret.append(self._sort(v))
            except (KeyError, TypeError, IndexError):
                ret.append(v)
            else:
                any_ = True
        if not any_:
            raise ValueError("a method wrapped by 'parameter_sort' must "
                             "return at least one dictionary with a key for "
                             "each parameter in: {0}"
                             .format(self.get_parameter_names()))

        return tuple(ret)
    return func


class Model1(ModelMixin):

    amp = Parameter(frozen=True, default=0.1, bounds=(-10, 10))
    mu = Parameter(size=3)
    log_sigma = Parameter(default=1)

    # Reparameterizations.
    sigma = Parameter(default=2, depends=log_sigma)
    inv_sigma = Parameter(default=3, depends=sigma)

    def get_value(self, x):
        return self.amp*np.exp(-np.sum((x-self.mu)**2)*self.inv_sigma)

    @sigma.getter
    def get_sigma(self):
        return np.exp(self.log_sigma)

    @sigma.setter
    def set_sigma(self, value):
        self.log_sigma = np.log(value)

    @inv_sigma.getter
    def get_inv_sigma(self):
        return 1.0 / self.sigma

    @parameter_sort
    def get_value_and_gradient(self, x):
        value = self.get_value(x)
        return value, dict(amp=0.0, mu=1.0, log_sigma=0.0)


if __name__ == "__main__":
    m = Model1(sigma=1.0, mu=0.0)
    print(m.log_sigma)
    print(m.sigma)
    print(m.inv_sigma)
    print(m.get_value(0.1))

    m.sigma = 10.0
    print(m.log_sigma)
    print(m.sigma)
    print(m.inv_sigma)
    print(m.get_value(0.1))

    # print(np.exp(-m.log_sigma))
    # print(m.get_parameter_names())

    # print(m)
    # print(m.get_bounds())
    # print(m.get_vector())
    # print(m.get_parameter_names(full=True))
    m.thaw_parameter("amp")

    # m2 = Model(log_sigma=10.0, mu=.5)
    # print(m2._vector)
    # print(m2.log_sigma)

    # print(m2.get_value(0.1))

    print(m.get_gradient(0.1))

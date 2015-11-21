# -*- coding: utf-8 -*-
"""
A flexible framework for building models for all your data analysis needs.

"""

__all__ = ["Parameter", "Relationship", "ModelMixin", "parameter_sort"]
__version__ = "0.0.1.dev0"
__author__ = "Daniel Foreman-Mackey"
__copyright__ = "Copyright 2015 Daniel Foreman-Mackey"
__contributors__ = [
    # Alphabetical by first name.
]

import re
import copy
import logging
import fnmatch
from collections import OrderedDict, Iterable

import numpy as np
from six import with_metaclass, iteritems


_prefix_re = re.compile("(.+?)(?:\((.*?)\))?:(.*)")


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


class Relationship(object):

    # As with parameters, we need to keep track of the order that
    # relationships are created.
    __creation_counter__ = 0

    def __init__(self, model=None, name=None, scalar=True, strict=None):
        self._creation_order = Relationship.__creation_counter__
        Relationship.__creation_counter__ += 1

        self.name = name
        self.model = model
        self.scalar = scalar
        if strict is None:
            strict = model is not None
        self.strict = strict

    def __len__(self):
        return len(self.model)

    def __repr__(self):
        args = ", ".join(map("{0}={{0.{0}}}".format,
                             ("model", "name", "scalar", "strict")))
        return "Relationship({0})".format(args.format(self))


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

    def __new__(cls, cls_name, parents, orig_dct):
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

        # Now we'll do the same and loop through and find the
        # reparameterizations and relationships.
        reparams = []
        relationships = []
        for name, obj in iteritems(dct):
            if isinstance(obj, Parameter):
                reparams.append((name, copy.deepcopy(obj)))
            elif isinstance(obj, Relationship):
                relationships.append((name, copy.deepcopy(obj)))

        dct["__reparams__"] = reparams = OrderedDict(
            sorted(reparams, key=lambda o: o[1]._creation_order)
        )
        [dct.pop(k) for k in reparams]

        dct["__relationships__"] = relationships = OrderedDict(
            sorted(relationships, key=lambda o: o[1]._creation_order)
        )
        [dct.pop(k) for k in relationships]

        # Build the dependency tree linking the parents to their children to
        # keep track of everything.
        for name, obj in iteritems(reparams):
            if obj.depends is None:
                continue
            for o in obj.depends:
                parent = parameters.get(o.name)
                parent = reparams[o.name] if parent is None else parent
                parent.add_dependent(obj)

        self = super(ModelMeta, cls).__new__(cls, cls_name, parents, dct)
        return self


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
        if len(vector):
            self._vector = np.concatenate(vector)
        else:
            self._vector = np.empty(0)

        # Make space for the relationships.
        self._relationships = OrderedDict(
            (k, None) if o.scalar else (k, [])
            for k, o in iteritems(self.__relationships__)
        )

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
            if (k not in self.__parameters__ and k not in self.__reparams__
                    and k not in self.__relationships__):
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
                             list(self.__parameters__.keys())
                             + list(self.__relationships__.keys())))
        return "{0}({1})".format(self.__class__.__name__, args.format(self))

    def __getattr__(self, name):
        if self.__parameters__ is None:
            raise AttributeError(name)

        # This is a hack to make sure that a user implemented gradient will
        # be preferred to the default numerical gradients. The reason why
        # this is needed is that I want to allow the user to implement
        # 'get_gradient_and_value' and/or 'get_gradient'.
        if name == "get_gradient":
            if "get_value_and_gradient" in dir(self):
                def f(*args, **kwargs):
                    return self.get_value_and_gradient(*args, **kwargs)[1]
            else:
                f = self._get_gradient
            return f
        if name == "get_value_and_gradient":
            return self._get_value_and_gradient

        # First check relationships.
        if name in self.__relationships__:
            return self._relationships[name]

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

        # First check relationships.
        if name in self.__relationships__:
            rel = self.__relationships__[name]
            if rel.strict and not isinstance(value, rel.model):
                logging.warn("incompatible type for '{0}'".format(name))
            if not rel.scalar and not isinstance(value, Iterable):
                value = [value]
            self._relationships[name] = value
            return

        # Then parameters/reparameterizations.
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
        return self.get_parameter(k, unpack=False)

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
        vector0 = np.array(self.get_vector())
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
        # Get the base vector for this model.
        if full:
            v = self._vector
        else:
            v = self._vector[~self._frozen]

        # Then append the vectors for all the relationships.
        v = [v]
        for k, rel in iteritems(self.__relationships__):
            o = self._relationships[k]
            if rel.scalar:
                v.append(o.get_vector(full=full))
            else:
                v += [m.get_vector(full=full) for m in o]

        return np.concatenate(v)

    def set_vector(self, value, full=False):
        """
        Set the current parameter vector.

        :param full: (optional)
            If ``True``, the "full" vector will be updated, not just the
            currently thawed parameters. (default: ``False``)

        """
        [p.reset() for p in self.__reparams__.values()]
        if full:
            n = len(self._vector)
            self._vector[:] = value[:n]
        else:
            n = len(self)
            self._vector[~self._frozen] = value[:n]

        # Then append the vectors for all the relationships.
        for k, rel in iteritems(self.__relationships__):
            o = self._relationships[k]
            if rel.scalar:
                d = len(o)
                o.set_vector(value[n:n+d], full=full)
                n += d
            else:
                for m in o:
                    d = len(m)
                    m.set_vector(value[n:n+d], full=full)
                    n += d

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
            bounds = self._bounds
        else:
            bounds = [b for i, b in enumerate(self._bounds)
                      if not self._frozen[i]]

        for k, rel in iteritems(self.__relationships__):
            o = self._relationships[k]
            if rel.scalar:
                o = [o]
            for m in o:
                bounds += m.get_bounds(full=full)

        return bounds

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
            if flag:
                # Loop over the relationships and validate each sub-model.
                for k, rel in iteritems(self.__relationships__):
                    m = self._relationships[k]
                    if rel.scalar:
                        m = [m]
                    for o in m:
                        if hasattr(o, "validate"):
                            flag &= o.validate()

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
        if not full:
            names = [n for i, n in enumerate(names) if not self._frozen[i]]
        for k, o in iteritems(self.__relationships__):
            rel = self._relationships[k]
            if o.scalar:
                names += ["{0}:{1}".format(k, n)
                          for n in rel.get_parameter_names(full=full)]
            else:
                names += ["{0}({2}):{1}".format(k, n, i)
                          for i, r in enumerate(rel)
                          for n in r.get_parameter_names(full=full)]
        return names

    @property
    def parameter_names(self):
        return self.get_parameter_names()

    def match_parameter(self, name):
        """
        Match a parameter name or pattern against the parameters and
        relationships of this model.

        Parameter names or patterns are matched using the `fnmatch module
        <https://docs.python.org/3.5/library/fnmatch.html>`_.  This means that
        you can use wild cards like ``*`` or ``?`` as described in the module
        documentation. The match will be done against the ``full`` parameter
        list; even frozen parameters will be be returned.

        :param name:
            The parameter name or pattern.

        :returns parameter_names:
            An array of indices into the full ``vector`` giving the locations
            of the parameters.

        :returns parameter_list:
            A list of matched parameter names. This has the same length and
            order as ``inds``.

        :returns relationships:
            A list of elements of the form ``(relationship, parameter)`` where
            ``relationship`` is a ``Relationship`` matched by the query and
            ``parameter`` is the name of the parameter for that sub-model.

        """
        # Search the parameter name list for matches.
        inds = []
        names = []
        relationships = []
        for i, n in enumerate(self.get_parameter_names(full=True)):
            if not fnmatch.fnmatch(n, name):
                continue

            inds.append(i)
            names.append(n)

            prefix = _prefix_re.findall(n)
            if not len(prefix):
                continue

            nm, ind, param = prefix[0]
            rel = self.__relationships__[nm]
            if rel.scalar:
                relationships.append((self._relationships[nm], param))
            else:
                relationships.append((self._relationships[nm][int(ind)],
                                      param))

        if not len(inds):
            raise KeyError("unknown parameter '{0}'".format(name))
        return inds, names, relationships

    def get_parameter(self, name, unpack=True):
        """
        Get the value of a parameter or set of parameters associated with a
        given name or pattern as matched by ``match_parameter``.

        :param name:
            The parameter name or pattern.

        :param unpack: (optional)
            If ``True``, the result will be a dictionary where the matched
            parameter names are the keys. Otherwise, the result is an array
            with the matched parameters in the correct order. (default:
            ``True``)

        """
        inds, names, _ = self.match_parameter(name)
        v = self.get_vector(full=True)[inds]
        if unpack:
            return dict(zip(names, v))
        return v

    def set_parameter(self, name, value):
        """
        Set the value of a parameter or set of parameters associated with a
        given name or pattern as matched by ``match_parameter``.

        :param name:
            The parameter name or pattern.

        :param value:
            This can be a ``dict`` with all the matched parameter names as
            keys or a value that can be assigned to the matched values.

        """
        inds, names, _ = self.match_parameter(name)
        v = self.get_vector(full=True)
        try:
            for nm, val in iteritems(value):
                v[inds[names.index(nm)]] = val
        except AttributeError:
            v[inds] = value
        self.set_vector(v, full=True)

    def freeze_parameter(self, name):
        """
        Freeze the parameter or set of parameters associated with a given name
        or pattern as matched by ``match_parameter``.

        :param name:
            The parameter name or pattern.

        """
        # Freeze the local model parameters first.
        n = len(self._vector)
        inds, names, relationships = self.match_parameter(name)
        self._frozen[list(i for i in inds if i < n)] = True

        # Then propagate across the relationships.
        for model, attr in relationships:
            model.freeze_parameter(attr)

    def thaw_parameter(self, name):
        """
        Thaw the parameter or set of parameters associated with a given name
        or pattern as matched by ``match_parameter``.

        :param name:
            The parameter name or pattern.

        """
        # Freeze the local model parameters first.
        n = len(self._vector)
        inds, names, relationships = self.match_parameter(name)
        self._frozen[list(i for i in inds if i < n)] = False

        # Then propagate across the relationships.
        for model, attr in relationships:
            model.thaw_parameter(attr)

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
        eps = kwargs.pop("_modeling_eps", 1.245e-8)
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
        grad2 = self.get_numerical_gradient(*args, **kwargs)
        kwargs.pop("_modeling_eps", None)
        grad1 = self.get_gradient(*args, **kwargs)
        flag = np.allclose(grad1, grad2) and np.allclose(
            self.get_value_and_gradient(*args, **kwargs)[1], grad2
        )
        if not flag:
            logging.warn("{0} != {1}".format(grad1, grad2))
        return flag

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

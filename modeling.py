import logging
import fnmatch
from collections import OrderedDict

import numpy as np
from six import with_metaclass, iteritems


class Parameter(object):

    _creation_counter = 0

    def __init__(self, default=None, bounds=None, length=0, frozen=False):
        self._creation_order = Parameter._creation_counter
        Parameter._creation_counter += 1

        self.default = default
        self.bounds = bounds
        self.length = length
        self.frozen = frozen

        # Check the bounds.
        if bounds is None:
            self.bounds = [(None, None) for i in range(max(1, length))]
        else:
            shape = np.shape(bounds)
            if shape == (2, ):
                self.bounds = [bounds for i in range(max(1, length))]
            elif shape != (max(1, length), 2):
                raise ValueError("invalid bounds; must have shape (length, 2) "
                                 "or (2,) not {0}".format(shape))

        # Check the default vector.
        if length > 0 and default is not None:
            try:
                float(default)
            except TypeError:
                if len(default) != length:
                    raise ValueError("invalid dimensions for default vector")

    def __len__(self):
        return self.length

    def __repr__(self):
        args = ", ".join(map("{0}={{0.{0}}}".format,
                             ("default", "bounds", "length")))
        return "Parameter({0})".format(args.format(self))

    def get_default(self):
        if len(self):
            if self.default is None:
                return np.nan + np.zeros(len(self))
            return self.default + np.zeros(len(self))
        if self.default is None:
            return np.nan
        return self.default


class ModelMeta(type):

    def __new__(cls, name, parents, dct):
        dct = dict(dct)

        # Loop over the members of the class and find all the `Parameter`s.
        # These will form the basis of the modeling protocol by exposing the
        # parameter names and other available settings.
        parameters = []
        for name, obj in iteritems(dct):
            if isinstance(obj, Parameter):
                parameters.append((name, obj))

        # The parameters are ordered by their creation order (i.e. the order
        # that they appear in the class definition) so that the `vector` will
        # be deterministic.
        dct["_parameters"] = parameters = OrderedDict(
            sorted(parameters, key=lambda o: o[1]._creation_order)
        )

        # Remove the parameter definitions from the namespace so they can be
        # overwritten.
        for k in parameters:
            dct.pop(k)

        return super(ModelMeta, cls).__new__(cls, name, parents, dct)


class ModelMixin(with_metaclass(ModelMeta, object)):

    def __new__(cls, *args, **kwargs):
        self = super(ModelMixin, cls).__new__(cls)

        # Preallocate space for the vector and compute how each parameter
        # will index into the vector.
        count = 0
        vector = []
        for k, o in iteritems(self._parameters):
            if len(o):
                o.index = slice(count, count + len(o))
                count += len(o)
            else:
                o.index = count
                count += 1
            vector.append(np.atleast_1d(o.get_default()))
        self._length = count
        self._vector = np.concatenate(vector)

        # Keep track of which parameters are frozen or thawed and save the
        # parameter bounds.
        self._frozen = np.zeros(count, dtype=bool)
        self._bounds = []
        for k, o in iteritems(self._parameters):
            self._frozen[o.index] = o.frozen
            self._bounds += o.bounds

        return self

    def __repr__(self):
        args = ", ".join(map("{0}={{0.{0}}}".format, self._parameters.keys()))
        return "{0}({1})".format(self.__class__.__name__, args.format(self))

    def __init__(self, **kwargs):
        for k, v in iteritems(kwargs):
            if k not in self._parameters:
                raise ValueError("unrecognized parameter '{0}'".format(k))
            setattr(self, k, v)

        # Check to make sure that all the parameters were given either with
        # a default value or in this initialization.
        if np.any(np.isnan(self._vector)):
            pars = []
            for k, o in iteritems(self._parameters):
                if np.any(np.isnan(self._vector[o.index])):
                    pars.append(k)
            raise ValueError(
                ("missing values for parameters: {0}. Use a default parameter "
                 "value or specify a value in the initialization of the model")
                .format(pars)
            )

    def __getattr__(self, name):
        if self._parameters is None or name not in self._parameters:
            raise AttributeError(name)

        o = self._parameters.get(name, None)
        return self._vector[o.index]

    def __setattr__(self, name, value):
        if self._parameters is None or name not in self._parameters:
            return super(ModelMixin, self).__setattr__(name, value)

        o = self._parameters.get(name, None)
        self._vector[o.index] = value

    def __getitem__(self, k):
        return self.get_parameter(k)

    def __setitem__(self, k, v):
        return self.set_parameter(k, v)

    def __len__(self):
        return len(self._frozen) - sum(self._frozen)

    def get_vector(self, full=False):
        if full:
            return self._vector
        return self._vector[~self._frozen]

    def set_vector(self, value, full=False):
        if full:
            self._vector[:] = value
        else:
            self._vector[~self._frozen] = value

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
        for k, o in iteritems(self._parameters):
            if len(o):
                names += ["{0}({1})".format(k, i) for i in range(len(o))]
            else:
                names.append(k)
        if full:
            return names
        return [n for i, n in enumerate(names) if not self._frozen[i]]

    def match_parameter(self, name):
        # Return fast for exact matches.
        if name in self._parameters:
            return self._parameters[name].index

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

    def __call__(self, *args, **kwargs):
        return self.get_value(*args, **kwargs)

    def get_value(self, *args, **kwargs):
        raise NotImplementedError("sublasses must implement the 'get_value' "
                                  "method")

    def get_gradient(self, *args, **kwargs):
        if hasattr(self, "get_value_and_gradient"):
            return self.get_value_and_gradient(*args, **kwargs)[1]

        logging.warn("using default numerical gradients; this might be slow "
                     "and numerically unstable")
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
    mu = Parameter(length=3)
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
    # print(m.get_bounds())
    # print(m.get_vector())

    print(m.get_value(0.1))
    print(m.get_gradient(0.1))

    # print(m._vector)
    # print(m.log_sigma)
    # print(m.get_parameter_names())
    # print(m.get_parameter_names(full=True))
    # print(m.get_parameter("log_sigma"))

    # m2 = Model(sigma=10.0)
    # print(m2._vector)
    # print(m2.sigma)

    # print(m._vector)
    # print(m.sigma)
    # print(m.amp)

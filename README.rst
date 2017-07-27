modeling.py
===========

This module was originally written to support model building with my Gaussian
Process packages `george <http://george.readthedocs.io>`_ and `celerite
<http://celerite.readthedocs.io/>`_, but then I started using it for a few
other projects and decided to split it out into its own project.

The basic goal is to provide a flexible skeleton for specifying probabilistic
models in Python. This module doesn't include any optimization routines and it
is agnostic about how you go about computing the value of the model, but it is
compatible with things like `scipy.optimize
<https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>`_,
`skopt <https://scikit-optimize.github.io/>`_, `emcee
<http://dan.iel.fm/emcee>`_, and it can probably be adapted to work with other
inference engines.

**Some features:**

1. The parameters of the models are named, they can be bounded, and they can
   be "frozen" or "thawed" so that you can fit for subsets of the parameters.
2. Models can be composed into "model sets".
3. More to come?

Installation
------------

First you need to install `NumPy <http://www.numpy.org/>`_ if you don't
already have it. Then:

.. code-block:: bash

    git clone https://github.com/dfm/modeling.py
    cd modeling.py
    python setup.py install

Usage
-----

For now, look in the `examples directory
<https://github.com/dfm/modeling.py/tree/master/examples>`_ for some sample
code.

License
-------

Copyright 2015, 2016, 2017 Daniel Foreman-Mackey

This is free software made available under the MIT License.
For details see the LICENSE file.

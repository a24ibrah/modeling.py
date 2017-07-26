#!/usr/bin/env python

import os
import sys

from setuptools import setup

if sys.argv[-1] == "publish":
    os.system("python setup.py sdist upload")
    sys.exit()

if sys.version_info[0] < 3:
    import __builtin__ as builtins
else:
    import builtins
builtins.__MODELING_SETUP__ = True
import modeling  # NOQA


setup(
    name="modeling",
    version=modeling.__version__,
    author=modeling.__author__,
    author_email="foreman.mackey@gmail.com",
    url="https://github.com/dfm/modeling.py",
    py_modules=["modeling"],
    description=("A flexible framework for building models for all your data "
                 "analysis needs."),
    long_description=open("README.rst").read(),
    package_data={"": ["LICENSE", "README.rst"]},
    include_package_data=True,
    install_requires=["numpy"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
    ],
)

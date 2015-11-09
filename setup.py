#!/usr/bin/env python

import os
import sys
import modeling
from setuptools import setup

if sys.argv[-1] == "publish":
    os.system("python setup.py sdist upload")
    sys.exit()


setup(
    name="modeling",
    version=modeling.__version__,
    author=modeling.__author__,
    author_email="foreman.mackey@gmail.com",
    url="https://github.com/dfm/modeling.py",
    py_modules=["corner"],
    description=("A flexible framework for building models for all your data "
                 "analysis needs."),
    long_description=open("README.rst").read(),
    package_data={"": ["LICENSE", "README.rst"]},
    include_package_data=True,
    classifiers=[
        "Development Status :: 1 - Production/Stable",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
    ],
)

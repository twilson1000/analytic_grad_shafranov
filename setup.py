#!/usr/bin/env python3
# -*- coding: utf-8
'''

'''
# Standard imports.
from setuptools import setup, find_packages

# Custom imports.
from analytic_grad_shafranov import __version__, __author__, __url__

def readme():
    with open('README.md') as f:
        return f.read()

setup(
    name="analytic_grad_shafranov",
    version=__version__,
    description="Python implementation of analytic Grad Shafranov solutions",
    long_description=readme(),
    url=__url__,
    packages=["analytic_grad_shafranov"],
    entry_points = {},
    author=__author__,
    author_email="thomas.wilson@ukaea.uk",
    install_requires=[ 
        "numpy",
        "scipy",
    ]
)
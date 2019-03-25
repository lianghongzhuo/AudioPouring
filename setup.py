#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author     : Hongzhuo Liang 
# E-mail     : liang@informatik.uni-hamburg.de
# Description: 
# Date       : 07/11/2018: 10:27 AM
# File Name  : setup.py

from setuptools import setup, find_packages

__version__ = "0.0.1"
setup(
    name="audio_pouring",
    version=__version__,
    keywords=["audio-pouring", "deep-learning"],
    description="audio pouring project code",
    license="MIT License",
    url="https://github.com/lianghongzhuo/AudioPouring",
    author="Hongzhuo Liang",
    author_email="liang@informatik.uni-hamburg.de",
    packages=find_packages(),
    include_package_data=True,
    platforms="any",
    install_requires=[]
)

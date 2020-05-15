#!/usr/bin/env python

from setuptools import setup

VERSION = '0.0.1'

setup(name='bilby_psd',
      description='',
      author='Greg Ashton, Colm Talbot',
      license="MIT",
      version=VERSION,
      packages=['bilby_psd'],
      package_dir={'bilby_psd': 'bilby_psd'},
      python_requires='>=3.5')

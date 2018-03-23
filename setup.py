from setuptools import setup
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md')) as f:
    imagej_long_description = f.read()


setup(
    name='imagej',
    version= '0.1.2',
    author='Yang Liu && Leon Yang',
    author_email='liu574@wisc.edu',
    url='https://github.com/imagej/imagej.py/',
    packages=['imagej'],
    platforms=['any'],
    description='Python wrapper for imagej',
    long_description=imagej_long_description,
    license='Apache 2.0'
)

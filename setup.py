from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md')) as f:
    pyimagej_long_description = f.read()

config={}
with open('imagej/config.py', 'r') as f:
    exec(f.read(), config)

setup(
    name='pyimagej',
    version=config['__version__'],
    author=config['__author__'],
    author_email='ctrueden@wisc.edu',
    url='https://github.com/imagej/pyimagej',
    packages=find_packages(),
    platforms=['any'],
    description='Python wrapper for ImageJ',
    long_description=pyimagej_long_description,
    long_description_content_type='text/markdown',
    license='Apache 2.0',
    install_requires=[
        'imglyb',
        'jpype1',
        'matplotlib',
        'numpy',
        'scyjava',
        'xarray'
    ],
    tests_require=[
        'pytest'
    ],
    entry_points={
        'console_scripts': [
            'imagej=imagej:imagej_main'
        ]
    }
)

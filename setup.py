from setuptools import setup
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md')) as f:
    readme = f.read()\
        .replace('](doc', '](https://github.com/imagej/pyimagej/blob/master/doc')

config={}
with open('src/imagej/config.py', 'r') as f:
    exec(f.read(), config)

setup(
    version=config['__version__'],
    long_description=readme,
)

from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md')) as f:
    readme = f.read()\
        .replace('](doc', '](https://github.com/imagej/pyimagej/blob/master/doc')\
        .replace('](issues', '](https://github.com/imagej/pyimagej/issues')

config={}
with open('imagej/config.py', 'r') as f:
    exec(f.read(), config)

setup(
    name='pyimagej',
    python_requires='>=3.6',
    version=config['__version__'],
    author=config['__author__'],
    author_email='ctrueden@wisc.edu',
    url='https://github.com/imagej/pyimagej',
    packages=find_packages(),
    platforms=['any'],
    description='Python wrapper for ImageJ',
    long_description=readme,
    long_description_content_type='text/markdown',
    license='Apache 2.0',
    install_requires=[
        'imglyb >= 2.0.0',
        'jgo >= 1.0.3',
        'jpype1 >= 1.3.0',
        'matplotlib',
        'numpy',
        'scyjava >= 1.4.0',
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

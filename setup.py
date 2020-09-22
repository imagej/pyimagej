from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md')) as f:
    pyimagej_long_description = f.read()


setup(
    name='pyimagej',
    # TODO: Unify version declaration to one place.
    # https://www.python.org/dev/peps/pep-0396/#deriving
    version= '0.6.1.dev0',
    author='Curtis Rueden, Leon Yang, Yang Liu, Michael Pinkert',
    author_email='ctrueden@wisc.edu',
    url='https://github.com/imagej/pyimagej',
    packages=find_packages(),
    platforms=['any'],
    description='Python wrapper for ImageJ',
    long_description=pyimagej_long_description,
    long_description_content_type='text/markdown',
    license='Apache 2.0',
    install_requires=[
        'matplotlib',
        'numpy',
        'xarray',
        'pillow', # for server
        'requests' # for server
    ],
    tests_require=[
        'pytest'
    ],
    entry_points={
        'console_scripts': [
            'imagej=imagej.imagej:imagej_main'
        ]
    }
)

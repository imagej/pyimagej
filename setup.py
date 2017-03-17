from distutils.core import setup
import re

with open('imagej.py', 'rb') as src:
    for line in src:
        m = re.match("__version__ = '(.*?)'", line.decode('utf-8'))
        if m is not None:
            imagej_version = m.group(1)
            break
    else:
        raise RuntimeError('Cannot find imagej version')

with open('README.md', 'rb') as readme:
    imagej_long_description = readme.read().decode('utf-8')

setup(
    name='imagej',
    version=imagej_version,
    author='Leon Yang',
    author_email='leon.gh.yang@gmail.com',
    url='https://github.com/imagej/imagej-server/',
    py_modules=['imagej'],
    platforms=['any'],
    install_requires=['requests', 'pillow'],
    description='Python client for imagej-server',
    long_description=imagej_long_description,
    license='Apache 2.0'
)

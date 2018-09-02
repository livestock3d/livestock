__author__ = "Christian Kongsgaard"
__license__ = "MIT"

# ---------------------------------------------------------------------------- #
# Imports

from setuptools import setup
from codecs import open
from os import path
import datetime

# ---------------------------------------------------------------------------- #
here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

now = datetime.datetime.now()
release_version = '1.post1'
version = f'{now.year}.{now.month}.{release_version}'

setup(
    name='livestock',
    version=version,
    description='Livestock is a plugin/library for Grasshopper '
                'written in Python',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/ocni-dtu/livestock',
    author='Christian Kongsgaard',
    author_email='ocni@dtu.dk',
    license='GNU GPLv3',
    keywords='hydrology 3dmodeling grasshopper',
    packages=['livestock', ],
    install_requires=['scipy', 'numpy', 'paramiko', 'pyshp', 'shapely',
                      'cmf==1.2', 'xmltodict', 'dbfread', 'progressbar2'],
    python_requires='>3.x',
    #entry_points={'console_scripts': ['livestock=livestock:main', ], },
)

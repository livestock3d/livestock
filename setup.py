# Imports
from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

with open(path.join(here, 'requirements.txt'), encoding='utf-8') as r:
    requirements = [name.strip() for name in r.readlines()]

setup(
      name='livestock',
      version='2018.02.4',
      description='Livestock is a plugin/library for Grasshopper written in Python',
      long_description=long_description,
      url='https://github.com/ocni-dtu/livestock_gh',
      author='Christian Kongsgaard Nielsen',
      author_email='ocni@dtu.dk',
      license='MIT',
      keywords='hydrology 3dmodeling grasshopper',
      packages=find_packages(exclude=['archive', 'config_livestock', 'test', 'wiki']),
      install_requires=requirements,
      python_requires='>=3',
      entry_points={'console_scripts': ['livestock=livestock:main', ], },
      )

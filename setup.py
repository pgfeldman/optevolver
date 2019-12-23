from setuptools import find_packages, setup

setup(
    name='optevolver',
    packages=find_packages(),
    version='0.1.4',
    description='Evolutionary parameter search for architecture and hyperparameters',
    long_description='Contains packages for creating sequence data (generators), working with Excel and JSON files (util) and evolutionary discovery of architecture and hyperparameters (hyperparameter). It is designed to work with Tensorflow 2.0 and can take advantage of multiple processors',
    author='Phil Feldman',
    license='MIT',
    long_description_content_type="text/markdown",
    url="https://github.com/pgfeldman/optevolver",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6'
)

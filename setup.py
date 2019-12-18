from setuptools import find_packages, setup

setup(
    name='optevolver',
    packages=find_packages(),
    version='0.1.2',
    description='Evolutionary parameter search for architecture and hyperparameters',
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

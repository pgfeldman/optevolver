OptEvolver
==============================

Evolutionary parameter search for architecture and hyperparameters

## Project Organization

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── generators     <- Scripts to download or generate data
    │   │   ├── descending.py
    │   │   ├── floatfunctions.py
    │   │   ├── intfunctions.py
    │   │   ├── JackTorrance.py
    │   │   └── texttokenizer.py    
    │   │
    │   ├── hyperparameter  <- Scripts to evolve hyperparameters and architecture
    │   │   ├── EnumeratedTypes.py
    │   │   ├── EvolutionaryOptimizer.py
    │   │   ├── ModelWriter.py
    │   │   ├── TF2OptimizationTestBase.py
    │   │   └── TF2OptimizationTest.py
    │   │
    │   ├── util           <- Scripts for various utilities
    │   │   ├── excel_utils.py
    │   │   ├── file_parser.py
    │   │   └── JsonUtils.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

<h1>Package Tutorial</h1>

Conceptually, there are two parts to this package. The core is the <b>EvolutionaryOptimizer.py</b> module, that builds a population of <em>Genomes</em>, and evolves that population over multiple generations against a fitness test. Two additional files, <b>TF2OptimizerTestBase.py</b> and <b>TF2OptimizerTest.py</b> demonstrate how to use the evolver to search hyperparameter and architecture spaces with Tensorflow 2.x. 

This tutorial will have two parts - the first will walk through a simple example that uses the evolver alone. The second will build an ensemble of tensorflow models, take their average fitness, and evolve a multilayer perceptron network to do sequence-to-sequence matching.

<h2>EvolutionaryOptimizer Tutorial</h2>
(The code for this tutorial is in the optevolver/examples/EO_example.py file)

This example creates a set of parameters for the evolver to work with, does an exhaustive evaluation for comparison, and then evolves a solution. Some graphs are generated at the end to show the fitness landscape, and the exhaustive and evolved solutions. To run the example, you will have to (pip) install the optevover package

<h3>Imports</h3>
The following are required for this example:
<pre>
import random
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from mpl_toolkits.mplot3d import axes3d, Axes3D  # <-- DON'T DELETE, and note the capitalization!

import optevolver.hyperparameter.ValueAxis as VA
import optevolver.hyperparameter.EvolutionaryOptimizer as EO
</pre>

<h3>Example evaluation function</h3>
<pre>
def example_evaluation_function(arguments: Dict) -> Tuple[Dict, Dict]:
    x = arguments['X'] + random.random() - 0.5
    y = arguments['Y'] + random.random() - 0.5
    val = np.cos(x) + x * .1 + np.sin(y) + y * .1

    return {EO.EvolverTags.FITNESS.value: val}, {
        EO.EvolverTags.FILENAME.value: "{}.tf".format(arguments[EO.EvolverTags.ID.value])}
</pre>





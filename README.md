OptEvolver
==============================

Evolutionary parameter search for architecture and hyperparameters

Project Organization
------------

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

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
    ├── examples           <- Directory that has self-contained examples from this package
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

This example creates a set of parameters for the evolver to work with, does an exhaustive evaluation for comparison, and then evolves a solution. Some graphs are generated at the end to show the fitness landscape, and the exhaustive and evolved solutions. To run the example, you will have to (pip) install the <b>optevolver</b> package

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
This is the function that is used to produce a fitness value. An "arguments" Dict  is passed in with the values to work with It's ['X'] and ['Y'] in this case, but it can be Ebochs, Batch size, Neurons per layer, etc. The function returns two Dicts (d1, d2), where d1 must contain:<br/>
- d1: {EvolverTags.FITNESS.value : <some fitness value>} <br/>
- d2: data that will be recorded to the spreadsheet for post-hoc analysis 
<pre>
def example_evaluation_function(arguments: Dict) -> Tuple[Dict, Dict]:
    x = arguments['X'] + random.random() - 0.5
    y = arguments['Y'] + random.random() - 0.5
    val = np.cos(x) + x * .1 + np.sin(y) + y * .1

    return {EO.EvolverTags.FITNESS.value: val}, {
        EO.EvolverTags.FILENAME.value: "{}.tf".format(arguments[EO.EvolverTags.ID.value])}
</pre>

<h3>Example save function</h3>
Called from the evolver when a new best average fitness is found, so that the new best results can be written out
<pre>
def example_save_function(name: str) -> str:
    # do something with the new best value
    return "new best value: {}".format(name)
</pre>

<h3>Setting the values to evolve</h3>
Set up the 'X' and 'Y' values so a finite number of options are generated. In this case, the value is an array of float that goes from -5.0 to 5.0 in 0.25 increments. That's 40 values per dimension. Since we'll be calculating a surface, the total number of points on this surface will be 40 * 40 or 1,600.
<pre>
v1 = VA.EvolveAxis("X", VA.ValueAxisType.FLOAT, min=-5, max=5, step=0.25)
v2 = VA.EvolveAxis("Y", VA.ValueAxisType.FLOAT, min=-5, max=5, step=0.25)
</pre>

<h3>Calculate the grid search version as a baseline</h3>
Iterate over the ranges in v1 and v2 and get a fitness value from our example_evaluation_function(). We set the initial fitness (prev_fitness) to -10, and every time we find a better value, we add that to our exhaustive_list so we can see how many steps it took. 
<pre>
prev_fitness = -10
count = 0
exhaustive_list = []
for x in range(len(v1.range_array)):
    for y in range(len(v2.range_array)):
        count += 1
        args = {'X': v1.range_array[x], 'Y': v2.range_array[y], EO.EvolverTags.ID.value: "eval_[{}]_[{}]".format(x, y)}
        d1, d2 = example_evaluation_function(args)
        cur_fitness = d1[EO.EvolverTags.FITNESS.value]
        if (cur_fitness > prev_fitness):
            prev_fitness = cur_fitness
            exhaustive_list.append(cur_fitness)
</pre>

<h3>Set up for the evolver </h3>
Now do the same thing using evoultionary fitness landscape evaluation. We first create an instance of the EvolutionaryOpimizer that keeps the top 50% of the genomes for each generation, and runs on zero threads. <br/>
Threads can equal the number of processors. Zero is best for stepping through code in a debugger.
<pre>
eo = EO.EvolutionaryOpimizer(keep_percent=.5, threads=0)
</pre>

Add the already created EvolveAxis to the EvolutionaryOptimizer in any order.
<pre>
eo.add_axis(v1)
eo.add_axis(v2)
</pre>

Create an initial population of 10 genomes.
<pre>
eo.create_intital_genomes(10)
</pre>

<h3>Run the evolver </h3>
Run for the same number of steps that it took to create the exhaustive list. Note - this is completely arbitrary so that some nice plots can be made. In an actual version, there should ba a max number of iterations that a fitness no longer improve. <br/>

Create a List of fitness values to plot.
<pre>
evolve_list = []
</pre>
Set the number of generations. We'll just go for the same number of steps as we saved off when doing the grid search. This will make for a nicer plot later.
<pre>
num_generations = len(exhaustive_list) 
for i in range(num_generations):
    # evolve a generation, providing the evaluation and save functions, and a crossover and mutation rate of 50%
    fitness = eo.run_optimizer(example_evaluation_function, example_save_function, crossover_rate=0.5, mutation_rate=0.5)
    evolve_list.append(fitness)
</pre>

<h3>Show results</h3>

First, print out all the genome data, and also the best genome
<pre>
print("xxxxxxxxxxxxxxxx\n{}".format(eo.to_string()))

best_genome = eo.get_ranked_genome(0)
best_genome_data = best_genome.get_data_list()
d: Dict
print("best genome = {}".format(best_genome.get_name()))
for i in range(len(best_genome_data)):
    d = best_genome_data[i]
    for key, val in d.items():
        print("data [{}]: {} = {}".format(i, key, val)
</pre>

Next, save the results to a spreadsheet for post hoc analysis.
<pre>
eo.save_results("evolve_test.xlsx")
</pre>
Thesaved spreadsheet will include all the best values for each generation and can be used to make nice plots:
![Example Spreadsheet](./reports/figures/ExcelHistory.png)

Last, plot the exhaustive and evolve sequences. The exhaustive line is almost deterministic and will pretty much look the same for each run. The evolved line is stochastic, and can change significantly for each run.

<pre>
fig = plt.figure(1)
plt.plot(exhaustive_list)
plt.plot(evolve_list)
plt.legend(["exhaustive ({} iterations)".format(count), "evolved ({} iterations)".format(num_generations)])

# draw a picture of our XY fitness landscape. This is the same range used to create the axis and the same equation in
# def example_evaluation_function(arguments: Dict) -> Dict:
fig = plt.figure(2)
ax = fig.gca(projection='3d')

# Make our 3d surface using the same equation in example_evaluation_function()
X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)
R = np.cos(X) + X * .1 + np.sin(Y) + Y * .1
Z = R

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

# s Show the plots
plt.show()
</pre>

This gives us two plots. The first is a 3D surface of our fitness landscape: 
![Fitness landscape](./reports/figures/Surface.png)
The second is a plot of best values. Here we can see the result of the earlier, exhaustive search. It took a total of 1,600 steps, and found a new best value 16 times. The evolutionary approach ran for just 16 generations, with a population of 10 for each calculation for a total of 160 steps. For a true apples to apples comparison, the grid search should work with an ensemble of 10 samples as well, so the evolutionary approach is at least 100 times faster than the grid search, and nearly as effective.
![Fitness landscape](./reports/figures/progress.png)

Next, we'll try this approach with Tensorflow architecture and hyperparameter values!



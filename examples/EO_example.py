import random
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from mpl_toolkits.mplot3d import axes3d, Axes3D  # <-- DON'T DELETE, and note the capitalization!

import optevolver.hyperparameter.ValueAxis as VA
import optevolver.hyperparameter.EvolutionaryOptimizer as EO

# example evaluation function
def example_evaluation_function(arguments: Dict) -> Tuple[Dict, Dict]:
    x = arguments['X'] + random.random() - 0.5
    y = arguments['Y'] + random.random() - 0.5
    val = np.cos(x) + x * .1 + np.sin(y) + y * .1

    return {EO.EvolverTags.FITNESS.value: val}, {
        EO.EvolverTags.FILENAME.value: "{}.tf".format(arguments[EO.EvolverTags.ID.value])}


# A stub of a save function that does nothing
def example_save_function(name: str) -> str:
    return "would have new best value: {}".format(name)


# create the x and y values for our surface. For this example, these are intervals from -5 to 5, with a step of 0.25
v1 = VA.EvolveAxis("X", VA.ValueAxisType.FLOAT, min=-5, max=5, step=0.25)
v2 = VA.EvolveAxis("Y", VA.ValueAxisType.FLOAT, min=-5, max=5, step=0.25)

# do an exhastive evaluation for comparison. Each time a new, better value is found, add it to the list for plotting
prev_fitness = -10
num_exhaust = 0
exhaustive_list = []
for x in range(len(v1.range_array)):
    for y in range(len(v2.range_array)):
        num_exhaust += 1
        args = {'X': v1.range_array[x], 'Y': v2.range_array[y], EO.EvolverTags.ID.value: "eval_[{}]_[{}]".format(x, y)}
        d1, d2 = example_evaluation_function(args)
        cur_fitness = d1[EO.EvolverTags.FITNESS.value]
        if (cur_fitness > prev_fitness):
            prev_fitness = cur_fitness
            exhaustive_list.append(cur_fitness)

# now do it using evoultionary fitness landscape evaluation
# create an instance of the EvolutionaryOpimizer that keeps the top 50% of the genomes for each generation.
# Threads can equal the number of processors. Zero is best for stepping through code in a debugger
eo = EO.EvolutionaryOpimizer(keep_percent=.5, threads=0)
# add the EvolveAxis. Order doesn't matter here
eo.add_axis(v1)
eo.add_axis(v2)

# create an initial population of 10 genomes
eo.create_intital_genomes(10)

# run for the same number of steps that it took to create the exhaustive list. Note - this is completely arbitrary
# so that some nice plots can be made. In an actual version, there should ba a max number of iterations that a fitness
# no longer improves
# create a List of fitness values to plot
evolve_list = []
# set the number of generations
num_generations = len(exhaustive_list) * 2
for i in range(num_generations):
    # evolve a generation, providing the evaluation and save functions, and a crossover and mutation rate of 50%
    fitness = eo.run_optimizer(example_evaluation_function, example_save_function, 0.5, 0.5)
    evolve_list.append(fitness)
    # print("best fitness = {:.3f}".format(fitness))

# print the genomes
print("xxxxxxxxxxxxxxxx\n{}".format(eo.to_string()))

best_genome = eo.get_ranked_genome(0)
best_genome_data = best_genome.get_data_list()
d: Dict
print("best genome = {}".format(best_genome.get_name()))
for i in range(len(best_genome_data)):
    d = best_genome_data[i]
    for key, val in d.items():
        print("data [{}]: {} = {}".format(i, key, val))

# save the results to a spreadsheet for post hoc analysis
eo.save_results("evolve_test.xlsx")

# plot the exhaustive and evolve sequences. The exhaustive line is almost deterministic and will pretty much look
# the same for each run. The evolved line is stochastic, and can change significantly for each run
fig = plt.figure(1)
plt.plot(exhaustive_list)
plt.plot(evolve_list)
plt.legend(["exhaustive ({} iterations)".format(num_exhaust), "evolved ({} iterations)".format(num_generations)])

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
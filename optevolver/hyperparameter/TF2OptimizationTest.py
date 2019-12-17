from typing import Dict, Tuple

import numpy as np
import time

import optevolver.generators.FloatFunctions as FF
import optevolver.hyperparameter.EvolutionaryOptimizer as EO
import optevolver.hyperparameter.ModelWriter as MW
import optevolver.hyperparameter.ValueAxis as VA
from optevolver.hyperparameter.TF2OptimizerTestBase import TF2OptimizationTestBase


class TF2OptimizationTest(TF2OptimizationTestBase):
    """
    Class that create a working implementation of the TF2OptomizerTestBase class, This gives an example of how to
    override the generate_train_test() method. Most of the work is done in this file outside the class, where the
    eval_func and save_func functions are created and attached to this class/superclass

    ...

    Attributes
    ----------
    All attributes are inherited from TF2OptimizationTestBase


    Methods
    -------
    generate_train_test() -> (np.ndarray, np.ndarray, np.ndarray):
        Generates a set of test and train data for sequence creation

    """

    def __init__(self, sequence_length: int = 100, device: str = "gpu:0"):
        """
        Parameters
        ----------
        sequence_length: int = 100
            The sice of the input and output vectors. Defaults to 100
        device: str = "gpu:0"
            The string used to set the Tensorflow distribution strategy. If, for example, the machine that this
            was running on had four cpus, an instance of this class would be created for 'cpu:0', 'cpu:1', 'cpu:2', and
            'cpu:3'. Default is 'gpu:0'
        """
        super().__init__(sequence_length, device=device)

    def generate_train_test(self, num_functions: int, rows_per_function: int, noise: float) -> \
            (np.ndarray, np.ndarray, np.ndarray):
        """
        Method that generates data so that you don't have to find data. You can optionally set the mat values directly.
        The values are set using a set of frequencies for a sin function. All sequences start at sin(0) = 0, then
        diverge from there. Returns a tuple of (full_mat, train_mat, text mat), where the test and train matrices are
        made of the evenly split full matrix.

        Parameters
        ----------
        num_functions: int
            the number of sin-based functions to create
        rows_per_function: int
            the number of rows to create for each sin function. With noise == 0, these rows would be identical
        noise: float
            the amount of noise to add to each generated value
        """

        # create  an instance of the class that will generate our values. Note that rows are twice the length of our sequence
        ff = FF.FloatFunctions(rows_per_function, 2 * self.sequence_length)
        npa = None
        for i in range(num_functions):
            mathstr = "math.sin(xx*{})".format(0.005 * (i + 1))
            # generate a dataframe that with values
            df2 = ff.generateDataFrame(mathstr, noise=noise)
            # convert the DataFrame to an Numpy array
            npa2 = df2.to_numpy()

            # if this is the first time, set the initial value, otherwise append
            if npa is None:
                npa = npa2
            else:
                ta = np.append(npa, npa2, axis=0)
                npa = ta

        # split into the train and test matrices
        split = np.hsplit(npa, 2)

        # return the tuple
        return npa, split[0], split[1]


########################################################################################
# The following code exercises the class by creating functions, training an ensemble of
# models and evolving the best solution given the number of generations

# external arguments to the eval and save functions
in_out_vector_size = 100
eval_folder_name = "../../models/eval"
best_folder_name = "../../models/best"


def eval_func(arguments: Dict) -> Tuple[Dict, Dict]:
    """
        This function takes the argument Dict, creates a TF2OptimizationTest class, builds and evaluates the model,
        and returns two Dicts, one for the evaluation stats and one to provide a filename for the model to be saved
        as 'best', if it has better results than a previous run.

        Parameters
        ----------
        arguments : Dict
            all the arguments that this function would need. The defaults, passed in from
            EvolutionaryOpimizer.run_optimizer() are passed into TF2OptimizationTestBase.build_model(), where
            they are stored and used by evaluate_madel

    """
    # set up device-specific threading. Default is the gpu
    device_name = "gpu:0"
    if EO.EvolverTags.THREAD_NAME.value in arguments:
        device_name = arguments[EO.EvolverTags.THREAD_NAME.value]

    # Instantiate the class
    tf2o = TF2OptimizationTest(in_out_vector_size, device=device_name)

    # build the model, based on the argments passed in from the evolver
    tf2o.build_model(arguments)
    # evaluate the model, based on the argments passed in from the evolver
    d1 = tf2o.evaluate_model(num_functions=10, rows_per_function=1, noise=0)

    # prepare our return Dicts. Note that our fitness value could be some other value, or calculated here
    d1[EO.EvolverTags.FITNESS.value] = d1['accuracy']
    filename = "model_{}.tf".format(arguments[EO.EvolverTags.ID.value])
    d2 = {"filename": filename}
    MW.write_model(eval_folder_name, filename, tf2o.model)
    return d1, d2


def save_func(name: str) -> str:
    """
        This function is called if the evaluated model from eval_func() has a new 'best' fitness value. It deletes the
        best directory, renames the current eval directory to the best directory, creates a new, empty eval directory,
        and returns a status string

        Parameters
        ----------
        name : str
            A label that this class returns that the EvolutionaryOptimizer can keep track of

    """
    MW.del_dir(best_folder_name)
    MW.move_dir(eval_folder_name, best_folder_name)
    MW.create_dir(eval_folder_name)
    filename = "{}/config.txt".format(best_folder_name)
    with open(filename, "w") as f:
        f.write("config = {}".format(name))
    return "saved {}".format(name)

# Exercise this class either by creating and saving an ensemble of models, using the saved models to make
# predictions, or both
if __name__ == "__main__":
    # choose whether to train, evaluate, or both
    do_train = False
    do_evaluate = True

    if do_train:
        # set up the filename that we will save our statistics to
        timestr = time.strftime("%H-%M_%m-%d-%Y", time.gmtime())
        filename = "../../data/evolve_{}.xlsx".format(timestr)

        # create independent EvolveAxis for hyperparameters (epochs and batch_size), and architecture search (num_neurons, num_layers)
        v1 = VA.EvolveAxis("epochs", VA.ValueAxisType.INTEGER, min=10, max=100, step=10)
        v2 = VA.EvolveAxis("batch_size", VA.ValueAxisType.INTEGER, min=2, max=20, step=1)
        v3 = VA.EvolveAxis("num_neurons", VA.ValueAxisType.INTEGER, min=10, max=1000, step=10)
        v4 = VA.EvolveAxis("num_layers", VA.ValueAxisType.INTEGER, min=1, max=10, step=1)

        # create the EvolutionaryOptimazer and add the EvolveAxis
        eo = EO.EvolutionaryOpimizer(threads=1)
        eo.add_axis(v1)
        eo.add_axis(v2)
        eo.add_axis(v3)
        eo.add_axis(v4)

        # based on the set of EvolveAxis, create a set of 10 randomly-generated genomes that will be the population
        # that we start with
        eo.create_intital_genomes(10)

        # set up the directory that we will save our current ensemble
        MW.create_dir(eval_folder_name)

        # set up the evolver. We will go for num_generations. Crossover rate is the chance that one parent will
        # contribute its EvolveAxis value to the newly created offspring. Mutation rate is the chance that a
        # gene will change after the initial crossover. Once crossover and mutation are finished, the newly
        # instantiated genome is added to the population
        num_generations = 50
        crossover_rate = 0.5
        mutation_rate = 0.5
        for i in range(num_generations):
            # run the optimizer with references to the eval and save functions, plus the crossover and mutation
            # rate. One of the reasons that we do this is that it can make sense to start with a high mutation and
            # crossover rate, but as we hillclimb, we may want more conservative values
            best_ensemble_average_fitness = eo.run_optimizer(eval_func, save_func, crossover_rate, mutation_rate)

            # print the current results
            print("best average fitness (generation {}) = {:.3f}".format(i, best_ensemble_average_fitness))

        # get the top-ranked (highest fitness) genome and print its name, which includes genome values
        best_genome = eo.get_ranked_genome(0)
        print("best genome = {}".format(best_genome.get_name()))

        # save the statistics to an excel file
        eo.save_results(filename)

    if do_evaluate:
        # since we're not evolving, we don't need the evolver. Just get the TF2OptimizationTest, which can read in and
        # evaluate an ensemble
        tfo = TF2OptimizationTest()

        # make a set of plots that include the individual elements of the ensemble and their average
        tfo.plot_population(best_folder_name, noise=0.0)

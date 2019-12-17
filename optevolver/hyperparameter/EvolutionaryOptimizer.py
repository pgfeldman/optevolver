# Evolutionary optimizer for hyperparameters and architecture. Project at https://github.com/pgfeldman/optevolver
import concurrent.futures
import copy
import datetime
import getpass
import os
import random
import re
import threading
from enum import Enum
from typing import Dict, List, Tuple, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d, Axes3D  # <-- DON'T DELETE, and note the capitalization!
from sklearn.utils import resample

import optevolver.hyperparameter.ValueAxis as VA
import optevolver.util.ExcelUtils as eu


class EvolverTags(Enum):
    """A class containing enumerations elements for use in the argument dictionaries"""
    FITNESS = "fitness"
    ID = "id"
    FUNCTION = "func"
    FLOAT = "float"
    GENERATION = "generation"
    GENOME = "genome"
    THREAD_NAME = "thread_str"
    FILENAME = "filename"


class Genome:
    """
    Class that handles the evolution of a set of ValueAxis (i.e. the chromosome)

    ...

    Attributes
    ----------
    chromosome_dict: Dict
    fitness: float
    ea_list: List
    population: List
    meta_info: Dict
    data_list: List
    generation:int

    Methods
    -------
    reset(self):
        Resets all the variables. Needed to eliminate class cross-contamination of class-global variables
    equals(self, g: "Genome") -> bool:
        Does a deep compare of two Genomes. returns a True if they have the same structure and value(s).
    get_chromosome_value(self, key: str) -> Dict:
    mutate(self, chance: float = 0.1):
    create_args_from_chromo(self, chromo: dict = None) -> Dict:
    create_dict_from_chromo(self, chromo: dict = None) -> Dict:
    calc_fitness(self, func, id_str: str) -> float:
    calc_fitness2(self, args: Dict):
    calc_fitness_stats(self, resample_size: int = 100) -> float:
    get_data_list(self) -> List:
    get_name(self) -> str:
    to_dict(self):
    to_string(self, meta: bool = True, chromo: bool = True) -> str:
    """
    chromosome_dict: Dict
    fitness: float
    ea_list: List
    population: List
    meta_info: Dict
    data_list: List
    generation = 0

    def __init__(self, evolve_axis_list: List, p1: 'Genome' = None, p2: 'Genome' = None, crossover: float = 0.5,
                 generation=0):
        """
        Parameters
        ----------
        evolve_axis_list : List
            The list of all EvolveAxis used to create this genome
        p1 :
            Optional parent for this Genome. Two are required to breed.
        p2
            Optional parent for this Genome. Two are required to breed.
        crossover: float
            probability that a chromosome will be selected randomly from p1
        generation: int
            The generation (as determined by the calling EvolutionaryOpimizer) that this genome belongs to
        """
        self.reset()
        self.generation = generation
        self.ea_list = copy.deepcopy(evolve_axis_list)
        ea: VA.EvolveAxis
        if p1 == None and p2 == None:
            for ea in self.ea_list:
                self.chromosome_dict[ea.name] = ea.get_random_val()

        else:
            # for ea in self.ea_list:
            for i in range(len(self.ea_list)):
                ea = self.ea_list[i]
                ea1 = p1.ea_list[i]
                ea2 = p2.ea_list[i]
                probability = random.random()
                if probability < crossover:
                    ea.set_value(ea1)
                else:
                    ea.set_value(ea2)
                self.chromosome_dict[ea.name] = ea.get_result()

    def reset(self):
        """Resets all the variables. Needed to eliminate class cross-contamination of class-global variables"""
        self.ea_list = []
        self.chromosome_dict = {}
        self.meta_info = {}
        self.fitness = 0
        self.population = []
        self.generation = 0
        self.data_list = []

    def equals(self, g: "Genome") -> bool:
        """Does a deep compare of two Genomes. returns a True if they have the same structure and value(s)

        Parameters
        ----------
        g : Genome
            The genome we are testing against
        """
        d1 = self.create_args_from_chromo()
        d2 = g.create_args_from_chromo()

        if len(d1) != len(d2):
            return False

        for key, val in d1.items():
            if d1[key] != d2[key]:
                return False

        return True

    def get_chromosome_value(self, key: str) -> Dict:
        """ Get the current value of a specified EvolveAxis

        Parameters
        ----------
        key : str
            The name of the EvolveAxis
        """
        return self.chromosome_dict[key]

    def mutate(self, chance: float = 0.1):
        """ Randomly set new values in the chromosomes that make up this genome

        Parameters
        ----------
        chance : float = 0.1
            The probability that any particular chromosome will mutate. Default is 10%
        """
        ea: VA.EvolveAxis
        for ea in self.ea_list:
            if random.random() < chance:  # mutate.
                # calculate a new random val
                self.chromosome_dict[ea.name] = ea.get_random_val()

    def create_args_from_chromo(self, chromo: dict = None) -> Dict:
        """ Creates a dictionary that provides values that can be evaluated using the callback function passed to the
        EvolutionaryOptimizer. An example of this is the function near the bottom of this file:
            def example_evaluation_function(arguments: Dict) -> Tuple[Dict, Dict]:
        The arguments:Dict parameter is created and returned by this method

        Parameters
        ----------
        chromo : dict = None
            An optional chromosome. Otherwise the arguments are created by using this Genome's self.chromosome_dict
        """
        if chromo == None:
            chromo = self.chromosome_dict

        to_return = {}
        ea: VA.EvolveAxis
        for ea in self.ea_list:
            to_return[ea.name] = ea.get_result()
        return to_return

    def create_dict_from_chromo(self, chromo: dict = None) -> Dict:
        """ Creates a dictionary that provides a detailed list of the parameters used by this genome. This differs from
        create_args_from_chromo() by including nested parameters of each EvolveAxis

        Parameters
        ----------
        chromo : dict = None
            An optional chromosome. Otherwise the arguments are created by using this Genome's self.chromosome_dict
        """
        if chromo == None:
            chromo = self.chromosome_dict

        to_return = {}
        ea: VA.EvolveAxis
        for ea in self.ea_list:
            dict = ea.get_last_history()
            for key, value in dict.items():
                to_return["{}".format(key)] = value
        return to_return

    def calc_fitness(self, func: Callable, id_str: str) -> float:
        """ Depricated - Conceptually the heart of the approach. A pointer to a function is passed in, which is used to
        calculate the fitness of whatever is being evaluated and returns it.

        Parameters
        ----------
        func : Callable
            The function that will produce some fitness value. It returns two Dicts (d1, d2), where d1 must contain a
            "fitness" value and d2, which contains data that will be recorded to the spreadsheet for post-hoc
            analysis
        id_str: str
            The name for this evaluation. Added to the argument Dict in case it is needed, for example, as a file name
        """
        args = self.create_args_from_chromo(self.chromosome_dict)
        args[EvolverTags.ID.value] = id_str
        d1, d2 = func(args)
        self.data_list.append(d2)
        self.fitness = d1[EvolverTags.FITNESS.value]
        self.population.append(self.fitness)
        return self.fitness

    def calc_fitness2(self, args: Dict):
        """ Conceptually the heart of the approach. A pointer to a function is passed in, which is used to
        calculate the fitness of whatever is being evaluated.

        Parameters
        ----------
        args : Dict
            Contains the arguments that will be passed to the evaluate function, and a reference to the function as
            well. The function is deleted from the arguments, and the remaining Dict os passed to the function, which
            is required to produce a fitness value.  It returns two Dicts (d1, d2), where d1 must contain a
            {EvolverTags.FITNESS.value : <some fitness value>} and d2, which contains data that will be recorded to the
            spreadsheet for post-hoc analysis
        """
        args.update(self.create_args_from_chromo())
        func = args[EvolverTags.FUNCTION.value]
        del args[EvolverTags.FUNCTION.value]
        d1, d2 = func(args)
        self.data_list.append(d2)
        self.fitness = d1[EvolverTags.FITNESS.value]
        self.population.append(self.fitness)

    def calc_fitness_stats(self, resample_size: int = 100) -> float:
        """ Creates a bootstrap resampling of the fitness values that have accumulated for this genome. Since the
        fitness value may be stochastic, it's best to return a reasonable mean value. It returns the mean
        fitness value from this population, and saves the 5%, 95%, minimum, and maximum values for post-hoc analysis

        Parameters
        ----------
        resample_size: int = 100
            The size of the bootstrap population to resample into
        """
        #  print("calc_fitness_stats(): population = {}".format(len(self.population)))
        boot = resample(self.population, replace=True, n_samples=resample_size, random_state=1)
        s = pd.Series(boot)
        conf = st.t.interval(0.95, len(boot) - 1, loc=s.mean(), scale=st.sem(boot))
        self.meta_info = {'mean': s.mean(), '5_conf': conf[0], '95_conf': conf[1], 'max': s.max(), 'min': s.min()}
        self.fitness = s.mean()
        return self.fitness

    def get_data_list(self) -> List:
        """ Returns the list of parameters for this genome over time for export to spreadsheet. printing, etc"""
        return self.data_list

    def get_name(self) -> str:
        """ Creates and returns a name constructed from the active key/value pairs in the active elements of the chromosomes"""
        d = self.create_dict_from_chromo()
        to_return = ""
        for key, val in d.items():
            to_return += "{}_".format(val)
        return to_return.rstrip("_")

    def to_dict(self) -> Dict:
        """ Returns a Dict that contains all the meta information about this genome (population, generation, etc), and
        the current parameters and values """
        to_return = {}
        to_return[EvolverTags.GENERATION.value] = self.generation
        for key, val in self.meta_info.items():
            to_return[key] = val
        to_return.update(self.create_dict_from_chromo())
        return to_return

    def to_string(self, meta: bool = True, chromo: bool = True) -> str:
        """ Returns a str that contains all the meta information about this genome (population, generation, etc), and
        the current parameters and values """
        to_return = "generation = {}, ".format(self.generation, )
        if meta:
            to_return += "meta: "
            for key, val in self.meta_info.items():
                to_return += "{}:{:.3f}, ".format(key, val)
        if chromo:
            to_return += "chromo: {}".format(self.create_dict_from_chromo(self.chromosome_dict))

        return to_return.rstrip(",")


class EvolutionaryOpimizer:
    """
    Class that manages the evolution of a population of Genomes

    ...

    Attributes
    ----------
    evolve_axis_list:List = []
        The master list of all the EvoveAxis that make up the Genomes.
    current_genome_list:List = []
        The list of currently active Genomes
    all_genomes_list:List = []
        The list of all Genomes, including inactive ones for post-hoc analysis
    best_genome_list:List = []
        The list of highest-fitness Genomes, Typically the top 10% - 50%
    best_genome_history_list:List = []
        The list of the best Genome from each of the generations
    keep_percent:float = 0.1
        The percent to keep in the "best_genome" popuation. Default is 10%
    resample_size:int = 100
        The bootstrap resample population. Default is 100
    num_genomes:int = 10
        The number of "live" Genomes in the population. Default is 10
    generation:int = 0
        The current generation
    logfile_name:str = "defaultLog.txt"
        The name of the debugging logfile. Useful for multithreading debugging
    threads:int = 0
        Number of threads/gpus
    thread_label:str = "gpu"
        The label associated with the threads. Typically this would be "gpu", "tpu", or "cpu"
    last_num_regex = None
        A regex to get the last number in a string. Used to determine which thread a process is running in

    Methods
    -------
    reset(self):
        Resets all the variables. Needed to eliminate class cross-contamination of class-global variables
    log(self, s: str):
        Opens the specifies log file, writes a string, and closes it
    add_axis(self, val_axis: VA.EvolveAxis):
        Adds an EvolveAxis to the master axis list - self.evolve_axis_list
    create_intital_genomes(self, num_genomes: int):
        create the genomes of generation 0
    breed_genomes(self, g1: Genome, g2: Genome, crossover_rate: float, mutation_rate: float) -> Genome:
        Take two parent genomes and breed a child Genome, then mutate that child and return it
    thread_function(self, args: List):
        The function called by the thread pooler. All arguments are passed in in a Dict, including the function that
        will do the model creation and evaluation. The number of the thread is determined and used to configure which
        tensor processor (CPU:x, GPU:x, or TPU:x) this thread will utilize
        utilize.
    run_optimizer(self, eval_func: Callable, save_func: Callable, crossover_rate: float, mutation_rate: float) -> float:
        Method that handles the evolution of a single generation of our population of Genomes, and returns an
        average fitness for the Ensemble associated with the best Genome
    get_ranked_chromosome(self, rank: int = 0) -> Dict:
        Get the Genome of the current nth rank, and return its chromosome Dict
    get_ranked_genome(self, rank: int = 0) -> Genome:
        Get the Genome of the current nth rank, and return it
    save_results(self, file_name: str, data_dict: Dict = None):
        Save the results of this population's evolution to an Excel spreadsheet for post hoc analysis
    to_string(self, meta: bool = True, chromo: bool = True) -> str:
        Returns a string representation of this class
    """
    evolve_axis_list: List = []
    current_genome_list: List = []
    all_genomes_list: List = []
    best_genome_list: List = []
    best_genome_history_list: List = []
    keep_percent: float = 0.1
    resample_size: int = 100
    num_genomes: int = 10
    generation: int = 0
    logfile_name: str = "defaultLog.txt"
    threads: int = 0
    thread_label: str = "gpu"
    last_num_regex = None

    def __init__(self, keep_percent: float = 0.1, pop_size: int = 10, resample_size: int = 100, threads: int = 0,
                 logfile: str = None, thread_label: str = "gpu"):
        """ Ctor - Sets up the the EvolutionaryOpimizer, but does not create the populations, since the
        EvolveAxis haven't been added yet
            Parameters
            ----------
            keep_percent : float
                The number of Genomes to keep from the previous generation. Defaults to 10%
            pop_size : int
                The number of Genomes in the population. Defaults to 10
            resample_size : int
                The bootstap distribution size that we calculate statistics from
            threads : int
                The number of device-specific threads that this class will manage. Default is 0

        """
        self.reset()
        self.keep_percent = keep_percent
        self.num_genomes = pop_size
        self.resample_size = resample_size
        self.threads = threads
        self.thread_label = thread_label
        if logfile != None:
            self.logfile_name = logfile
        try:
            os.remove(self.logfile_name)
        except OSError as e:  ## if failed, report it back to the user ##
            print("Error: %s - %s. Creating file." % (e.filename, e.strerror))

    def reset(self):
        """ Resets all the variables. Needed to eliminate class cross-contamination of class-global variables """
        self.evolve_axis_list = []
        self.all_genomes_list = []
        self.current_genome_list = []
        self.best_genome_list = []
        self.best_genome_history_list = []
        self.keep_percent = 0.1
        self.resample_size = 100
        self.num_genomes = 10
        self.generation = 0
        self.threads = 0
        self.thread_label = "gpu"
        last_num_in_str_re = '(\d+)(?!.*\d)'
        self.last_num_regex = re.compile(last_num_in_str_re)

    def log(self, s: str):
        """ Opens the specifies log file, writes a string, and closes it
            Parameters
            ----------
            s : str
                The string to write to file

        """
        with open(self.logfile_name, "a") as f:
            f.write("{}\n".format(s))

    def add_axis(self, val_axis: VA.EvolveAxis):
        """ Adds an EvolveAxis to the master axis list - self.evolve_axis_list
            Parameters
            ----------
            val_axis : EvolveAxis
                The initialized EvovleAxis

        """
        self.evolve_axis_list.append(val_axis)

    def create_intital_genomes(self, num_genomes: int):
        """ create the genomes of generation 0
            Parameters
            ----------
            num_genomes : int
                The number of Genomes to create as our evolving population

        """
        self.num_genomes = num_genomes
        for i in range(num_genomes):
            # create a genome without parents. This genome will be a member of generation 0
            g = Genome(self.evolve_axis_list, generation=self.generation)
            # append to the list of currently active Genomes
            self.current_genome_list.append(g)
            # append to the list of all Genomes
            self.all_genomes_list.append(g)

    def breed_genomes(self, g1: Genome, g2: Genome, crossover_rate: float, mutation_rate: float) -> Genome:
        """ Take two parent genomes and breed a child Genome, then mutate that child and return it

            Parameters
            ----------
            g1 : Genome
                Parent 1
            g2 : Genome
                Parent 2
            crossover_rate: float
                probability that a chromosome will be selected randomly from p1
            mutation_rate: float
                The generation (as determined by the calling EvolutionaryOpimizer) that this genome belongs to

        """
        g = Genome(self.evolve_axis_list, g1, g2, crossover_rate, generation=self.generation)
        g.mutate(mutation_rate)
        return g

    def thread_function(self, args: Dict):
        """ The function called by the thread pooler. All arguments are passed in in a Dict, including the function that
        will do the model creation and evaluation. The number of the thread is determined and used to configure which
        tensor processor (CPU:x, GPU:x, or TPU:x) this thread will utilize
        utilize.
            Parameters
            ----------
            args : Dict
                The values that are needed to calculate and evaluate fitness. An example would be:
                {EvolverTags.ID.value: "eval_{}".format(i), EvolverTags.FUNCTION.value: eval_func, EvolverTags.GENOME.value: g}
                where i is the index in a list of Genomes, eval_func is a reference to the function that will
                calculate and evaluate fitness, and g is the Genome that contains the parameters to be evaluated

        """
        # get the last number in the thread name. This is how we figure out the id of the device we'll use
        num = self.last_num_regex.search(threading.current_thread().name)
        # create the tf.distribute compatable argument for the device
        thread_str = "{}:{}".format(self.thread_label, int(num.group(0)))
        args[EvolverTags.THREAD_NAME.value] = thread_str

        # get the genome we'll evaluate and delete it from the arguments
        g = args[EvolverTags.GENOME.value]
        del args[EvolverTags.GENOME.value]
        # print("thread_func() args = {}".format(args))
        # evaluate the genome, using  the eval_func from the args Dict
        g.calc_fitness2(args)

    def run_optimizer(self, eval_func: Callable, save_func: Callable, crossover_rate: float,
                      mutation_rate: float) -> float:
        """ Method that handles the evolution of a single generation of our population of Genomes, and returns an
        average fitness for the Ensemble associated with the best Genome

            Parameters
            ----------
            eval_func: Callable
                The function that performs the construction and evaluation of the model
            save_func: Callable
                The function that performs the saving of the ensemble of models that
            crossover_rate: float
                probability that a chromosome will be selected randomly from p1
            mutation_rate: float
                The generation (as determined by the calling EvolutionaryOpimizer) that this genome belongs to
        """
        # increment the current generation first. This way we can tell the difference between these generations and the
        # initial, 'generation 0' Genomes
        self.generation += 1

        # Declare types before the loop so the IDE knows what's going on
        g: Genome
        best_fitness = -1000.0
        # iterate over all the current Genomes
        for g in self.current_genome_list:
            # set up the task list (needed for threading)
            task_list = []
            for i in range(self.num_genomes):
                task = {EvolverTags.ID.value: "eval_{}".format(i), EvolverTags.FUNCTION.value: eval_func,
                        EvolverTags.GENOME.value: g}
                task_list.append(task)
            # A population of 0 means that this is a new Genome. We don't have to re-calculate a Genome's fitness
            if len(g.population) == 0:
                if self.threads == 0:
                    # if there are no threads, call g.calc_fitness directly. This makes debugging MUCH easier
                    for t in task_list:
                        g.calc_fitness2(t)
                else:
                    # if there are threads, execute using the thread pool executing the thread_function with the
                    # task_list as the set of parameters
                    with concurrent.futures.ThreadPoolExecutor(max_workers=self.threads) as executor:
                        executor.map(self.thread_function, task_list)

                # calculate the fitess statistics for the ensemble of models created for the Genome
                fitness = g.calc_fitness_stats(resample_size=self.resample_size)

                # if the fitness is better, save it
                if fitness > best_fitness:
                    result = save_func(g.get_name())
                    self.log(result)
                    best_fitness = fitness

                # log the new Genome, now that the values have been calculated. Note that we log all Genomes,
                # so we can see how effectively we're increasing fitness
                self.log(g.to_string(meta=True, chromo=True))

        # sort the list in place soch that the highest value of fitness is at index zero
        self.current_genome_list.sort(key=lambda x: x.fitness, reverse=True)
        # self.log(self.to_string())

        # determine how many Genomes we're going to keep. If we use the default population size of ten, and the
        # default keep_percent of 10%, then we would keep
        num_best = int(np.ceil(len(self.current_genome_list) * self.keep_percent))
        self.best_genome_list = []

        # build a list of the best performing Genomes by taking the top performing Genome(s) of this generation.
        # This could be the same Genome as the previous generation
        bg = self.current_genome_list[0]
        best_fitness = bg.fitness
        self.best_genome_history_list.append(bg)
        print("best: {}".format(bg.to_string(meta=True, chromo=False)))

        # append the best Genomes to the best_genome_list, and keep track of any new
        # best_fitness (This shouldn't change from above?)
        for i in range(num_best):
            g = self.current_genome_list[i]
            self.best_genome_list.append(g)
            best_fitness = max(best_fitness, g.fitness)

        # clear the current_genome_list out and repopulate
        self.current_genome_list = []
        # first, add the best Genome(s) back in
        for g in self.best_genome_list:
            self.current_genome_list.append(g)

        # randomly breed new genomes with a chance of mutation. Stop when we've generated a population
        # of Genome's we've never had before
        while len(self.current_genome_list) < self.num_genomes:
            # choose two random parents, with replacement
            g1i = random.randrange(len(self.best_genome_list))
            g2i = random.randrange(len(self.best_genome_list))
            g1 = self.best_genome_list[g1i]
            g2 = self.best_genome_list[g2i]

            # create a new Genome for evaluation
            g = self.breed_genomes(g1, g2, crossover_rate, mutation_rate)

            # test against all previous Genomes for a match. If there is, we'll try again
            match = False
            for gtest in self.all_genomes_list:
                if g.equals(gtest):
                    match = True
                    break

            # if there is no match with a previous Genome, add it to the current_genome_list for evaluation
            # and the all_genomes_list history
            if not match:
                self.current_genome_list.append(g)
                self.all_genomes_list.append(g)

        # return the highest fitness for this set of Genomes
        return best_fitness

    def get_ranked_chromosome(self, rank: int = 0) -> Dict:
        """ Get the Genome of the current nth rank, and return its chromosome Dict
            Parameters
            ----------
            rank: int = 0
                The index of the Genome

        """
        self.best_genome_history_list.sort(key=lambda x: x.fitness, reverse=True)
        g = self.best_genome_history_list[rank]
        c = g.chromosome_dict
        return c

    def get_ranked_genome(self, rank: int = 0) -> Genome:
        """ Get the Genome of the current nth rank, and return it
            Parameters
            ----------
            rank: int = 0
                The index of the Genome

        """
        self.best_genome_history_list.sort(key=lambda x: x.fitness, reverse=True)
        g = self.best_genome_history_list[rank]
        return g

    def save_results(self, file_name: str, data_dict: Dict = None):
        """ Save the results of this population's evolution to an Excel spreadsheet for post hoc analysis
            Parameters
            ----------
            file_name: str
                The name of the Excel file
            data_dict: Dict = None
                Optional dictionary of additional information to save

        """
        print("save_results({})".format(file_name))

        # sort the list
        self.best_genome_history_list.sort(key=lambda x: x.fitness, reverse=True)

        # create the setup Dict that will contain the meta information about this run
        setup = {}
        setup["user"] = getpass.getuser()
        setup["date"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        setup["resample_size"] = self.resample_size
        setup["keep percent"] = self.keep_percent
        setup["num genomes"] = self.num_genomes
        if data_dict:
            for key, val in data_dict.items():
                setup[key] = val

        # create an Excel workbook
        wb = eu.ExcelUtils()
        wb.to_excel(file_name)

        # write the setup data to its own tab
        wb.dict_to_spreadsheet("setup", setup)

        # set up the list of best chromosomes. These is a sequential list of the
        # best chromosome for each generations
        chromosome_list = []
        g: Genome
        for g in self.best_genome_history_list:
            chromosome_list.append(g.to_dict())

        # save this list to its own tab
        wb.dict_list_matrix_to_spreadsheet("Chromosomes", chromosome_list)

        # write and close
        wb.finish_up()

    def to_string(self):
        """  Returns a string representation of this class """
        str = "All genomes:\n"
        for g in self.current_genome_list:
            str += g.to_string() + "\n"
        str += "\nBest genomes:\n"
        for g in self.best_genome_list:
            str += g.to_string() + "\n"
        return str


# The following code provides an example of how to use the EvolutionaryOptimizer class

# This is an evaluation function that is passed to the Evolutionary Optomizer. There are three parameters passed in
# using the 'arguments' Dict. X and Y are used to create a surface that can be visualized
# (and are the only values used for an exhaustive search). An additional function parameter is added if available.
# Two Dicts are returned, one with a fitness value, and one with an ID
def example_evaluation_function(arguments: Dict) -> Tuple[Dict, Dict]:
    x = arguments['X'] + random.random() - 0.5
    y = arguments['Y'] + random.random() - 0.5
    val = np.cos(x) + x * .1 + np.sin(y) + y * .1
    if 'Zfunc' in arguments:
        z = arguments['Zfunc'] + random.random() - 0.5
        val += z

    return {EvolverTags.FITNESS.value: val}, {
        EvolverTags.FILENAME.value: "{}.tf".format(arguments[EvolverTags.ID.value])}


# A stub of a save function that
def example_save_function(name: str) -> str:
    return "would have new best value: {}".format(name)


# The next four functions are used as elements of vzfunc EvolveAxis
def plus_func(v1: float, v2: float) -> float:
    return v1 + v2


def minus_func(v1: float, v2: float) -> float:
    return v1 - v2


def mult_func(v1: float, v2: float) -> float:
    return v1 * v2


def div_func(v1: float, v2: float) -> float:
    if v2 > 0:
        return v1 / v2
    return 0


# The main entry point if used as a standalone example
if __name__ == '__main__':
    # create the x and y values for our surface. For this example, these are intervals from -5 to 5, with a step of 0.25
    v1 = VA.EvolveAxis("X", VA.ValueAxisType.FLOAT, min=-5, max=5, step=0.25)
    v2 = VA.EvolveAxis("Y", VA.ValueAxisType.FLOAT, min=-5, max=5, step=0.25)

    # create an Evolve axis that contains a List of functions, and two EvolveAxis that will be the arguments for those functions.
    # First, we create a List of function references
    func_array = [plus_func, minus_func, mult_func, div_func]
    # Next, we create the vzfunc EvolveAxis, using the List of functions
    vzfunc = VA.EvolveAxis("Zfunc", VA.ValueAxisType.FUNCTION, range_array=func_array)
    # Add child EvolveAxis that can provide the arguments to the functions. The order that they are instanced is
    # the order in the function's argument list
    vzvals = VA.EvolveAxis("Zvals1", VA.ValueAxisType.FLOAT, parent=vzfunc, min=0, max=5, step=0.5)
    vzvals = VA.EvolveAxis("Zvals2", VA.ValueAxisType.FLOAT, parent=vzfunc, min=0, max=5, step=0.5)

    # do an exhaustive evaluation for comparison. Each time a new, better value is found, add it to the list for plotting
    prev_fitness = -10
    num_exhaust = 0
    exhaustive_list = []
    for x in range(len(v1.range_array)):
        for y in range(len(v2.range_array)):
            num_exhaust += 1
            args = {'X': v1.range_array[x], 'Y': v2.range_array[y], EvolverTags.ID.value: "eval_[{}]_[{}]".format(x, y)}
            d1, d2 = example_evaluation_function(args)
            cur_fitness = d1[EvolverTags.FITNESS.value]
            if (cur_fitness > prev_fitness):
                prev_fitness = cur_fitness
                exhaustive_list.append(cur_fitness)

    # now do it using evoultionary fitness landscape evaluation
    # create an instance of the EvolutionaryOpimizer that keeps the top 50% of the genomes for each generation.
    # Threads can equal the number of processors. Zero is best for stepping through code in a debugger
    eo = EvolutionaryOpimizer(keep_percent=.5, threads=0)
    # add the EvolveAxis. Order doesn't matter here
    eo.add_axis(v1)
    eo.add_axis(v2)
    eo.add_axis(vzfunc)

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

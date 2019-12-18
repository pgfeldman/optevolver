import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import time
from tensorflow.keras import layers

import optevolver.generators.FloatFunctions as FF


class TF2OptimizationTestBase:
    """
    A base class that provides the framework for evolutionary optimization of architecture and hyperparameter space
    using Tensorflow 2.0 Keras.

    ...

    Attributes
    ----------
    sequence_length: int
        The size of the input and output vector. This assumes a sequence to sequence mapping where the
        sequences are the same length
    model: tf.keras.Sequential
        The default Tensorflow model
    full_mat: np.ndarray
        The full matrix of test/train values. Split into train_mat and test_mat
    train_mat: np.ndarray
        The rows of the full_mat that are used to train the model
    test_mat: np.ndarray
        The rows of the full_mat that are used to evaluate the fitness of the model while it's training
    predict_mat: np.ndarray
        A new matrix that the behavior of the trained model can be evaluated against
    model_args: Dict
        A Dict that is used to store the arguments used for model architecture and hyperparameters for each
        evaluation
    strategy: tf.distribute.OneDeviceStrategy
        The mechanism that TF uses to allocate models across multiple processors
    device: str
        The device that the model is running on

    Methods
    -------
    reset()
       Resets all the variables. Needed to eliminate class cross-contamination of class-global variables
    generate_train_test() -> (np.ndarray, np.ndarray, np.ndarray):
        Generates a set of test and train data for sequence creation
    build_model()
        Overridable method for the construction of the keras model. The default method creates a MLP model with
        variable neurons and layers based on the arguments passed in as a Dict
    evaluate model()
        Overridable method for the training and evaluation of a keras model. The default method varies batch size, and
        epochs based on the arguments passed in as a Dict
    plot_population()
        The normal use case for this class is to produce an ensemble of models that are stored in a directory. This
        method reads in each of the models and evaluates them against new data. Input, target, and predictions are
        generated for the ensembles, whch are plotted using pyplot
    plot_all()
        A convenience method that plots the full_mat, train_mat, test_mat, and predict_mat
    plot_mats()
        Plots a matrix with some adjustable display parameters


    """
    sequence_length: int
    model: tf.keras.Sequential
    full_mat: np.ndarray
    train_mat: np.ndarray
    test_mat: np.ndarray
    predict_mat: np.ndarray
    model_args: Dict
    strategy: tf.distribute.OneDeviceStrategy
    device: str

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
        self.reset()
        self.sequence_length = sequence_length
        self.strategy = tf.distribute.OneDeviceStrategy(device="/{}".format(device))
        self.device = device

    def reset(self):
        """Resets all the variables. Needed to eliminate class cross-contamination of class-global variables"""
        self.sequence_length = 0
        self.model_args = {}
        self.model: tf.keras.Sequential = None
        self.full_mat: np.ndarray = None
        self.train_mat: np.ndarray = None
        self.test_mat: np.ndarray = None
        self.predict_mat: np.ndarray = None
        self.noise = 0
        self.strategy = None
        self.device = "gpu:0"

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


    def build_model(self, args: Dict) -> tf.keras.Sequential:
        """
        Overridable method that builds a tf.keras.Sequential model and returns it. All arguments are passed in the form
        of a Dict, so that there can be any amount of paramaters used to produce the model. In this example case, a
        multilayer perceptron is built using a Dict that would contain {'num_neurons' : x, 'num_layers' : y}. The model
        is defined to run on the device passed in (or the default of 'gpu:0') to the constructor. Returns an executable
        model with input and output size of self.sequence_length

        Parameters
        ----------
        num_functions: int
            the number of sin-based functions to create
        rows_per_function: int
            the number of rows to create for each sin function. With noise == 0, these rows would be identical
        noise: float
            the amount of noise to add to each generated value
        """
        with self.strategy.scope():
            # make a copy of the args that we can manipulate
            self.model_args = args.copy()

            # some defaults
            num_layers = 1
            num_neurons = 200

            #change the defaults if there are args in the Dict to do so
            if 'num_neurons' in self.model_args:
                num_neurons = self.model_args['num_neurons']
            if 'num_layers' in self.model_args:
                num_layers = self.model_args['num_layers']

            # create the model
            self.model = tf.keras.Sequential()

            # Add the input layer with sequence_length units to the model
            self.model.add(layers.Dense(self.sequence_length, activation='relu', input_shape=(self.sequence_length,)))
            # Add the number of layers and neurons as specified in the args
            for i in range(num_layers):
                self.model.add(layers.Dense(num_neurons, activation='relu'))
            # Add the output layer with sequence_length units to the model
            self.model.add(layers.Dense(self.sequence_length))

            # set up our loss and optimization functions. These could be passed in as well
            loss_func = tf.keras.losses.MeanSquaredError()
            opt_func = tf.keras.optimizers.Adam(0.01)

            # create the model
            self.model.compile(optimizer=opt_func,
                               loss=loss_func,
                               metrics=['accuracy'])
        #return the model
        return self.model

    def evaluate_model(self, num_functions: int = 10, rows_per_function: int = 1, noise: float = 0) -> Dict:
        """
        Overridable method that evaluates a tf.keras model and returns a dictionary containing metrics
        (loss, accuracy, duration) about that run
        run.

        Parameters
        ----------
        num_functions: int
            the number of sin-based functions to create that we will evaluate against
        rows_per_function: int
            the number of rows to create for each sin function. With noise == 0, these rows would be identical
        noise: float
            the amount of noise to add to each generated value
        """

        # set the defaults
        self.noise = noise
        epochs = 40
        batch_size = 2

        # update the defaults if 'epochs' or 'batch_size' is in the self.model_args Dict that was passed into
        # build_model()
        if 'epochs' in self.model_args:
            epochs = self.model_args['epochs']
        if 'batch_size' in self.model_args:
            batch_size = self.model_args['batch_size']

        # generate the full, train, and test matrices
        self.full_mat, self.train_mat, self.test_mat = self.generate_train_test(num_functions, rows_per_function, noise)

        # set up values that we will use in the evaluation
        results_dict = {}
        start = time.time()

        # use the target processor
        with self.strategy.scope():
            # fit and evaluate the model, getting the list of results back
            self.model.fit(self.train_mat, self.test_mat, epochs=epochs, batch_size=batch_size)
            result_list = self.model.evaluate(self.train_mat, self.test_mat)
            # build our return values
            stop = time.time()
            loss = result_list[0]
            accuracy = result_list[1]
            duration = stop - start
            # print (helps to see what's going on)
            print("{}: loss = {:.4f}, accuracy = {:.4f}, duration = {:.4f} seconds".format(self.device, loss, accuracy, duration))
            # build the results Dict
            results_dict = {'loss': loss, 'accuracy': accuracy, 'duration': duration}

        # generate some data for testing
        self.full_mat, self.train_mat, self.test_mat = self.generate_train_test(rows_per_function, rows_per_function, noise)

        # calculate the predicted values and save them
        self.predict_mat = self.model.predict(self.train_mat)

        # return the results
        return results_dict

    def plot_population(self, dirname: str, num_functions: int = 10, rows_per_function: int = 1, noise: float = 0):
        """
        loads an ensemble of models and plots their predictions against input values using pyplot

        Parameters
        ----------
        dirname: str
            the name of the root directory that contains the TF2.0 models
        num_functions: int
            the number of sin-based functions to create that we will evaluate against
        rows_per_function: int
            the number of rows to create for each sin function. With noise == 0, these rows would be identical
        noise: float
            the amount of noise to add to each generated value
        """

        # generate a new  full, train, and test matrices
        self.full_mat, self.train_mat, self.test_mat = self.generate_train_test(num_functions, rows_per_function, noise)

        # create the matrix that will store the ensemble values for plotting
        avg_mat = np.zeros(self.test_mat.shape)

        # change to the directory contiaing the models
        d = os.getcwd()
        os.chdir(dirname)
        # iterate over all the child directories
        with os.scandir() as entries:
            count = 1
            for entry in entries:
                #don't do anything if the entry is not a directory
                if entry.is_file() or entry.is_symlink():
                    os.remove(entry.path)
                elif entry.is_dir():
                    count += 1
                    print("loading: {}".format(entry.name))
                    # load the model
                    new_model = tf.keras.models.load_model(entry.name)
                    #generate a prediction matrix
                    self.predict_mat = new_model.predict(self.train_mat)
                    # add these values to the target matrix
                    avg_mat = np.add(self.predict_mat, avg_mat)
                    # add the predict values to the "All predictions" chart. This will give us multiple overlayed lines
                    self.plot_mats(self.predict_mat, rows_per_function, "All Predictions", 0)
        # plot the train and test matrices
        self.plot_mats(self.train_mat, rows_per_function, "Training Set", 1)
        self.plot_mats(self.test_mat, rows_per_function, "Ground Truth", 2)

        # normalize the target matrix we've been summing our values to
        avg_mat = avg_mat / count
        # plot the average of the ensemble
        self.plot_mats(avg_mat, rows_per_function, "Ensemble Average", 3)
        # show the plots
        plt.show()
        # change back
        os.chdir(d)

    def plot_all(self, pop_size:int=10):
        """
        Plot the values of this class's matrices

        Parameters
        ----------
        pop_size: int = 10
            the number of functions that are being used to evaluate this model
        """
        self.plot_mats(self.full_mat, pop_size, "Full Data", 1)
        self.plot_mats(self.train_mat, pop_size, "Input Vector", 2)
        self.plot_mats(self.test_mat, pop_size, "Output Vector", 3)
        self.plot_mats(self.predict_mat, pop_size, "Predict", 4)
        plt.show()

    def plot_mats(self, mat: np.ndarray, cluster_size: int, title: str, fig_num: int, linestyle:str='solid', marker:str='None'):
        """
        Plots a matrix with some adjustable display parameters

        Parameters
        ----------
        mat: np.ndarray
            the matrix we are plotting
        cluster_size: int
            the size of the 'color cluster' in this class, this is usually the rows_per_function argument
        title: str
            the title of the chart
        fig_num: int
            the pyplot figure number
        linestyle:str='solid' (see https://matplotlib.org/3.1.0/gallery/lines_bars_and_markers/linestyles.html)
            the linestyle. Options include solid, dotted, dashed, and dashdot
        marker:str='None' (see matplotlib.org/3.1.1/api/markers_api.html)
            the line marker style. Options include ',' (point), 'o' (circle), 'v' (triangle), 's' (square)

        """

        if title != None:
            plt.figure(fig_num)

        i = 0
        for row in mat:
            cstr = "C{}".format(int(i / cluster_size))
            plt.plot(row, color=cstr, linestyle=linestyle, marker=marker)
            i += 1

        if title != None:
            plt.title(title)

# Exercise this class by warning that it needs a subclass
if __name__ == "__main__":
    print("TF2OptimizerTestBase needs a subclass...")

import os
import random
import pandas as pd
import math # REQUIRED


class FloatFunctions:
    """
    Class that generates floating-point data for an arbitrary function, passed in as a string in the form:

    generate(target_directory_name, "1.0", delta=1.0, noise=0.33)
    generate(target_directory_name, "xx/34", 1.0, noise=0.33)
    generate(target_directory_name, "1.0-xx/34", 1.0, noise=0.33)
    generate(target_directory_name, "math.sin(xx)/2+0.5", 0.33, noise=0.33)
    ...

    Attributes
    ----------
    num_rows:int = 0
        The number of rows to generate
    sequence_length:int = 0
        How many values each row has
    step:int = 0
        The amount the input value for each row should increment with respect to the previous row. A step of 1,
        on a function of 'xx' would give the following
        0.0 1.0 2.0 3.0 4.0 5.0
        1.0 2.0 3.0 4.0 5.0 6.0
        2.0 3.0 4.0 5.0 6.0 7.0
        A step of 0 disables this feature.
    ragged:int = 0
        The number of values that can be randomly deleted from the end of a sequence
    write_header:bool = False
        Flag for writing meta information at row 0
    df: pd.DataFrame
        The pandas.DataFrame that stores the matrix of results


    Methods
        generateDataFrame(self, function:str="xx", delta:float=1.0, noise:float=0.0, seed:int=1) -> pd.DataFrame:
            Generates a pandas.DataFrame containing the generated data
        generate(self, directory="./", function="xx", delta=1.0, noise=0.0, seed=1):
            Writes out a .csv file containing the generated data
    -------
    """

    num_rows:int = 0
    sequence_length:int = 0
    step:int = 0
    ragged:int = 0
    write_header:bool = False
    df: pd.DataFrame

    def __init__(self, rows: int = 100, sequence_length: int = 24, step: int = 0, ragged: int = 0,
                 write_header: bool = False):
        """ Constructor: sets up the parameters for generating the data

        Parameters
        ----------
        num_rows:int
            The number of rows in the generated sample. Default is 100
        sequence_length:int
            The number of columns in the generated data. Default is 24 (hours per day, if you were wondering)
        step = step
            The amount the input value for each row should increment with respect to the previous row. A step of 1,
            on a function of 'xx' would give the following
            0.0 1.0 2.0 3.0 4.0 5.0
            1.0 2.0 3.0 4.0 5.0 6.0
            2.0 3.0 4.0 5.0 6.0 7.0
            A step of 0 disables this feature.
        ragged = ragged
            The number of values that can be randomly deleted from the end of a sequence
        write_header = write_header
            Flag for writing meta information at row 0
        """
        self.num_rows = rows
        self.sequence_length = sequence_length
        self.step = step
        self.ragged = ragged
        self.write_header = write_header

    def generateDataFrame(self, function:str="xx", delta:float=1.0, noise:float=0.0, seed:int=1) -> pd.DataFrame:
        """ Method that populates a pandas.dataframe based on the values sent to the constructor, and the
            string equation that is passed in. 
            
            Parameters
            ----------
            function: str
                A string like "math.sin(xx)/2+0.5" that is interpreted by eval(). Yes, this is dangerous. Don't pass in 
                anything stupid. Default is 'xx'
            delta: float = 1.0
                The amount that xx is incremented by. Default is 1.0
            noise: float = 0.0
                A randomly generated value to be added to each generated value. The noise value calculation is:
                (random.random() * 2.0 - 1.0) * noise
            seedL int = 1
                A seed value for the random generator so that results with noise can be repeatedly generated
        """

        random.seed(seed)
        # code to step the function goes here
        mat = []
        xx = 0
        for row in range(0, self.num_rows):
            row_str = ""
            max_col = self.sequence_length
            if self.ragged > 0:
                max_col -= random.randrange(self.ragged)

            row = []
            for col in range(0, max_col):
                result = eval(function)
                result += (random.random() * 2.0 - 1.0) * noise
                row.append(result)
                xx += delta
            row_str = row_str.rstrip(",")
            mat.append(row)
            if self.step == 0:
                xx = 0
            else:
                xx -= delta * (max_col - self.step)
        self.df = pd.DataFrame(mat)
        return self.df

    def generate(self, directory="./", function="xx", delta=1.0, noise=0.0, seed=1):
        """ Method that populates a .csv file, based on the values sent to the constructor, and the
            string equation that is passed in.

            Parameters
            ----------
            directory: str = './'
                The directory that the file will be written to. The file will have the name of the function that
                was passed in, with the special characters filtered out. so 'xx*2.0' would be saved as
                'float_function_xxtimes2.0_100_lines_25
            function: str
                A string like "math.sin(xx)/2+0.5" that is interpreted by eval(). Yes, this is dangerous. Don't pass in
                anything stupid. Default is 'xx'
            delta: float = 1.0
                The amount that xx is incremented by. Default is 1.0
            noise: float = 0.0
                A randomly generated value to be added to each generated value. The noise value calculation is:
                (random.random() * 2.0 - 1.0) * noise
            seedL int = 1
                A seed value for the random generator so that results with noise can be repeatedly generated
        """
        random.seed(seed)
        clean_func = function.replace("*", "times")
        clean_func = clean_func.replace("/", "div")
        clean_func = clean_func.replace("+", "add")
        clean_func = clean_func.replace("-", "sub")
        file_name = "{}float_function_{}_lines_{}_sequence_{}_ragged_{}.csv".format(directory, clean_func,
                                                                                    self.num_rows, self.sequence_length,
                                                                                    self.ragged)
        print("saving as ", file_name)

        if (os.path.exists(file_name)):
            os.remove(file_name)

        f = open(file_name, "x")
        if self.write_header:
            config_line = '"function":{}, "rows":{}, "sequence_length":{}, "step":{}, "delta":{}, "ragged":{}, "type":"floating_point"'.format(
                function, self.num_rows, self.sequence_length, self.step, delta, self.ragged)
            f.write("# confg: {}\n".format(config_line))

        # code to step the function goes here
        xx = 0
        for row in range(0, self.num_rows):
            row_str = ""
            max_col = self.sequence_length
            if self.ragged > 0:
                max_col -= random.randrange(self.ragged)
            for col in range(0, max_col):
                result = eval(function)
                result += (random.random() * 2.0 - 1.0) * noise
                row_str += "{:.2f},".format(result)
                xx += delta
            row_str = row_str.rstrip(",")
            f.write("{}\n".format(row_str))
            if self.step < 0:
                xx = 0
            else:
                xx -= delta * (max_col - self.step)
        f.flush()

# Exercises the FloatFunctions.
if __name__ == "__main__":
    ff = FloatFunctions(rows=25, sequence_length=100)
    # ff = FloatFunctions(rows = 25, sequence_length=34, ragged=10)

    df = ff.generateDataFrame("math.sin(xx)/2+0.5", 0.33)
    # df = ff.generateDataFrame("math.sin(xx)", noise=0.1)
    print(df)
    # ff.generate("../../data/input_data/", "math.sin(xx/2.0)*math.sin(xx/4.0)*math.cos(xx/8.0)", noise=0.0)
    # ff.generate("./", "math.sin(xx)*math.sin(xx/2.0)*math.cos(xx/4.0)", noise=0.1)
    # ff.generate("../clustering/", "math.sin(xx)*math.sin(xx/2.0)", noise=0.1)
    # ff.generate("../../data/input_data/", "1.0", delta=1.0)
    # ff.generate("../../data/input_data/", "xx/34", 1.0)
    # ff.generate("../../data/input_data/", "1.0-xx/34", 1.0)
    # ff.generate("../../data/input_data/", "math.sin(xx)/2+0.5", 0.33)

    # ff.generate("../../data/input_data/", "1.0", delta=1.0, noise=0.33)
    # ff.generate("../../data/input_data/", "xx/34", 1.0, noise=0.33)
    # ff.generate("../../data/input_data/", "1.0-xx/34", 1.0, noise=0.33)
    # ff.generate("../../data/input_data/", "math.sin(xx)/2+0.5", 0.33, noise=0.33)

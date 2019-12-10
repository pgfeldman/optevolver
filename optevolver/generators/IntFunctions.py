import os
import numpy
import math # this is needed in eval!!!
import sys

class IntFunctions:
    def __init__(self, rows=100, sequence_length=10, step = 1, min = 0, max = 100):
        self.num_rows = rows
        self.sequence_length = sequence_length
        self.step = step
        self.min = min
        self.max = max

    def gen_float_data(self, function = "xx", delta = 1.0):
        # first, generate the floating point version of the data, save it and get min/max for the dataset
        float_mat = numpy.empty((self.num_rows, self.sequence_length))
        xx = 0
        fmin = sys.float_info.max
        fmax = sys.float_info.min
        for row in range(0, self.num_rows):
            for col in range(0, self.sequence_length):
                result = eval(function)
                if result > fmax:
                    fmax = result
                if result < fmin:
                    fmin = result
                float_mat[row][col] = result
                xx += delta
            xx -= delta*(self.sequence_length-self.step)
        source_range = fmax - fmin
        target_range = self.max - self.min
        scalar = target_range / source_range
        return float_mat, scalar, fmin

    def generate(self, directory = "./", function = "xx", delta = 1.0):
        float_mat, scalar, fmin = self.gen_float_data(function, delta)

        #Now that we have the data, scalars and offsets, turn into integers and write to file
        file_name = "{}int_function_lines_{}_sequence_{}.txt".format (directory, self.num_rows, self.sequence_length)
        print ("saving as ", file_name)

        if(os.path.exists(file_name)):
            os.remove(file_name)

        f = open(file_name, "x")
        config_line = '"function":{}, "rows":{}, "sequence_length":{}, "step":{}, "delta":{}, "type":"floating_point"'.format(function, self.num_rows, self.sequence_length, self.step, delta)
        f.write("#confg: {"+config_line+"}\n")

        xx = 0
        for row in range(0, self.num_rows):
            for col in range(0, self.sequence_length):
                result = float_mat[row][col]
                result = (result - fmin) * scalar + self.min
                f.write("{}, ".format(int(result)))
                xx += delta
            f.write("\n")
            xx -= delta*(self.sequence_length-self.step)
        f.flush()

    def generate_text(self, token_file = "../../input_data/test.txt", directory = "./", function = "xx", delta = 1.0):
        # get the tokens
        f = open(token_file)
        token_list = list()
        for line in f:
            words = line.replace(",", "").split()
            if words[0].isnumeric():
                token_list.append(words[1])

        # compute the function
        float_mat, scalar, fmin = self.gen_float_data(function, delta)

        #Now that we have the data, scalars and offsets, turn into words and write to file
        file_name = "{}words_function_lines_{}_sequence_{}.txt".format (directory, self.num_rows, self.sequence_length)
        print ("saving as ", file_name)

        if(os.path.exists(file_name)):
            os.remove(file_name)

        f = open(file_name, "x")
        config_line = '"function":{}, "rows":{}, "sequence_length":{}, "step":{}, "delta":{}, "type":"words"'.format(function, self.num_rows, self.sequence_length, self.step, delta)
        f.write("#confg: {"+config_line+"}\n")

        xx = 0
        for row in range(0, self.num_rows):
            for col in range(0, self.sequence_length):
                result = float_mat[row][col]
                result = (result - fmin) * scalar + self.min
                f.write("{}, ".format(token_list[int(result)]))
                xx += delta
            f.write("\n")
            xx -= delta*(self.sequence_length-self.step)
        f.flush()

# Exercises the IntFunctions.
if __name__ == "__main__":
    intf = IntFunctions(sequence_length=20)
    intf.generate("../../data/input_data/", "math.sin(xx)*math.sin(xx/2.0)*math.cos(xx/4.0)", 0.4)
    intf.generate_text("../../data/input_data/callofthewild_tokens.txt", "../../data/input_data/", "math.sin(xx)*math.sin(xx/2.0)*math.cos(xx/4.0)", 0.4)
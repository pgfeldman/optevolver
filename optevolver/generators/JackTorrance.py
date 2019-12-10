import os

class JackTorrance:
    all_words_array = ["all ", "work ", "and ", "no ", "play ", "makes ", "jack ", "a ", "dull ", "boy "]
    #all_words_array = ["zero ", "one ", "two ", "three ", "four ", "five ", "six ", "seven ", "eight ", "nine "]
    def __init__(self, lines=100, words=26, step = 26):
        self.num_lines = lines
        self.words_per_line = words
        self.step = step

    def generate(self, directory = "./"):
        file_name = "{}torrance_{}_lines_{}_words.txt".format (directory, self.num_lines, self.words_per_line)
        print ("saving as ", file_name)

        if(os.path.exists(file_name)):
            os.remove(file_name)

        f = open(file_name, "x")
        config_line = '"rows":{}, "sequence_length":{}, "step":{}, "type":"words"'.format(self.num_lines, self.words_per_line, self.step)
        f.write("#confg: {"+config_line+"}\n")
        cur_word = 0
        for lines in range(0, self.num_lines):
            for words in range(0, self.words_per_line):
                index = cur_word % len(self.all_words_array)
                f.write(self.all_words_array[index])
                #print(all_work[index], end="")
                cur_word += 1
            f.write("\n")
            cur_word = (lines+1)*self.step
        f.flush()

    def generate_indices(self, directory = "./"):
        file_name = "{}torrance_{}_lines_{}_indices.txt".format (directory, self.num_lines, self.words_per_line)
        print ("saving as ", file_name)

        if(os.path.exists(file_name)):
            os.remove(file_name)

        f = open(file_name, "x")
        config_line = '"rows":{}, "sequence_length":{}, "step":{}, "type":"integer"'.format(self.num_lines, self.words_per_line, self.step)
        f.write("#confg: {"+config_line+"}\n")
        cur_word = 0
        for lines in range(0, self.num_lines):
            for words in range(0, self.words_per_line):
                index = cur_word % len(self.all_words_array)
                f.write("{}, ".format(index))
                #print(all_work[index], end="")
                cur_word += 1
            f.write("\n")
            cur_word = (lines+1)*self.step
        f.flush()

# Exercises the JackTorrance class.
if __name__ == "__main__":
    jt = JackTorrance(step=3)
    #jt.generate("../../data/input_data/")
    jt.generate_indices("../../data/input_data/")
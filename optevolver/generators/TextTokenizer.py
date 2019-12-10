import os
import re

class TextTokenizer:
    line_regex = re.compile(r"([-!?\.]\"|[!?\.])")
    punctuation_regex = re.compile(r"[\",\-\_)(;“”]")
    document_array = []
    document_set = set()

    def __init__(self, input_dir = "../../data/input_data/", input_file="test.txt"):
        path = input_dir+input_file
        in_file = open(path, "r", encoding="utf8")
        self.name = input_file.replace(".txt", "")

        #read file into memory
        #TODO: create a sentence buffer so that the number of characters between punctuation is kept under the specified max
        for i, line in enumerate(in_file):
            line = self.punctuation_regex.sub("",line)
            line = line.replace(".", " . ")
            line = line.replace("?", " ? ")
            line = line.replace("!", " ! ")
            line_array = line.lower().split()
            self.document_array.extend(line_array)
            self.document_set.update(line_array)
            print(line_array)

    def tokenize(self, output_dir="./", max_chars = 280):
        path = output_dir+self.name+"_tokens.txt"
        print ("saving as ", path)

        if(os.path.exists(path)):
            os.remove(path)

        out_file = open(path, "x")

        out_file.write("index, token\n")
        # code to step the function goes here
        #write out the numeric token list

        for i, t in enumerate(self.document_set):
            out_file.write("{}, {}\n".format(i, t))
        out_file.flush()

    def generate_text(self, output_dir ="./", sequence_length=10, step=1):
        file_name = "{}text_tokenizer_{}_sequence_length_{}_step.txt".format (output_dir, sequence_length, step)
        print ("saving as ", file_name)

        if(os.path.exists(file_name)):
            os.remove(file_name)

        f = open(file_name, "x")
        config_line = '"sequence_length":{}, "step":{}, "type":"words"'.format(sequence_length, step)
        f.write("#confg: {"+config_line+"}\n")
        cur_word = 0
        cur_col = 0
        row_string = ""
        # code to step the function goes here
        while cur_word < len(self.document_array):
            word = self.document_array[cur_word]
            row_string += word
            #f.write(word)
            cur_word += 1
            cur_col += 1
            if cur_col >= sequence_length:
                cur_col = 0
                cur_word -= (sequence_length-step)
                f.write(row_string+"\n")
                row_string = ""
            else:
                row_string += ", "
                #f.write(", ")
        f.flush()

    def generate_indices(self, output_dir ="./", sequence_length=10, step=1):
        file_name = "{}{}_index_tokenizer_{}_sequence_length_{}_step.txt".format (output_dir, self.name, sequence_length, step)
        print ("saving as ", file_name)

        if(os.path.exists(file_name)):
            os.remove(file_name)

        f = open(file_name, "x")
        config_line = '"sequence_length":{}, "step":{}, "type":"integer"'.format(sequence_length, step)
        f.write("#confg: {"+config_line+"}\n")
        unique_word_list = list(self.document_set)
        cur_word = 0
        cur_col = 0
        row_string = ""
        # code to step the function goes here
        while cur_word < len(self.document_array):
            word = self.document_array[cur_word]
            index = unique_word_list.index(word)
            #f.write("{}".format(index))
            row_string += "{}".format(index)
            cur_word += 1
            cur_col += 1
            if cur_col >= sequence_length:
                cur_col = 0
                cur_word -= (sequence_length-step)
                f.write(row_string+"\n")
                row_string = ""
            else:
                row_string += ", "
                #f.write(", ")
        f.flush()

# Exercises the JackTorrance class.
if __name__ == "__main__":
    tt = TextTokenizer("../../data/input_data/", "callofthewild.txt")
    tt.tokenize("../../data/input_data/")
    tt.generate_text("../../data/input_data/")
    tt.generate_indices("../../data/input_data/")
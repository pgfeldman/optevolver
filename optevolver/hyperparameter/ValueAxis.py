import random
from enum import Enum
from typing import Dict, List


class ValueAxisType(Enum):
    """A class containing enumerations for ValueAxis and EvolveAxis variables"""
    STRING = 0
    INTEGER = 1
    FLOAT = 2
    FUNCTION = 3
    VALUEAXIS = 4
    UNSET = 5


class ValueAxis:
    """
    Class that contains data and a range of values to iterate over. ValueAxis can have one child that can then in turn be iterated, ad infinitum

    ...

    Attributes
    ----------
    name: str
        The name of the variable manipulated in ths class
    ntype: ValueAxisType
        The type that the class should return. Helpful for some logic that changes for strings, functions and numbers
    min: number
        If the type is a number, the class will create a range_array between the min and max values, using the step
    max: number
        If the type is a number, the class will create a range_array between the min and max values, using the step
    step: number
        If the type is a number, the class will create a range_array between the min and max values, using the step
    index: int
        The position in the range_array
    range_array: List
        The array of values (strings, functions, ValueAxis, numbers) that the class will iterate over
    child: ValueAxis
        A nexted class that will iterate completely for every step of the parent class
    cur_val:
        The current value of class (i.e. range_array[index])

    Methods
    -------
    reset()
        Resets all the variables. Needed to eliminate class cross-contamination of class-global variables
    def convert_type(label) -> ValueAxisType:
        Takes a string or ValueAxisType and if it can match, returns the appropriate ValueAxisType
    cascading_step() -> bool:
        Returns the next value in a stack of one or more nested ValueAxis'. The bottom ValueAxis is incremented by one step
        on its range_array. If all ranges at all levels have been iterated, cascading_step() returns True, otherwise, False.
        Values for each range are accessed by get_cur_var() from each range_array.
    get_cur_val():
        Returns the value that is pointed to by range_array[index]. Can be one of any of the types in ValueAxisType
    get_size():
        Returns the number of items that this ValueAxis will iterate over
    to_string():
        Returns a string representation of this class
    """
    name = "unset"
    ntype = ValueAxisType.INTEGER
    min = 0
    max = 0
    step = 0
    index = 0
    range_array = []
    child = None
    cur_val = None

    def __init__(self, name: str, child: 'ValueAxis', ntype: ValueAxisType, min=0, max=0, step=0, range_array=[]):
        """
        Parameters
        ----------
        name : str
            The name of the variable
        child: ValueAxis
            A child ValueAxis to completely step through for every single step if self
        ntype: ValueAxisType
            The type of the variable. Can be STRING, INTEGER, FLOAT, FUNCTION, VALUEAXIS, or UNSET
        min: number (Default = 0)
            If the type is a number, the class will create a range_array between the min and max values, using the step
        max: number (Default = 0)
            If the type is a number, the class will create a range_array between the min and max values, using the step
        step: number (Default = 0)
            If the type is a number, the class will create a range_array between the min and max values, using the step
        range_array: List
        The array of values (strings, functions, ValueAxis, numbers) that the class will iterate over
        """
        self.reset()
        self.name = name
        self.child = child
        self.ntype = self.convert_type(ntype)
        if len(range_array) == 0:
            if ntype == ValueAxisType.INTEGER:
                self.min = int(min)
                self.max = int(max)
                self.step = int(step)
                for val in range(self.min, self.max, self.step):
                    self.range_array.append(val)
            elif ntype == ValueAxisType.FLOAT:
                self.min = float(min)
                self.max = float(max)
                self.step = float(step)
                val = self.min
                while val < self.max:
                    self.range_array.append(val)
                    val += self.step
        else:
            self.range_array = range_array.copy()
        self.cur_val = self.range_array[self.index]

    def reset(self):
        """Resets all the variables. Needed to eliminate class cross-contamination of class-global variables"""
        self.ntype = ValueAxisType.INTEGER
        self.min = 0
        self.max = 0
        self.step = 0
        self.index = 0
        self.cur_val = None
        self.range_array = []
        self.child = None
        self.name = "unset"

    def convert_type(self, label) -> ValueAxisType:
        """Takes a string or ValueAxisType and if it can match, returns the appropriate ValueAxisType

        Parameters
        ----------
        label:
            Either a ValueAxisType (in which case, it is returned unchanged), or a string, which can be:
                'string', 'integer', 'float', 'function', or 'valueaxis'
        """
        if isinstance(label, str):
            if label.lower() == "string":
                return ValueAxisType.STRING

            if label.lower() == "integer":
                return ValueAxisType.INTEGER

            if label.lower() == "float":
                return ValueAxisType.FLOAT

            if label.lower() == "function":
                return ValueAxisType.FUNCTION

            if label.lower() == "valueaxis":
                return ValueAxisType.VALUEAXIS
        return label

    def cascading_step(self) -> bool:
        """Steps to the next value in a stack of one or more nested ValueAxis'. The bottom ValueAxis is incremented by one step
        on its range_array.

        If all ranges at all levels have been iterated, cascading_step() returns True, otherwise, False.
        Values for each range are accessed by get_cur_var() from each range_array.
        """
        self.cur_val = self.range_array[self.index]
        # print("{} cur_val = {}".format(self.name, self.cur_val))

        child_complete = True
        if self.child:
            child_complete = self.child.cascading_step()

        if child_complete:
            self.index += 1
            if self.index >= len(self.range_array):
                self.index = 0
                return True
        return False

    def get_cur_val(self):
        """Returns the value that is pointed to by range_array[index]. Can be one of any of the types in ValueAxisType"""
        if (self.ntype == ValueAxisType.INTEGER):
            return int(self.cur_val)
        return self.cur_val

    def get_size(self) -> int:
        """Returns the number of items that this ValueAxis will iterate over"""
        return len(self.range_array)

    def to_string(self):
        """Returns a string representation of the current value and the range array for this and any child ValueAxis(s)"""
        str = "{}:\n\tcontents = {}".format(self.name, self.range_array)
        if self.child != None:
            str += "\n\tchild = {}".format(self.child.name)
        return str


########################################################

class EvolveAxis:
    """
    Class that contains data and a range of values to evolve over. EvolveAxis can have a parent that they are associated with
    so clusters of values can evolve in a correct context. For example if one value in the range_array was the function 'foo(a, b)'
    and the next was 'bar(c)', a nd be should only evolve when foo() is the active function

    ...

    Attributes
    ----------
    name: str
       The name of the variable manipulated in ths class
    parent: EvolveAxis
        A parent EvolveAxix that this EvolveAxis may associate with to cluster evolution steps contextually
    children: List
        An array of children that have this EvolveAxis set as their parent
    ntype: ValueAxisType
       The type that the class should return. Helpful for some logic that changes for strings, functions and numbers
    min: number
       If the type is a number, the class will create a range_array between the min and max values, using the step
    max: number
       If the type is a number, the class will create a range_array between the min and max values, using the step
    step: number
       If the type is a number, the class will create a range_array between the min and max values, using the step
    index: int
       The position in the range_array
    range_array: List
       The array of values (strings, functions, ValueAxis, numbers) that the class will iterate over
    cur_val:
        The current value of class (i.e. range_array[index])
    history_list: List
        A list of all calculated values, appended to each time calc_random_val() is called

    Methods
    -------
    reset()
       Resets all the variables. Needed to eliminate class cross-contamination of class-global variables
    convert_type(label) -> ValueAxisType:
       Takes a string or ValueAxisType and if it can match, returns the appropriate ValueAxisType
    set_value(): EvolveAxis
        Takes an EvolveAxis that is assumed to be the same, and recursively sets the index and cur_value of all child EvolveAxis
    get_random_val():
        Computes a random value for the self.index, which then points to a new value. If the value_type is a FUNCTION, then 
        the child EvolveAxis are called as well. If the ntype is VALUEAXIS, a similar process to FUNCTION should happen, but this is 
        only stubbed out with the default behavior. The results of this calculation are stored in self.result, which is returned
    get_result():
       Returns self.result
    get_last_history(): Dict
        Returns the last entry in self.history_list
    get_indexed_history(index): Dict
        Returns self.history_list[index]
    get_num_history(): int
        Returns the number of elements in the history array
    add_to_dict(d)
        Adds a key/value pair to d where key is the name of the EvolveAxis, and value is self.result, or the name of the function
    get_size():
        Returns the number of items that this ValueAxis will iterate over
    to_string():
       Returns a string representation of this class
       """
    name = ValueAxisType.UNSET
    parent = None
    children = []
    ntype = ValueAxisType.INTEGER
    min = 0
    max = 0
    step = 0
    index = 0
    range_array = []
    history_list = []
    cur_val = None
    result = None

    def __init__(self, name: str, ntype: ValueAxisType, parent: "EvolveAxis" = None, min=0, max=0, step=0,
                 range_array=[]):
        """
       Parameters
       ----------
       name : str
           The name of the variable
       ntype: ValueAxisType
           The type of the variable. Can be STRING, INTEGER, FLOAT, FUNCTION, VALUEAXIS, or UNSET
       parent: ValueAxis, optional
           An optional parent ValueAxis that this ValueAxis can coordinate with during evolution
       min: number (Default = 0), optional
           If the type is a number, the class will create a range_array between the min and max values, using the step
       max: number (Default = 0), optional
           If the type is a number, the class will create a range_array between the min and max values, using the step
       step: number (Default = 0), optional
           If the type is a number, the class will create a range_array between the min and max values, using the step
       range_array: List, optional
       The array of values (strings, functions, ValueAxis, numbers) that the class will iterate over
       """
        self.reset()
        self.name = name
        self.ntype = self.convert_type(ntype)

        if parent != None:
            self.parent = parent
            parent.children.append(self)

        if len(range_array) == 0:
            if ntype == ValueAxisType.INTEGER:
                self.min = int(min)
                self.max = int(max)
                self.step = int(step)
                for val in range(self.min, self.max, self.step):
                    self.range_array.append(val)
            elif ntype == ValueAxisType.FLOAT:
                self.min = float(min)
                self.max = float(max)
                self.step = float(step)
                val = self.min
                while val < self.max:
                    self.range_array.append(val)
                    val += self.step
        else:
            self.range_array = range_array.copy()
        self.cur_val = self.range_array[self.index]

    def reset(self):
        """Resets all the variables. Needed to eliminate class cross-contamination of class-global variables"""
        self.ntype = ValueAxisType.INTEGER
        self.min = 0
        self.max = 0
        self.step = 0
        self.index = 0
        self.cur_val = None
        self.result = None
        self.range_array = []
        self.name = "unset"
        self.parent = None
        self.children = []
        self.history_list = []

    def convert_type(self, label) -> ValueAxisType:
        """Takes a string or ValueAxisType and if it can match, returns the appropriate ValueAxisType

        Parameters
        ----------
        label:
            Either a ValueAxisType (in which case, it is returned unchanged), or a string, which can be:
                'string', 'integer', 'float', 'function', or 'valueaxis'
        """
        if isinstance(label, str):
            if label.lower() == "string":
                return ValueAxisType.STRING

            if label.lower() == "integer":
                return ValueAxisType.INTEGER

            if label.lower() == "float":
                return ValueAxisType.FLOAT

            if label.lower() == "function":
                return ValueAxisType.FUNCTION

            if label.lower() == "valueaxis":
                return ValueAxisType.VALUEAXIS
        return label

    def set_value(self, ea: "EvolveAxis"):
        """Takes an EvolveAxis that is assumed to be the same, and recursively sets the index and cur_value of all child EvolveAxis

        Parameters
        ----------
        ea:
            The EvolveAxis that we wish to be set to match, including all our children
        """
        self.index = ea.index
        self.cur_val = self.result = ea.cur_val

        if self.ntype == ValueAxisType.FUNCTION:
            args = []
            c: EvolveAxis
            for i in range(len(ea.children)):
                c = self.children[i]
                eac = ea.children[i]
                c.set_value(eac)
                args.append(c.get_result())
            self.result = self.cur_val(*args)

        d = {}
        self.add_to_dict(d)
        self.history_list.append(d)
        return self.result

    def get_random_val(self):
        """Computes a random value for the self.index, which then points to a new value. If the value_type is a
        FUNCTION, then the child EvolveAxis are called as well. If the ntype is VALUEAXIS, a similar process to FUNCTION should happen, but this is
        only stubbed out with the default behavior. The results of this calculation are stored in self.result, which is returned"""

        self.index = random.randrange(len(self.range_array))
        self.cur_val = self.result = self.range_array[self.index]

        if self.ntype == ValueAxisType.FUNCTION:
            # print("func name = {}".format(self.cur_val.__name__))
            args = []
            c: EvolveAxis
            for c in self.children:
                args.append(c.get_random_val())
            self.result = self.cur_val(*args)

        elif self.ntype == ValueAxisType.VALUEAXIS:
            self.result = self.cur_val.get_random_val()

        d = {}
        self.add_to_dict(d)
        self.history_list.append(d)
        return self.result

    def get_result(self):
        """Returns self.result"""

        return self.result

    def get_last_history(self) -> Dict:
        """Returns the last entry in self.history_list"""
        return self.history_list[-1]

    def get_indexed_history(self, index: int) -> Dict:
        """Returns self.history_list[index]

        Parameters
        ----------
        index:
            The index of the history item, where [0] is first
        """
        result = self.history_list[index]
        return result

    def get_num_history(self) -> int:
        """Returns the number of elements in the history array"""
        return len(self.history_list)

    def add_to_dict(self, d: Dict):
        """Adds a key/value pair to d where key is the name of the EvolveAxis, and value is self.result, or the name of the function

        Parameters
        ----------
        d:
            The dictionary that we will add entries to
        """
        if self.cur_val != None:
            if self.ntype == ValueAxisType.FUNCTION:
                d["{}".format(self.name)] = self.get_result()
                cv = self.cur_val
                d["{}_function".format(self.name)] = cv.__name__
                # d["result"] = self.get_result()
                c: EvolveAxis
                for c in self.children:
                    c.add_to_dict(d)
            else:
                d["{}".format(self.name)] = self.get_result()
                # d["result"] = self.get_result()

    def get_size(self) -> int:
        """Returns the number of items that this ValueAxis will iterate over"""
        return len(self.range_array)

    def to_string(self):
        """Returns a string representation of this class"""
        cv = "None"
        if self.cur_val != None:
            cv = self.cur_val
        if self.ntype == ValueAxisType.FUNCTION:
            cv = cv.__name__
        elif self.ntype == ValueAxisType.VALUEAXIS:
            cv = cv.get_cur_val()
        str = "{}: cur_value = {}".format(self.name, cv)
        c: EvolveAxis
        for c in self.children:
            str += "\n\t{}".format(c.to_string())
        return str


##########################################################################
# Example usage, evaluation and class exercising

def plus_func(v1: float, v2: float) -> float:
    return v1 + v2


def minus_func(v1: float, v2: float) -> float:
    return v1 - v2


def mult_func(v1: float, v2: float) -> float:
    return v1 * v2


def div_func(v1: float, v2: float) -> float:
    return v1 / v2


def exercise_ValueAxis():
    # Create three ValueAxis, one for float, one for int, and one for strings, pre-set in an array
    name_list = ["phil", "aaron", "jeff"]

    # Create the ValueAxis. In this case, 'v3' is the 'top' of the stack, in that it has 'v2 as a child, who has
    # 'v1' as a child. What this means is that 'v1' will go though one complete iteration for every single step
    # of 'v2' and so on
    v1 = ValueAxis("float vals", child=None, ntype=ValueAxisType.FLOAT, min=0, max=1, step=0.25)
    v2 = ValueAxis("int vals", child=v1, ntype=ValueAxisType.INTEGER, min=0, max=4, step=1)
    v3 = ValueAxis("names", child=v2, ntype=ValueAxisType.FLOAT, range_array=name_list)

    total_permutations = v1.get_size() * v2.get_size() * v3.get_size()
    print("Permuting data {} steps:".format(total_permutations))
    # iterate until all permutations are exhausted
    done = False
    count = 1
    while not done:
        print("\tstep {}: {} = {}. {} = {}, {} = {}".format(count, v1.name, v1.get_cur_val(), v2.name, v2.get_cur_val(),
                                                            v3.name, v3.get_cur_val()))
        # cause the next set of permutation to happen
        done = v3.cascading_step()
        count += 1


def exercise_EvolveAxis():
    """ This method creates and and exercises a set of EvolveAxis - one function (defined above), and two floating point arguments. """
    # create a list of functions that all take two arguments (functions are defined above)
    func_array = [plus_func, minus_func, mult_func, div_func]

    # create the parent axis
    eafunc = EvolveAxis("Zfunc", ValueAxisType.FUNCTION, range_array=func_array)

    # create the child axis
    ea_arg_1 = EvolveAxis("Zvals1", ValueAxisType.FLOAT, parent=eafunc, min=1, max=5, step=0.5)
    ea_arg_2 = EvolveAxis("Zvals2", ValueAxisType.FLOAT, parent=eafunc, min=1, max=5, step=0.5)

    total_combinations = eafunc.get_size() * ea_arg_1.get_size() * ea_arg_2.get_size()
    print("Generating data from a total space of {} combinations:".format(total_combinations))
    # run five times
    count = 5
    for i in range(count):
        # get a random value based on random selections from the parent and child EvolveAxis
        eafunc.get_random_val()

        # get the current number of histories we have
        index = eafunc.get_num_history()

        # get a (Dict) representation of the current state of the history stack
        d = eafunc.get_last_history()

        # print it
        print("\tvzfunc[{}]: {} = {}".format(index, eafunc.name, d))

    print("Recorded Data:")
    # go through the history so we can verufy that it is the same as what we just generated
    for i in range(count):
        d = eafunc.get_indexed_history(i)
        print("\tvzfunc[{}]: {} = {}".format(i, eafunc.name, d))


# The main entry point
if __name__ == '__main__':
    print("Exercising ValueAxis")
    exercise_ValueAxis()
    print("\nExercising EvolveAxis")
    exercise_EvolveAxis()

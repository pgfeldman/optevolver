import json
import pprint
from typing import List, Dict, Set


class JsonUtils:
    json_dict = {}
    json_str = ""

    def __init__(self, filename:str = None):
        if filename:
            f = open(filename, 'r', encoding="utf8")
            lines = f.read()
            # print("lines = {}".format(lines))
            f.close()
            self.parse(lines)

    def parse(self, json_str:str):
        self.json_str = json_str
        self.set_dict(json.loads(json_str))

    def set_dict(self, source_dict:Dict):
        self.json_dict = source_dict

    def get_dict(self):
        return self.json_dict

    def get_str(self):
        return self.json_str

    def lfind(self, query_list:List, target_list:List, targ_str:str = "???"):
        for tval in target_list:
            #print("lfind: tval = {}, query_list[0] = {}".format(tval, query_list[0]))
            if isinstance(tval, dict):
                val = self.dfind(query_list[0], tval, targ_str)
                if val:
                    return val
            elif tval == query_list[0]:
                return tval

    def dfind(self, query_dict:Dict, target_dict:Dict, targ_str:str = "???"):
        for key, qval in query_dict.items():
            tval = target_dict[key]
            #print("dfind: key = {}, qval = {}, tval = {}".format(key, qval, tval))
            if isinstance(qval, dict):
                val =  self.dfind(qval, tval, targ_str)
                if val:
                    return val
            elif isinstance(qval, list):
                return self.lfind(qval, tval, targ_str)
            else:
                if qval == targ_str:
                    return tval
                if qval != tval:
                    break

    def find(self, query_dict:Dict):
        # pprint.pprint(query_dict)
        result = self.dfind(query_dict, self.json_dict)
        return result

    def pprint(self):
        pprint.pprint(self.json_dict)


if __name__ == '__main__':
    ju = JsonUtils("../../data/output_data/lstm_structure.json")
    # ju.pprint()
    result = ju.find({"config":[{"class_name":"Masking", "config":{"batch_input_shape": "???"}}]})
    print("result 1 = {}".format(result))
    result = ju.find({"config":[{"class_name":"Masking", "config":{"mask_value": "???"}}]})
    print("result 2 = {}".format(result))

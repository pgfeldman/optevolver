import argparse
from enum import Enum
from typing import List, Dict

import numpy as np
import pandas
import xlsxwriter


class ExcelUtils:
    """
    Class that handles reading and writing Dicts to Excel

    ...

    Attributes
    ----------
    workbook:xlsxwriter.Workbook = None
        an xlsxwriter workbook for writing out some styles of data
    pandas_xlsx:pandas.ExcelFile = None
        an Pandas Excel for reading and writing out some styles of data


    Methods
    -------
    to_excel(self, file_name: str):
    def finish_up(self):
    def dict_list_matrix_to_spreadsheet(self, sheet_name, dict_list: List):
    def dict_to_spreadsheet(self, sheet_name, val_dict: Dict):

    read_dataframe_spreadsheet(self, file_name: str, sheet_name: str = None, header=None, index_col=None,
                                   transpose: bool = False) -> pandas.DataFrame:
    read_dataframe_excel(self, file_name: str, sheet_name: str = None, header=None, index_col=None,
                                 transpose: bool = False) -> pandas.DataFrame:
    read_dataframe_sheet(self, sheet_name: str, header=None, index_col=None, transpose: bool = False) -> pandas.DataFrame:
    read_dataframe_csv(file_name: str, header=None, index_col=None, transpose: bool = False) -> pandas.DataFrame:

    """
    workbook:xlsxwriter.Workbook = None
    pandas_xlsx:pandas.ExcelFile = None

    ########### Xlsxwriter methods

    def to_excel(self, file_name: str):
        """ Instantiates a xlsxwriter.Workbook that we can write to
            Parameters
            ----------

            file_name: str
                The name of the file we're going to write to
        """
        self.workbook = xlsxwriter.Workbook(file_name)

    def finish_up(self):
        """ Close the workbook """
        self.workbook.close()

    def dict_list_matrix_to_spreadsheet(self, sheet_name, dict_list: List):
        """ Write out a list of Dict values as a spreadsheet

            Parameters
            ----------

            sheet_name: str
                The name of the sheet (or tab) in the workbook
            dict_list: List
                A list of dicts (e.g. [{}, {}, {}]
        """
        worksheet = self.workbook.add_worksheet(name=sheet_name)

        header = dict_list[0].keys()
        col = 0
        for key in header:
            if isinstance(key, Enum):
                key = key.value
            worksheet.write(0, col, key)
            col += 1

        row = 0
        for entry in dict_list:
            row += 1
            col = 0
            for key in header:
                val = entry[key]
                worksheet.write(row, col, val)
                col += 1

    def dict_to_spreadsheet(self, sheet_name, val_dict: Dict):
        """ Write a Dict to a workbook sheet where each key/value pair is a row. Good for a page describing an
            experiment.

            Parameters
            ----------

            sheet_name: str
                The name of the sheet (or tab) in the workbook
            val_dict: List
                A dict containing a set of key/value pairs
        """
        worksheet = self.workbook.add_worksheet(name=sheet_name)

        row = 0
        for key in val_dict:
            val = val_dict[key]
            worksheet.write(row, 0, key)
            worksheet.write(row, 1, val)
            row += 1

    ############ Pandas methods

    def read_dataframe_spreadsheet(self, file_name: str, sheet_name: str = None, header: int=None, index_col:int=None,
                                   transpose: bool = False) -> pandas.DataFrame:
        """ Read an Excel sheet from an Excel .xlsx or .csv file. Returns a Pandas.Dataframe if successful.

            Parameters
            ----------

            file_name: str
                The name of the file to read. Can be either .xlsx or .csv
            sheet_name: str
                The (optional) name of the sheet within the workbook to read
            header : int, list of int, default 0
                Row (0-indexed) to use for the column labels of the parsed DataFrame. If a list of integers is
                passed those row positions will be combined into a MultiIndex. Use None if there is no header.
            index_col : int, list of int, default None
                Column (0-indexed) to use as the row labels of the DataFrame. Pass None if there is no such column.
                If a list is passed, those columns will be combined into a MultiIndex. If a subset of data is selected
                with usecols, index_col is based on the subset.
            transpose: bool, default False
                A flag to return the file as read in or its transpose
        """
        if file_name.endswith(".xlsx"):
            return self.read_dataframe_excel(file_name, sheet_name=sheet_name, header=header, index_col=index_col,
                                             transpose=transpose)
        elif file_name.endswith(".csv"):
            return self.read_dataframe_csv(file_name, header=header, index_col=index_col, transpose=transpose)

        return None

    def read_dataframe_excel(self, file_name: str, sheet_name: str = None, header=None, index_col=None,
                             transpose: bool = False) -> pandas.DataFrame:
        """ Read an Excel sheet from an Excel .xlsx file. Returns a Pandas.Dataframe if successful.

            Parameters
            ----------

            file_name: str
                The name of the file to read. Can be either .xlsx or .csv
            sheet_name: str
                The (optional) name of the sheet within the workbook to read
            header : int, list of int, default 0
                Row (0-indexed) to use for the column labels of the parsed DataFrame. If a list of integers is
                passed those row positions will be combined into a MultiIndex. Use None if there is no header.
            index_col : int, list of int, default None
                Column (0-indexed) to use as the row labels of the DataFrame. Pass None if there is no such column.
                If a list is passed, those columns will be combined into a MultiIndex. If a subset of data is selected
                with usecols, index_col is based on the subset.
            transpose: bool, default False
                A flag to return the file as read in or its transpose
        """
        self.pandas_xlsx = pandas.ExcelFile(file_name)
        if self.pandas_xlsx:
            if sheet_name:
                return self.read_dataframe_sheet(sheet_name, header=header, index_col=index_col)
            # otherwise, read the default
            df = pandas.read_excel(self.pandas_xlsx, header=header, index_col=index_col)
            mat = df.to_numpy()  # get the data matrix
            mat = mat.astype(np.float64)  # force it to float64
            indices = df.index.values
            cols = df.columns.values
            df = pandas.DataFrame(mat, indices, cols)
            if transpose:
                return df.T
            return df
        return None

    def read_dataframe_sheet(self, sheet_name: str, header=None, index_col=None,
                             transpose: bool = False) -> pandas.DataFrame:
        """ Read an Excel sheet from the current self.pandas_xlsx workbook. Returns a Pandas.Dataframe if successful.

            Parameters
            ----------
            sheet_name: str
                The name of the sheet within the workbook to read
            header : int, list of int, default 0
                Row (0-indexed) to use for the column labels of the parsed DataFrame. If a list of integers is
                passed those row positions will be combined into a MultiIndex. Use None if there is no header.
            index_col : int, list of int, default None
                Column (0-indexed) to use as the row labels of the DataFrame. Pass None if there is no such column.
                If a list is passed, those columns will be combined into a MultiIndex. If a subset of data is selected
                with usecols, index_col is based on the subset.
            transpose: bool, default False
                A flag to return the file as read in or its transpose
        """
        if self.pandas_xlsx and sheet_name:
            df = pandas.read_excel(self.pandas_xlsx, sheet_name=sheet_name, header=header, index_col=index_col)
            mat = df.values  # get the data matrix
            mat = mat.astype(np.float64)  # force it to float64
            indices = df.index.values
            cols = df.columns.values
            df = pandas.DataFrame(mat, indices, cols)
            if transpose:
                return df.T
            return df
        return None

    @staticmethod
    def read_dataframe_csv(file_name: str, header=None, index_col=None, transpose: bool = False) -> pandas.DataFrame:
        """ Read a .csv file. Returns a Pandas.Dataframe if successful.

            Parameters
            ----------

            file_name: str
                The name of the file to read. Can be either .xlsx or .csv
            header : int, list of int, default 0
                Row (0-indexed) to use for the column labels of the parsed DataFrame. If a list of integers is
                passed those row positions will be combined into a MultiIndex. Use None if there is no header.
            index_col : int, list of int, default None
                Column (0-indexed) to use as the row labels of the DataFrame. Pass None if there is no such column.
                If a list is passed, those columns will be combined into a MultiIndex. If a subset of data is selected
                with usecols, index_col is based on the subset.
            transpose: bool, default False
                A flag to return the file as read in or its transpose
        """
        df = pandas.read_csv(file_name, header=header, index_col=index_col)
        if transpose:
            return df.T
        return df

    @staticmethod
    def write_dataframe_excel(df: pandas.DataFrame, file_name: str, sheet_name: str = 'Sheet1', writer:pandas.ExcelWriter=None,
                              save_file: bool = True) -> pandas.ExcelWriter:
        """ Write a pandas.DataFrame to a file. If writer is not defined, this method will define one and
            return it

            Parameters
            ----------

            df: pandas.DataFrame
                The DataFrame we are going to be writing
            file_name: str
                The name of the file to be written
            sheet_name: str, default 'Sheet1'
                The name of the sheet in this workbook
            writer:pandas.ExcelWriter, default None
                The writer for this DataFrame. maintained if there are multiple sheets to write
            save_file: bool, default True
                A flag to write out the file
        """
        if not writer:
            writer = pandas.ExcelWriter(file_name)
        df.to_excel(writer, sheet_name=sheet_name)
        if save_file:
            writer.save()
        return writer

    @staticmethod
    def write_dict_excel(dict: Dict, file_name: str, sheet_name: str = 'Sheet1', writer=None,
                         save_file: bool = True) -> pandas.ExcelWriter:
        df = pandas.DataFrame(dict)
        return ExcelUtils.write_dataframe_excel(df, file_name, sheet_name, writer, save_file)


def main():
    """ The standalone version of this class that reads in and prints out a dataframe to exercise the methods"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--excelfile", type=str, help="Excel file name")
    parser.add_argument("--sheet_name", type=str, help="sheet within.xlsx file")
    parser.add_argument("--header_index", type=str, help="header row number (optional)")
    parser.add_argument("--row_id_index", type=str, help="row id column index (optional)")

    args = parser.parse_args()

    eu = ExcelUtils()

    if args.excelfile:
        header = None
        row_index = None
        sheet_name = None
        if args.header_index:
            header = args.header_index
        if args.row_id_index:
            row_index = args.row_id_index
        if args.sheet_name:
            sheet_name = args.sheet_name

        # def read_dataframe_spreadsheet(self, file_name: str, sheet_name: str=None, header = None, index_col = None) -> pandas.DataFrame:
        df = eu.read_dataframe_excel(args.excelfile, sheet_name, header, row_index)

        print("{} header = {}, col_index = {}, sheet = {}".format(args.excelfile, sheet_name, header, row_index))
    else:
        name_array = ["NoHeadersNoNames", "HeadersNoNames", "NamesNoHeaders", "NamesHeaders"]
        postfix_array = ["csv", "xlsx"]

        for name in name_array:
            for postfix in postfix_array:
                header = 0
                row_index = 0
                filename = "{}.{}".format(name, postfix)
                if name.find("NoHeaders") != -1:
                    header = None
                if name.find("NoNames") != -1:
                    row_index = None
                print("\n----------------\n{} header = {}, col_index = {}".format(filename, header, row_index))
                df = eu.read_dataframe_spreadsheet(filename, header=header, index_col=row_index)
                print(df)

# entry point
if __name__ == "__main__":
    main()

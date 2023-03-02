# ==============================================================================
# File: dataset_validator.py
# Project: allison
# File Created: Tuesday, 22nd November 2022 8:29:55 pm
# Author: Dillon Koch
# -----
# Last Modified: Tuesday, 22nd November 2022 8:29:56 pm
# Modified By: Dillon Koch
# -----
#
# -----
# class for validating a dataset is clean and updating values
# ==============================================================================

import sys
from os.path import abspath, dirname

import numpy as np
import pandas as pd

ROOT_PATH = dirname(dirname(abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)


class Dataset_Validator:
    def __init__(self):
        pass

    def set_to_nan(self, series, valid_types=None, invalid_vals=None, min_val=None, max_val=None):  # Run
        """
        Iterating through a pd.Series, setting values to np.nan if conditions are met
        - valid_types describes the only types a value can be, or it's set to nan
        - invalid_vals sets any value in the list to nan
        - min/max val give ranges the values are allowed to be
        """
        vals = list(series)
        for i, val in enumerate(vals):

            valid_type_check = (valid_types is not None) and not isinstance(val, valid_types)
            invalid_val_check = (invalid_vals is not None) and val in invalid_vals
            min_val_check = (min_val is not None) and val < min_val
            max_val_check = (max_val is not None) and val > max_val

            if any([valid_type_check, invalid_val_check, min_val_check, max_val_check]):
                vals[i] = np.nan
                # print(f"{val} -> np.nan")

        return pd.Series(vals)

    def replace_vals(self, series, replace_val, new_val):  # Run
        """
        Goes through every value in a pd.Series, replacing "replace_val" with "new_val"
        - if value isn't a string, just keeping the original
        """
        vals = list(series)
        for i, val in enumerate(series):
            if isinstance(val, str) and replace_val in val:
                vals[i] = val.replace(replace_val, new_val)
                print(f"{val} -> {vals[i]}")

        return pd.Series(vals)

    def replace_vals_equal(self, series, replace_val, new_val, set_lower=False):  # Run
        """
        Replacing values in a series that are fully equal to replace_val with new_val
        """
        vals = list(series)
        for i, val in enumerate(vals):
            if isinstance(val, str):
                val = val.lower() if set_lower else val
                if val == replace_val:
                    vals[i] = new_val

        return pd.Series(vals)

    def split_vals(self, series, split_str, keep_before, set_to_str=True, start=None, end=None):  # Run
        """
        splitting values in a pd.Series on a split_str, keeping the part before or afer
        """
        start = start if start is not None else 0
        end = end if end is not None else 1000
        vals = list(series)

        for i, val in enumerate(vals):
            val = str(val) if set_to_str else val
            if isinstance(val, str) and split_str in val[start:end]:
                idx = 0 if keep_before else 1
                vals[i] = val.split(split_str)[idx]
                print(f"{val} -> {vals[i]}")

        return pd.Series(vals)

    def float_if_possible(self, series):  # Run
        """
        setting values to float if possible, otherwise keeping the original
        """
        vals = list(series)
        for i, val in enumerate(vals):
            try:
                vals[i] = float(val)
            except BaseException:
                vals[i] = val

        return pd.Series(vals)

    def dataset_info(self, df):  # QA Testing
        """
        prints out information about a dataset and its values to check if values make sense
        """
        print("-" * 100)
        print("COLUMNS:\n")
        for column in list(df.columns):
            numeric = isinstance(list(df[column])[0], (int, float))
            if numeric:
                min_val = min([val for val in df[column] if isinstance(val, (int, float))])
                max_val = max([val for val in df[column] if isinstance(val, (int, float))])
                print(f"\t{column}: {min_val} - {max_val}")
            else:
                num_vals = len(set(list(df[column])))
                print(f"\t{column}: {num_vals} unique values")


if __name__ == '__main__':
    x = Dataset_Validator()
    self = x

# ==============================================================================
# File: test_data_schema.py
# Project: allison
# File Created: Wednesday, 1st March 2023 6:14:59 am
# Author: Dillon Koch
# -----
# Last Modified: Wednesday, 1st March 2023 6:15:00 am
# Modified By: Dillon Koch
# -----
#
# -----
# using a data schema to test that a dataset is formatted as it should be
# ==============================================================================


import datetime
import re
import sys
from os.path import abspath, dirname

import numpy as np
from tqdm import tqdm

ROOT_PATH = dirname(dirname(dirname(abspath(__file__))))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)

from src.utilities.match_team import Match_Team
from src.utilities.misc import load_json


class Test_Data_Schema:
    def __init__(self, league):
        self.league = league
        self.match_team = Match_Team(league)

    def test_type(self, val, type_str, allow_null):  # Top Level
        if not type_str:
            return

        if allow_null and ((isinstance(val, float) and np.isnan(val)) or (val is None)):
            return

        if not val and not allow_null and val != 0:
            raise ValueError(f"val {val} cannot be null")

        d = {"string": str, "int": int, "float": (float, int), "Date": str,
             "Datetime": str}

        desired_type = d[type_str]
        assert isinstance(val, desired_type)

        if desired_type == "Date":
            assert re.match(r"\d{4}-\d{2}-\d{2}", val)
        elif desired_type == "Datetime":
            assert re.match(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}", val)

    def test_min(self, val, input_min):  # Top Level
        if (isinstance(val, float) and np.isnan(val)) or (val is None):
            return

        if input_min and val:
            assert val >= input_min

    def test_max(self, val, input_max):  # Top Level
        if (isinstance(val, float) and np.isnan(val)) or (val is None):
            return

        if input_max:
            assert val <= input_max

    def test_whitelist(self, val, whitelist):  # Top Level
        """
        whitelists: team names
        """
        if whitelist == "teams":
            assert val in self.match_team.valid_teams, f"Team {val} not in valid teams!"

    def test_regex(self, val, input_regex):  # Top Level
        pass

    def test_null(self, schema, df):  # Top Level
        pass

    def run(self, schema_path, df):  # Run
        schema = load_json(schema_path)
        features = list(schema.keys())

        for feature in tqdm(features):
            print(feature)
            feature_d = schema[feature]
            vals = list(df[feature])
            for val in vals:
                self.test_type(val, feature_d['type'], feature_d['allow_null'])
                self.test_min(val, feature_d['min'])
                self.test_max(val, feature_d['max'])

                self.test_whitelist(val, feature_d['whitelist'])
                self.test_regex(val, feature_d['regex'])
                self.test_null(val, feature_d['allow_null'])
        print("DATA SCHEMA TESTS PASS")


if __name__ == '__main__':
    x = Test_Data_Schema()
    self = x
    x.run()

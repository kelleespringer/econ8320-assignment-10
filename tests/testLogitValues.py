import unittest
import json
import sys as system
import io
import pandas as pd
import re
import numpy as np


with open("Lesson.ipynb", "r") as file:
    f_str = file.read()

doc = json.loads(f_str)

code = [i for i in doc['cells'] if i['cell_type']=='code']
si = {}
for i in code:
    for j in i['source']:
        if "#si-exercise" in j:
            exec(compile("".join(i['source']), '<string>', 'exec'))

class TestCase(unittest.TestCase):

    def testLogitCorrectValues(self):
        data = pd.read_csv("tests/files/assignment8Data.csv")
        x = data.loc[:100, ['sex', 'age', 'educ']]
        y = data.loc[:100, 'white']
        reg = RegressionModel(x, y, create_intercept=True, regression_type='logit')
        reg.fit_model()

        # Expected values
        expected_values = {
            'sex': {'coefficient': -1.1229, 'standard_error': 0.3980, 'z_stat': -2.8215, 'p_value': 0.00478},
            'age': {'coefficient': -0.0070, 'standard_error': 0.0108, 'z_stat': -0.6472, 'p_value': 0.5175},
            'educ': {'coefficient': -0.0465, 'standard_error': 0.1010, 'z_stat': -0.4602, 'p_value': 0.6453},
            'intercept': {'coefficient': 5.7354, 'standard_error': 1.1266, 'z_stat': 5.0908, 'p_value': 3.56499e-07}
        }

        # Debug output: print results from the model
        print("Debug: Model Coefficients:", reg.results.coefficients)
        print("Debug: Model Standard Errors:", reg.results.standard_errors)
        print("Debug: Model Z-Statistics:", reg.results.test_statistics)
        print("Debug: Model P-Values:", reg.results.p_values)

        # Check each variable
        for var in ['sex', 'age', 'educ', 'intercept']:
            print(f"Debug: Checking {var}")
            self.assertAlmostEqual(reg.results.coefficients[var], expected_values[var]['coefficient'], places=4)
            self.assertAlmostEqual(reg.results.standard_errors[var], expected_values[var]['standard_error'], places=4)
            self.assertAlmostEqual(reg.results.test_statistics[var], expected_values[var]['z_stat'], places=4)
            self.assertAlmostEqual(reg.results.p_values[var], expected_values[var]['p_value'], places=4)

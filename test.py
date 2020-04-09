import unittest
from ESRNN.M4_data import prepare_M4_data, FREQ_DICT
from ESRNN.utils_evaluation import evaluate_prediction_owa
from ESRNN import ESRNN
import pandas as pd
import numpy as np
import os
import argparse
import yaml

def test_dates(dataset_name, n_obs=1000000):

    dir = os.environ.get('TEST_DATE_ESRNN_DIR')

    assert dir, "Define env var TEST_DATE_ESRNN_DIR"

    # Open config for
    config_file = './configs/{}.yaml'.format(dataset_name)
    with open(config_file, 'r') as stream:
        config = yaml.safe_load(stream)

    X_train_df, y_train_df, X_test_df, y_test_df = prepare_M4_data(dataset_name=dataset_name, directory = dir, num_obs=n_obs)

    # Model part
    frcy_model = FREQ_DICT[dataset_name]
    model = ESRNN(max_epochs=1, batch_size = 10,
                  max_periods=config['data_parameters']['max_periods'],
                  seasonality=config['data_parameters']['seasonality'],
                  input_size=config['data_parameters']['input_size'],
                  output_size=config['data_parameters']['output_size'],
                  frequency=config['data_parameters']['frequency'])

    model.fit(X_train_df, y_train_df, X_test_df, y_test_df)

    # Predict on test set
    y_hat_df = model.predict(X_test_df)

    evals = evaluate_prediction_owa(y_hat_df, y_train_df, X_test_df, y_test_df, naive2_seasonality=1)

    # Getting freqs
    freqs = [df.groupby('unique_id')['ds'].apply(lambda df: df.dt.freq).unique() for df in (X_train_df, y_train_df, X_test_df, y_test_df, y_hat_df)]
    freqs = list(np.unique(freqs))

    # Checking cols of test
    bool_test_hat_equals = X_test_df.equals(y_hat_df[X_test_df.columns])

    # Non null predicitons
    null_preds = all(evals)

    return freqs, bool_test_hat_equals, null_preds


class TestDates(unittest.TestCase):
    #directory = sys.argv[1]

    def test_daily(self):

        print('\n\nTesting Daily data')
        freqs, equals, null_preds = test_dates('Daily')

        self.assertEqual(freqs, [np.array('D')])
        self.assertTrue(equals)
        self.assertTrue(null_preds)

        del freqs, equals, null_preds

    def test_hourly(self):

        print('\n\nTesting Hourly data')
        freqs, equals, null_preds = test_dates('Hourly')

        self.assertEqual(freqs, [np.array('H')])
        self.assertTrue(equals)
        self.assertTrue(null_preds)

        del freqs, equals, null_preds

    def test_monthly(self):

        print('\n\nTesting Monthly data')
        freqs, equals, null_preds = test_dates('Monthly')

        self.assertEqual(freqs, [np.array('MS')])
        self.assertTrue(equals)
        self.assertTrue(null_preds)

        del freqs, equals, null_preds

    def test_quarterly(self):

        print('\n\nTesting Quarterly data')
        freqs, equals, null_preds = test_dates('Quarterly')

        self.assertEqual(freqs, [np.array('QS-OCT')])
        self.assertTrue(equals)
        self.assertTrue(null_preds)

        del freqs, equals, null_preds

    def test_weekly(self):

        print('\n\nTesting Weekly data')
        freqs, equals, null_preds = test_dates('Weekly')

        self.assertEqual(freqs, [np.array('W-SUN')])
        self.assertTrue(equals)
        self.assertTrue(null_preds)

        del freqs, equals, null_preds

    def test_yearly(self):

        print('\n\nTesting Yearly data')
        freqs, equals, null_preds = test_dates('Yearly')

        self.assertEqual(freqs, ['AS-JAN'])
        self.assertTrue(equals)
        self.assertTrue(null_preds)

        del freqs, equals, null_preds



if __name__ == '__main__':
    dir = os.environ.get('TEST_DATE_ESRNN_DIR')

    assert dir, "Define env var TEST_DATE_ESRNN_DIR"

    unittest.main()

import unittest
import os
from pathlib import Path
import shutil
from datetime import datetime
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from ds_core.properties.property_manager import PropertyManager
from ds_capability import *
from ds_capability.components.commons import Commons
from ds_nn.megatron.tokenizers.column_coder import code_schema, ColumnCodes

# Pandas setup
pd.set_option('max_colwidth', 320)
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 99)
pd.set_option('expand_frame_repr', True)


class ColumnCoderTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_for_smoke(self):
        fe = FeatureEngineer.from_memory()
        fs = FeatureSelect.from_memory()
        _ = fs.add_connector_uri('card', uri="s3://project-hadron-cs-repo/downloads/data/card_transaction_750000.csv")
        card = fs.load_canonical('card')
        card = fe.tools.correlate_replace(card, header='Amount', pattern='$', replacement='')
        card = fs.tools.auto_cast_types(card, inc_category=False)

    def test_code_schema(self):
        fe = FeatureEngineer.from_memory()
        fs = FeatureSelect.from_memory()
        _ = fs.add_connector_uri('card', uri="~/code/jupyter/neural_network/nemo/source/card_transaction.5000.csv")
        card = fs.load_canonical('card')
        card = fe.tools.correlate_replace(card, header='Amount', pattern='$', replacement='', to_header='Amount')
        card = fe.tools.model_drop_columns(card, headers='mask')
        card = fe.tools.correlate_date_element(card, header='Time', elements={'hr': 'Hour', 'min': 'Minute'},
                                              date_format='%H:%M:%S', drop_header=True)
        card = fs.tools.auto_cast_types(card, include_category=False, include_timestamp=False)
        float_columns = ['Amount']
        category_columns = ['Year', 'Month', 'Day', 'Hour', 'Minute', 'Card', 'Use Chip',
                            'Merchant Name', 'Merchant City', 'Merchant State', 'MCC',
                            'Errors?']
        integer_columns = ['Zip']
        tab_structure, example_tabs = code_schema(card, target='User', int_columns=integer_columns,
                                                  float_columns=float_columns, category_columns=category_columns)
        cc = ColumnCodes.get_column_codes(tab_structure, example_tabs)

        float_str = '12.7'
        token_ids = cc.encode('Amount', float_str)
        print('token ids for {} is {}'.format(float_str, token_ids))
        amt_str = cc.decode('Amount', token_ids)
        print('recovered Amt for {} is {}'.format(float_str, amt_str))

        int_str = '10345'
        token_ids = cc.encode('Zip', int_str)
        print('token ids for {} is {}'.format(int_str, token_ids))
        amt_str = cc.decode('Zip', token_ids)
        print('recovered Zip for {} is {}'.format(int_str, amt_str))

        city_str = 'ONLINE'
        token_ids = cc.encode('Merchant City', city_str)
        print('token ids for {} is {}'.format(city_str, token_ids))
        amt_str = cc.decode('Merchant City', token_ids)
        print('recovered City for {} is {}'.format(city_str, amt_str))



    def test_raise(self):
        startTime = datetime.now()
        with self.assertRaises(KeyError) as context:
            env = os.environ['NoEnvValueTest']
        self.assertTrue("'NoEnvValueTest'" in str(context.exception))
        print(f"Duration - {str(datetime.now() - startTime)}")


def tprint(t: pa.table, headers: [str, list] = None, d_type: [str, list] = None, regex: [str, list] = None):
    _ = Commons.filter_columns(t.slice(0, 10), headers=headers, d_types=d_type, regex=regex)
    print(Commons.table_report(_).to_string())


if __name__ == '__main__':
    unittest.main()

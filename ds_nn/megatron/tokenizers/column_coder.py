# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# created under: https://github.com/NVIDIA/NeMo

import math
from typing import Dict, List, Tuple
import pyarrow as pa
import pyarrow.compute as pc
import numpy as np
from numpy import ndarray
from sklearn.preprocessing import PowerTransformer, QuantileTransformer, RobustScaler

__all__ = ["IntCode", "FloatCode", "CategoryCode", "ColumnCodes", 'code_schema']


def code_schema(canonical : pa.Table, target: str):
    if target in canonical.column_names:
        canonical.drop_columns(target)
    tab_structure = []
    example_tabs = {}
    for n in canonical.column_names:
        c = canonical.column(n)
        if pa.types.is_floating(c.type):
            item = {
                "name": c,
                "code_type": "float",
                "args": {
                    "code_len": 3,  # number of tokens used to code the column
                    "base": 32,  # the positional base number. ie. it uses 32 tokens for one digit
                    "fillall": True,  # whether to use full base number for each token or derive it from the data.
                    "hasnan": False,  # can it handles nan or not
                    "transform": "yeo-johnson",  # can be ['yeo-johnson', 'quantile', 'robust'],
                }
            }
        elif pa.types.is_integer(c.type):
            item = {
                "name": c,
                "code_type": "int",
                "args": {
                    "code_len": 3,  # number of tokens used to code the column
                    "base": 47,  # the positional base number. ie. it uses 32 tokens for one digit
                    "fillall": True,  # whether to use full base number for each token or derive it from the data.
                    "hasnan": True,  # can it handles nan or not
                }
            }
        elif pa.types.is_string(c.type) or pa.types.is_dictionary(c.type):
            item = {
                "name": c,
                "code_type": "category",
            }
        else:
            raise ValueError(f"The column '{n}' of type '{c.type}' is not supported")
        tab_structure.append(item)
        # example tabs
        for col in tab_structure:
            if col['code_type'] == "category":
                example_tabs[n] = pc.unique(pc.utf8_trim_whitespace(c))
            else:
                example_tabs[n] = pc.unique(pc.drop_null(c))

    return tab_structure, example_tabs


class Code(object):
    def compute_code(self, data_series: ndarray):
        """
        @params:
            data_series: an array of input data used to calculate mapping
        """
        raise NotImplementedError()

    def __init__(self, col_name: str, code_len: int, start_id: int, fillall: bool = True, hasnan: bool = True):
        """
        @params:
            col_name: name of the column
            code_len: number of tokens used to code the column.
            start_id: offset for token_id.
            fillall: if True, reserve space for digit number even the digit number is
            not present in the data_series. Otherwise, only reserve space for the numbers
            in the data_series.
            hasnan: if True, reserve space for nan
        """
        self.name = col_name
        self.code_len = code_len
        self.start_id = start_id
        self.end_id = start_id
        self.fillall = fillall
        self.hasnan = hasnan

    def encode(self, item: str) -> List[int]:
        raise NotImplementedError()

    def decode(self, ids: List[int]) -> str:
        raise NotImplementedError()

    @property
    def code_range(self) -> List[Tuple[int, int]]:
        """
        get the vocab id range for each of the encoded tokens
        @returns [(min, max), (min, max), ...]
        """
        return [(self.start_id, self.end_id)]


class IntCode(Code):
    def __init__(
        self, col_name: str, code_len: int, start_id: int, fillall: bool = True, base: int = 100, hasnan: bool = True
    ):
        super().__init__(col_name, code_len, start_id, fillall, hasnan)
        self.NA_token_id = None
        self.NA_token = None
        self.digits_item_to_id = None
        self.digits_id_to_item = None
        self.base = base
        self.int_min: int = None

    def compute_code(self, data_series: ndarray):
        significant_val = self.array_convert_to_int(data_series)

        digits_id_to_item = [{} for _ in range(self.code_len)]
        digits_item_to_id = [{} for _ in range(self.code_len)]
        for i in range(self.code_len):
            id_to_item = digits_id_to_item[i]
            item_to_id = digits_item_to_id[i]
            v = (significant_val // self.base ** i) % self.base
            if self.fillall:
                uniq_items = range(0, self.base)
            else:
                uniq_items = sorted(np.unique(v).tolist())
            for k in range(len(uniq_items)):
                item = str(uniq_items[k])
                item_to_id[item] = self.end_id
                id_to_item[self.end_id] = item
                self.end_id += 1
        self.digits_id_to_item = digits_id_to_item
        self.digits_item_to_id = digits_item_to_id
        self.NA_token = 'nan'
        if self.hasnan:
            self.end_id += 1  # add the N/A token
            codes = []
            ranges = self.code_range
            for i in ranges:
                codes.append(i[1] - 1)
            self.NA_token_id = codes

    def array_convert_to_int(self, val: ndarray):
        val = val.astype(int)
        self.int_min = val.min()
        return val - self.int_min

    def convert_to_int(self, val: float) -> int:
        return int(val) - self.int_min

    def reverse_convert_to_int(self, val: int) -> int:
        return val + self.int_min

    @property
    def code_range(self) -> List[Tuple[int, int]]:
        """
        get the vocab id range for each of the encoded tokens
        @returns [(min, max), (min, max), ...]
        """
        # first largest digits
        outputs = []
        c = 0
        for i in reversed(range(self.code_len)):
            ids = self.digits_id_to_item[i].keys()
            if c == 0:
                if self.hasnan:
                    outputs.append((min(ids), max(ids) + 2))  # the first token contains the N/A
                else:
                    outputs.append((min(ids), max(ids) + 1))  # non N/A
            else:
                outputs.append((min(ids), max(ids) + 1))
            c += 1
        return outputs

    def encode(self, item: str) -> List[int]:
        if self.hasnan and item == self.NA_token:
            return self.NA_token_id
        elif not self.hasnan and item == self.NA_token:
            raise ValueError(f"colum {self.name} cannot handle nan, please set hasnan=True")
        val = float(item)
        val_int = self.convert_to_int(val)
        digits = []
        for i in range(self.code_len):
            digit = (val_int // self.base ** i) % self.base
            digits.append(str(digit))
        if (val_int // self.base ** self.code_len) != 0:
            raise ValueError("not right length")
        codes = []
        for i in reversed(range(self.code_len)):
            digit_str = digits[i]
            if digit_str in self.digits_item_to_id[i]:
                codes.append(self.digits_item_to_id[i][digit_str])
            else:
                # find the nearest encode id
                allowed_digits = np.array([int(d) for d in self.digits_item_to_id[i].keys()])
                near_id = np.argmin(np.abs(allowed_digits - int(digit_str)))
                digit_str = str(allowed_digits[near_id])
                codes.append(self.digits_item_to_id[i][digit_str])
                # logging.warning('out of domain num is encounterd, use nearest code')
        return codes

    def decode(self, ids: List[int]) -> str:
        if self.hasnan and ids[0] == self.NA_token_id[0]:
            return self.NA_token
        v = 0
        for i in reversed(range(self.code_len)):
            digit = int(self.digits_id_to_item[i][ids[self.code_len - i - 1]])
            v += digit * self.base ** i
        v = self.reverse_convert_to_int(v)
        return str(v)


class FloatCode(IntCode):
    def __init__(
        self,
        col_name: str,
        code_len: int,
        start_id: int,
        fillall: bool = True,
        base: int = 100,
        hasnan: bool = True,
        transform: str = 'quantile',
    ):
        super().__init__(col_name, code_len, start_id, fillall, base, hasnan)
        self.mval = None
        self.extra_digits = None
        if transform == 'yeo-johnson':
            self.scaler = PowerTransformer(standardize=True)
        elif transform == 'quantile':
            self.scaler = QuantileTransformer(output_distribution='uniform', n_quantiles=100)
        elif transform == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError('Supported data transformations are "yeo-johnson", "quantile", and "robust"')

    def convert_to_int(self, val: float) -> int:
        val = np.expand_dims(np.array(val), axis=0)
        values = self.scaler.transform(val[:, None])[:, 0] - self.mval
        values = (values * self.base ** self.extra_digits).astype(int)
        output = values[0]
        return output

    def array_convert_to_int(self, val: ndarray):
        values = self.scaler.fit_transform(val[:, None])[:, 0]
        self.mval = values.min()
        values = values - self.mval
        digits = int(math.log(values.max(), self.base)) + 1
        # extra digits used for 'float' part of the number
        extra_digits = self.code_len - digits
        if extra_digits < 0:
            raise ValueError("need large length to code the nummber")
        self.extra_digits = extra_digits
        values = (values * self.base ** self.extra_digits).astype(int)
        return values

    def reverse_convert_to_int(self, val: int) -> float:
        val = val / self.base ** self.extra_digits
        val = np.expand_dims(np.array(val), axis=0)
        v = self.scaler.inverse_transform(val[:, None] + self.mval)[0, 0]
        return v

    def decode(self, ids: List[int]) -> str:
        if self.hasnan and ids[0] == self.NA_token_id[0]:
            return self.NA_token
        v = 0
        for i in reversed(range(self.code_len)):
            digit = int(self.digits_id_to_item[i][ids[self.code_len - i - 1]])
            v += digit * self.base ** i
        v = self.reverse_convert_to_int(v)
        accuracy = max(int(abs(np.log10(0.1 / self.base ** self.extra_digits))), 1)
        return f"{v:.{accuracy}f}"


class CategoryCode(Code):
    def __init__(self, col_name: str, start_id: int):
        super().__init__(col_name, 1, start_id, True, False)
        self.item_to_id = None
        self.id_to_item = None

    def compute_code(self, data_series: ndarray):
        uniq_items = np.unique(data_series).tolist()
        id_to_item = {}
        item_to_id = {}
        for i in range(len(uniq_items)):
            item = str(uniq_items[i])
            item_to_id[item] = self.end_id
            id_to_item[self.end_id] = item
            self.end_id += 1
        self.id_to_item = id_to_item
        self.item_to_id = item_to_id

    def encode(self, item) -> List[int]:
        return [self.item_to_id[item]]

    def decode(self, ids: List[int]) -> str:
        return self.id_to_item[ids[0]]


column_map = {"int": IntCode, "float": FloatCode, "category": CategoryCode}


class ColumnCodes(object):
    def __init__(self):
        self.column_codes: Dict[str, Code] = {}
        self.columns = []
        self.sizes = []

    @property
    def vocab_size(self):
        return self.column_codes[self.columns[-1]].end_id

    def register(self, name: str, ccode: Code):
        self.columns.append(name)
        self.column_codes[name] = ccode
        self.sizes.append(ccode.code_len)

    def encode(self, col: str, item: str) -> List[int]:
        if col in self.column_codes:
            return self.column_codes[col].encode(item)
        else:
            raise ValueError(f"cannot encode {col} {item}")

    def decode(self, col: str, ids: List[int]) -> str:
        if col in self.column_codes:
            return self.column_codes[col].decode(ids)
        else:
            raise ValueError("cannot decode")

    def get_range(self, column_id: int) -> List[Tuple[int, int]]:
        return self.column_codes[self.columns[column_id]].code_range

    @classmethod
    def get_column_codes(cls, column_configs, example_arrays):
        column_codes = cls()
        beg = 0
        cc = None
        for config in column_configs:
            col_name = config['name']
            coder = column_map[config['code_type']]
            args = config.get('args', {})
            start_id = beg if cc is None else cc.end_id
            args['start_id'] = start_id
            args['col_name'] = col_name
            cc = coder(**args)
            cc.compute_code(example_arrays[col_name])
            column_codes.register(col_name, cc)
        return column_codes
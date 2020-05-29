import mock
import numpy as np
import pandas as pd
import pytest
from six import PY3

from dtale.column_replacements import ColumnReplacement

if PY3:
    from contextlib import ExitStack
else:
    from contextlib2 import ExitStack


def replacements_data():
    return pd.DataFrame.from_dict({
        'a': ['a', 'UNknown', 'b'],
        'b': ['', ' ', ' - '],
        'c': [1, '', 3],
        'd': [1.1, np.nan, 3],
        'e': ['a', np.nan, 'b']
    })


def verify_builder(builder, checker):
    checker(builder.build_replacements())
    assert builder.build_code()


@pytest.mark.unit
def test_spaces(unittest):
    df = replacements_data()
    data_id, replacement_type = '1', 'spaces'
    with ExitStack() as stack:
        stack.enter_context(mock.patch('dtale.global_state.DATA', {data_id: df}))

        builder = ColumnReplacement(data_id, 'b', replacement_type, {})
        verify_builder(builder, lambda col: unittest.assertEqual(list(col.values), ['', np.nan, ' - ']))

        builder = ColumnReplacement(data_id, 'b', replacement_type, {'value': 'blah'})
        verify_builder(builder, lambda col: unittest.assertEqual(list(col.values), ['', 'blah', ' - ']))


@pytest.mark.unit
def test_string(unittest):
    df = replacements_data()
    data_id, replacement_type = '1', 'strings'
    with ExitStack() as stack:
        stack.enter_context(mock.patch('dtale.global_state.DATA', {data_id: df}))

        cfg = {'value': 'unknown', 'ignoreCase': True, 'isChar': False}
        builder = ColumnReplacement(data_id, 'a', replacement_type, cfg)
        verify_builder(builder, lambda col: unittest.assertEqual(list(col.values), ['a', np.nan, 'b']))

        cfg = {'value': 'unknown', 'ignoreCase': False, 'isChar': False}
        builder = ColumnReplacement(data_id, 'a', replacement_type, cfg)
        verify_builder(builder, lambda col: unittest.assertEqual(list(col.values), ['a', 'UNknown', 'b']))

        cfg = {'value': 'unknown', 'ignoreCase': True, 'isChar': False, 'replace': 'missing'}
        builder = ColumnReplacement(data_id, 'a', replacement_type, cfg)
        verify_builder(builder, lambda col: unittest.assertEqual(list(col.values), ['a', 'missing', 'b']))

        cfg = {'value': '-', 'ignoreCase': True, 'isChar': True}
        builder = ColumnReplacement(data_id, 'b', replacement_type, cfg)
        verify_builder(builder, lambda col: unittest.assertEqual(list(col.values), ['', ' ', np.nan]))

        cfg = {'value': '-', 'ignoreCase': True, 'isChar': True, 'replace': 'missing'}
        builder = ColumnReplacement(data_id, 'b', replacement_type, cfg)
        verify_builder(builder, lambda col: unittest.assertEqual(list(col.values), ['', ' ', 'missing']))


@pytest.mark.unit
def test_value(unittest):
    df = replacements_data()
    data_id, replacement_type = '1', 'value'
    with ExitStack() as stack:
        stack.enter_context(mock.patch('dtale.global_state.DATA', {data_id: df}))

        cfg = {'value': [dict(value='nan', replace='for test')]}
        builder = ColumnReplacement(data_id, 'e', replacement_type, cfg)
        verify_builder(builder, lambda col: unittest.assertEqual(list(col.values), ['a', 'for test', 'b']))

        cfg = {'value': [dict(value='nan', replace='for test'), dict(value='a', replace='d')]}
        builder = ColumnReplacement(data_id, 'e', replacement_type, cfg)
        verify_builder(builder, lambda col: unittest.assertEqual(list(col.values), ['d', 'for test', 'b']))

        cfg = {'value': [dict(value='nan', agg='median')]}
        builder = ColumnReplacement(data_id, 'd', replacement_type, cfg)
        verify_builder(builder, lambda col: unittest.assertEqual(list(col.values), [1.1, 2.05, 3]))


@pytest.mark.unit
def test_imputers(unittest):
    df = replacements_data()
    data_id, replacement_type = '1', 'imputer'
    with ExitStack() as stack:
        stack.enter_context(mock.patch('dtale.global_state.DATA', {data_id: df}))

        cfg = {'type': 'iterative'}
        builder = ColumnReplacement(data_id, 'd', replacement_type, cfg)
        verify_builder(builder, lambda col: unittest.assertEqual(list(col.values), [1.1, 2.05, 3]))

        cfg = {'type': 'knn', 'n_neighbors': 3}
        builder = ColumnReplacement(data_id, 'd', replacement_type, cfg)
        verify_builder(builder, lambda col: unittest.assertEqual(list(col.values), [1.1, 2.05, 3]))

        cfg = {'type': 'simple'}
        builder = ColumnReplacement(data_id, 'd', replacement_type, cfg)
        verify_builder(builder, lambda col: unittest.assertEqual(list(col.values), [1.1, 2.05, 3]))

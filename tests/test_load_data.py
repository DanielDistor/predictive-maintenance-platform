import os
import pandas as pd
from src.load_data import load_train, load_test, load_rul, COLUMNS

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def test_load_train_returns_dataframe():
    assert isinstance(load_train(FIXTURES), pd.DataFrame)


def test_load_train_has_26_columns():
    assert load_train(FIXTURES).shape[1] == 26


def test_load_train_column_names():
    assert list(load_train(FIXTURES).columns) == COLUMNS


def test_load_train_row_count():
    assert len(load_train(FIXTURES)) == 5


def test_load_test_returns_dataframe():
    assert isinstance(load_test(FIXTURES), pd.DataFrame)


def test_load_test_has_26_columns():
    assert load_test(FIXTURES).shape[1] == 26


def test_load_test_column_names():
    assert list(load_test(FIXTURES).columns) == COLUMNS


def test_load_test_row_count():
    assert len(load_test(FIXTURES)) == 2


def test_load_rul_returns_dataframe():
    assert isinstance(load_rul(FIXTURES), pd.DataFrame)


def test_load_rul_columns():
    assert list(load_rul(FIXTURES).columns) == ["engine_id", "rul"]


def test_load_rul_row_count():
    assert len(load_rul(FIXTURES)) == 2


def test_load_rul_engine_ids_are_one_indexed():
    assert load_rul(FIXTURES)["engine_id"].tolist() == [1, 2]


def test_load_rul_values():
    assert load_rul(FIXTURES)["rul"].tolist() == [112, 98]

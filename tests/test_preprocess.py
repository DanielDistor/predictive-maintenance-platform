import pandas as pd
from src.preprocess import add_rul


def _sample_df():
    return pd.DataFrame({
        "engine_id": [1, 1, 1, 2, 2],
        "cycle":     [1, 2, 3, 1, 2],
        "sensor_1":  [0.1, 0.2, 0.3, 0.4, 0.5],
    })


def test_add_rul_adds_rul_column():
    assert "RUL" in add_rul(_sample_df()).columns


def test_add_rul_correct_values():
    df = add_rul(_sample_df())
    engine1 = df[df["engine_id"] == 1].sort_values("cycle")["RUL"].tolist()
    engine2 = df[df["engine_id"] == 2].sort_values("cycle")["RUL"].tolist()
    assert engine1 == [2, 1, 0]
    assert engine2 == [1, 0]


def test_add_rul_no_max_cycle_column():
    assert "max_cycle" not in add_rul(_sample_df()).columns


def test_add_rul_preserves_row_count():
    original = _sample_df()
    assert len(add_rul(original)) == len(original)

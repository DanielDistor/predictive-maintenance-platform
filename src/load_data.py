import os
import pandas as pd

COLUMNS = [
    "engine_id", "cycle",
    "op_1", "op_2", "op_3",
    "sensor_1", "sensor_2", "sensor_3", "sensor_4", "sensor_5",
    "sensor_6", "sensor_7", "sensor_8", "sensor_9", "sensor_10",
    "sensor_11", "sensor_12", "sensor_13", "sensor_14", "sensor_15",
    "sensor_16", "sensor_17", "sensor_18", "sensor_19", "sensor_20",
    "sensor_21",
]


def _read_txt(path):
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    df.dropna(axis=1, how="all", inplace=True)
    return df


def load_train(data_dir="data"):
    df = _read_txt(os.path.join(data_dir, "train_FD001.txt"))
    df.columns = COLUMNS
    return df


def load_test(data_dir="data"):
    df = _read_txt(os.path.join(data_dir, "test_FD001.txt"))
    df.columns = COLUMNS
    return df


def load_rul(data_dir="data"):
    df = _read_txt(os.path.join(data_dir, "RUL_FD001.txt"))
    df.columns = ["rul"]
    df.insert(0, "engine_id", range(1, len(df) + 1))
    return df

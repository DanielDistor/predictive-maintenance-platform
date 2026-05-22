# Predictive Maintenance Platform — Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Load the NASA C-MAPSS FD001 dataset and display a Streamlit dashboard showing engine metrics, sensor trends, and a data preview.

**Architecture:** Two focused `src/` modules handle data concerns (`load_data.py` reads raw files, `preprocess.py` computes RUL). `app.py` imports both and renders the Streamlit UI. Test fixtures are small copies of the real data format so tests run without the full dataset.

**Tech Stack:** Python 3.8+, Pandas, NumPy, Plotly, Streamlit, pytest

---

## File Map

| File | Status | Responsibility |
|---|---|---|
| `.gitignore` | Create | Exclude data/, venv/, caches |
| `requirements.txt` | Create | Project dependencies |
| `src/__init__.py` | Create | Makes src a Python package |
| `src/load_data.py` | Create | Read .txt files, assign column names |
| `src/preprocess.py` | Create | Add RUL column to train DataFrame |
| `src/features.py` | Create | Phase 2 placeholder |
| `src/model.py` | Create | Phase 2 placeholder |
| `src/predict.py` | Create | Phase 2 placeholder |
| `tests/__init__.py` | Create | Test package marker |
| `tests/fixtures/train_FD001.txt` | Create | 5-row sample for tests |
| `tests/fixtures/test_FD001.txt` | Create | 2-row sample for tests |
| `tests/fixtures/RUL_FD001.txt` | Create | 2-line sample for tests |
| `tests/test_load_data.py` | Create | Tests for load_data.py |
| `tests/test_preprocess.py` | Create | Tests for preprocess.py |
| `app.py` | Create | Streamlit dashboard entry point |
| `README.md` | Modify | Dataset download instructions + setup |
| `notebooks/.gitkeep` | Create | Keeps empty dir in git |

---

## Task 1: Project Scaffolding

**Files:**
- Create: `.gitignore`
- Create: `requirements.txt`
- Create: `src/__init__.py`
- Create: `src/features.py`
- Create: `src/model.py`
- Create: `src/predict.py`
- Create: `tests/__init__.py`
- Create: `notebooks/.gitkeep`

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p src tests/fixtures notebooks
```

- [ ] **Step 2: Create `.gitignore`**

```
data/
venv/
__pycache__/
*.pyc
.DS_Store
.superpowers/
CMaps/
.pytest_cache/
*.egg-info/
```

- [ ] **Step 3: Create `requirements.txt`**

```
streamlit
pandas
numpy
plotly
scikit-learn
pytest
```

- [ ] **Step 4: Create package markers and placeholders**

`src/__init__.py` — empty file

`src/features.py`:
```python
# Phase 2
```

`src/model.py`:
```python
# Phase 2
```

`src/predict.py`:
```python
# Phase 2
```

`tests/__init__.py` — empty file

`notebooks/.gitkeep` — empty file

- [ ] **Step 5: Create and activate virtual environment, install dependencies**

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Expected: all packages install without errors.

- [ ] **Step 6: Commit**

```bash
git add .gitignore requirements.txt src/ tests/__init__.py notebooks/
git commit -m "feat: scaffold project structure and install dependencies"
```

---

## Task 2: Test Fixtures

**Files:**
- Create: `tests/fixtures/train_FD001.txt`
- Create: `tests/fixtures/test_FD001.txt`
- Create: `tests/fixtures/RUL_FD001.txt`

These are tiny copies of the real format (space-delimited, no header, trailing space). Engine 1 has 2 cycles, engine 2 has 3 cycles — used to verify RUL math.

- [ ] **Step 1: Create `tests/fixtures/train_FD001.txt`**

```
1 1 -0.0007 -0.0004 100.0 518.67 641.82 1589.70 1400.60 14.62 21.61 554.36 2388.06 9046.19 1.30 47.47 521.66 2388.02 8138.62 8.4195 0.03 392 2388 100.00 39.06 23.4190 
1 2 0.0019 -0.0003 100.0 518.67 642.15 1591.82 1403.14 14.62 21.61 553.75 2388.04 9044.07 1.30 47.49 522.28 2388.07 8131.49 8.4318 0.03 392 2388 100.00 39.00 23.4236 
2 1 -0.0043 0.0003 100.0 518.67 642.35 1587.99 1404.20 14.62 21.61 554.26 2388.08 9052.94 1.30 47.27 522.42 2388.03 8133.23 8.4178 0.03 390 2388 100.00 38.95 23.3442 
2 2 0.0012 0.0001 100.0 518.67 641.96 1590.01 1401.70 14.62 21.61 554.60 2388.00 9049.51 1.30 47.35 521.86 2388.01 8140.38 8.4291 0.03 391 2388 100.00 39.04 23.4155 
2 3 -0.0025 0.0002 100.0 518.67 641.60 1588.43 1399.80 14.62 21.61 554.10 2388.05 9047.22 1.30 47.41 521.50 2388.04 8136.95 8.4150 0.03 392 2388 100.00 38.99 23.3987 
```

- [ ] **Step 2: Create `tests/fixtures/test_FD001.txt`**

```
1 192 -0.0007 -0.0004 100.0 518.67 641.82 1589.70 1400.60 14.62 21.61 554.36 2388.06 9046.19 1.30 47.47 521.66 2388.02 8138.62 8.4195 0.03 392 2388 100.00 39.06 23.4190 
2 287 0.0019 -0.0003 100.0 518.67 642.15 1591.82 1403.14 14.62 21.61 553.75 2388.04 9044.07 1.30 47.49 522.28 2388.07 8131.49 8.4318 0.03 392 2388 100.00 39.00 23.4236 
```

- [ ] **Step 3: Create `tests/fixtures/RUL_FD001.txt`**

```
112 
98 
```

- [ ] **Step 4: Commit**

```bash
git add tests/fixtures/
git commit -m "test: add fixture files for unit tests"
```

---

## Task 3: `src/load_data.py` (TDD)

**Files:**
- Create: `tests/test_load_data.py`
- Create: `src/load_data.py`

- [ ] **Step 1: Write failing tests in `tests/test_load_data.py`**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_load_data.py -v
```

Expected: `ImportError` or `ModuleNotFoundError` — `src/load_data.py` does not exist yet.

- [ ] **Step 3: Implement `src/load_data.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_load_data.py -v
```

Expected: 13 tests PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/load_data.py tests/test_load_data.py
git commit -m "feat: implement load_data with tests"
```

---

## Task 4: `src/preprocess.py` (TDD)

**Files:**
- Create: `tests/test_preprocess.py`
- Create: `src/preprocess.py`

- [ ] **Step 1: Write failing tests in `tests/test_preprocess.py`**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_preprocess.py -v
```

Expected: `ImportError` — `src/preprocess.py` does not exist yet.

- [ ] **Step 3: Implement `src/preprocess.py`**

```python
def add_rul(df):
    max_cycle = (
        df.groupby("engine_id")["cycle"]
        .max()
        .reset_index()
        .rename(columns={"cycle": "max_cycle"})
    )
    df = df.merge(max_cycle, on="engine_id")
    df["RUL"] = df["max_cycle"] - df["cycle"]
    df.drop(columns=["max_cycle"], inplace=True)
    return df
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_preprocess.py -v
```

Expected: 4 tests PASSED.

- [ ] **Step 5: Run full test suite to confirm nothing broke**

```bash
pytest -v
```

Expected: 17 tests PASSED.

- [ ] **Step 6: Commit**

```bash
git add src/preprocess.py tests/test_preprocess.py
git commit -m "feat: implement preprocess with RUL computation and tests"
```

---

## Task 5: Streamlit Dashboard (`app.py`)

**Files:**
- Create: `app.py`

- [ ] **Step 1: Create `app.py`**

```python
import streamlit as st
import plotly.graph_objects as go
from src.load_data import load_train
from src.preprocess import add_rul

st.set_page_config(page_title="Predictive Maintenance Platform", layout="wide")


@st.cache_data
def get_data():
    df = load_train()
    return add_rul(df)


df = get_data()

st.title("Predictive Maintenance Platform")
st.caption("NASA C-MAPSS FD001 — Turbofan Engine Degradation Dataset")

col1, col2, col3 = st.columns(3)
col1.metric("Total Engines", df["engine_id"].nunique())
col2.metric("Total Rows", f"{len(df):,}")
col3.metric("Average RUL", round(df["RUL"].mean(), 1))

st.divider()

engine_id = st.selectbox("Select Engine", sorted(df["engine_id"].unique()))

sensor_cols = [f"sensor_{i}" for i in range(1, 22)]
selected_sensors = st.multiselect(
    "Select Sensors",
    sensor_cols,
    default=["sensor_2", "sensor_3", "sensor_4"],
)

if selected_sensors:
    engine_df = df[df["engine_id"] == engine_id].sort_values("cycle")
    fig = go.Figure()
    for sensor in selected_sensors:
        fig.add_trace(go.Scatter(
            x=engine_df["cycle"],
            y=engine_df[sensor],
            mode="lines",
            name=sensor,
        ))
    fig.update_layout(
        title=f"Sensor Readings — Engine {engine_id}",
        xaxis_title="Cycle",
        yaxis_title="Sensor Value",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Select at least one sensor to display the chart.")

st.divider()

st.subheader("Dataset Preview")
st.caption("First 100 rows of training data with RUL")
st.dataframe(df.head(100), use_container_width=True)
```

- [ ] **Step 2: Run the dashboard**

```bash
streamlit run app.py
```

Expected: browser opens at `http://localhost:8501`

- [ ] **Step 3: Verify all 6 elements are present and working**

Checklist:
- [ ] Three metric cards show correct numbers (100 engines, 20631 rows, ~107.8 avg RUL)
- [ ] Engine selector dropdown lists engine IDs
- [ ] Sensor multiselect defaults to sensor_2, sensor_3, sensor_4
- [ ] Plotly chart updates when engine or sensors change
- [ ] Clearing all sensors shows the info message
- [ ] Data table shows first 100 rows with RUL column

- [ ] **Step 4: Stop the server (Ctrl+C) and commit**

```bash
git add app.py
git commit -m "feat: add streamlit dashboard with metrics, chart, and data preview"
```

---

## Task 6: README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Replace `README.md` content**

```markdown
# Predictive Maintenance Platform

Streamlit dashboard for predictive maintenance using NASA C-MAPSS sensor data
to monitor equipment health and forecast remaining useful life.

## Dataset Setup

The NASA C-MAPSS dataset is not included in this repo. Download it from Kaggle:

1. Go to https://www.kaggle.com/datasets/behrad3d/nasa-cmaps
2. Click **Download** (free Kaggle account required)
3. Unzip the archive
4. Copy these files into the `data/` folder:
   - `train_FD001.txt`
   - `test_FD001.txt`
   - `RUL_FD001.txt`

## Setup

**Requirements:** Python 3.8+

```bash
git clone <repo-url>
cd predictive-maintenance-platform

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Place the dataset files in `data/` as described above, then:

```bash
streamlit run app.py
```

The dashboard opens at http://localhost:8501.

## Running Tests

```bash
pytest -v
```

## Project Structure

```
src/
  load_data.py    reads .txt files and assigns column names
  preprocess.py   computes Remaining Useful Life (RUL) for training data
  features.py     Phase 2 placeholder
  model.py        Phase 2 placeholder
  predict.py      Phase 2 placeholder
app.py            Streamlit dashboard entry point
data/             dataset files (not in git — see Dataset Setup above)
tests/            unit tests
```
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add setup instructions and dataset download guide"
```

---

## Done

Run the full test suite one final time to confirm everything passes:

```bash
pytest -v
```

Expected: **17 tests PASSED**

Then launch the app:

```bash
streamlit run app.py
```

# Predictive Maintenance Platform — Phase 1 Design

**Date:** 2026-05-21
**Scope:** Data loading + basic Streamlit dashboard (Phase 1 only)
**Stack:** Python, Pandas, NumPy, Plotly, Streamlit
**Dataset:** NASA C-MAPSS FD001 (train, test, RUL)

---

## Goal

Load the NASA C-MAPSS turbofan engine degradation dataset and display a basic Streamlit dashboard showing engine metrics, sensor trends, and a data preview. No ML models in this phase.

---

## Project Structure

```
predictive-maintenance-platform/
  data/
    train_FD001.txt       ← not in git (see README for download)
    test_FD001.txt
    RUL_FD001.txt
    train_FD002.txt       ← available for future phases
    test_FD002.txt
    RUL_FD002.txt
    train_FD003.txt
    test_FD003.txt
    RUL_FD003.txt
    train_FD004.txt
    test_FD004.txt
    RUL_FD004.txt
  notebooks/              ← empty, reserved for exploration
  src/
    load_data.py          ← active: data loading functions
    preprocess.py         ← active: RUL computation
    features.py           ← placeholder (Phase 2)
    model.py              ← placeholder (Phase 2)
    predict.py            ← placeholder (Phase 2)
  app.py                  ← Streamlit entry point
  requirements.txt
  README.md
  .gitignore
```

---

## Data Format

Each `.txt` file is space-delimited with no header row and trailing whitespace that produces an empty last column (must be dropped on load).

**26 columns assigned at load time:**
```
engine_id, cycle, op_1, op_2, op_3,
sensor_1 ... sensor_21
```

**RUL file:** One integer per line — the true remaining useful life for the last observed cycle of each test engine.

---

## Module Design

### `src/load_data.py`

Three public functions, no side effects:

- `load_train(data_dir="data")` → `pd.DataFrame`
  Reads `train_FD001.txt`, assigns column names, drops trailing empty column.

- `load_test(data_dir="data")` → `pd.DataFrame`
  Reads `test_FD001.txt`, assigns column names, drops trailing empty column.

- `load_rul(data_dir="data")` → `pd.DataFrame`
  Reads `RUL_FD001.txt`, returns DataFrame with columns `engine_id` (1-indexed) and `rul`.

### `src/preprocess.py`

One public function:

- `add_rul(df)` → `pd.DataFrame`
  Takes the train DataFrame, groups by `engine_id`, computes `max_cycle` per engine, adds `RUL = max_cycle - cycle` column. Returns the modified DataFrame.

### `src/features.py`, `src/model.py`, `src/predict.py`

Empty placeholder files. Each contains a single comment: `# Phase 2`.

---

## Dashboard Design (single-column layout)

`app.py` loads and preprocesses data once via `@st.cache_data`, then renders top-to-bottom:

1. **Page title** — `st.title("Predictive Maintenance Platform")`
2. **Three metric cards** (`st.columns(3)` + `st.metric`):
   - Total Engines
   - Total Rows
   - Average RUL across all training rows (rounded to 1 decimal)
3. **Engine selector** — `st.selectbox("Select Engine", sorted engine IDs)`
4. **Sensor selector** — `st.multiselect("Select Sensors", sensor_1–sensor_21, default=[sensor_2, sensor_3, sensor_4])` so the chart stays readable
5. **Sensor line chart** — Plotly `go.Figure` with one line per selected sensor for the chosen engine, x-axis = cycle
6. **Data preview** — `st.dataframe(df.head(100))` with caption

---

## Error Handling

Only one boundary: the data files. If a file is missing, `load_train()` will raise a `FileNotFoundError` from pandas. No custom error handling needed — the traceback message is clear enough for a dev-only tool. The README download instructions prevent the issue.

---

## Environment Setup

- Python `venv` + `pip`
- `requirements.txt`: `streamlit`, `pandas`, `numpy`, `plotly`, `scikit-learn`
- `scikit-learn` included now so Phase 2 (ML) doesn't require an environment change

---

## .gitignore

```
data/
venv/
__pycache__/
*.pyc
.DS_Store
.superpowers/
```

---

## README Sections

1. Project description
2. Dataset download instructions (Kaggle link for NASA C-MAPSS)
3. Setup: clone → create venv → pip install → place data files → streamlit run app.py
4. Project structure overview

---

## Out of Scope (Phase 2+)

- RUL prediction (ML model)
- Feature engineering
- Health score / risk level indicators
- FD002/FD003/FD004 datasets
- Test dataset visualization

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

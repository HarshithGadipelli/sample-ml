
# College Rank Intelligence System

A laptop-friendly Flask web app for college quality ranking, category prediction, semester-wise survey aggregation, admin retraining, and improvement suggestions.

## Batches

### Batch 1 — Core logic
- `core.py`
- `train_model.py`

### Batch 2 — Website
- `app.py`
- `templates/`
- `static/`

### Batch 3 — Sample data generator
- `generate_sample_data.py`
- `data/`

### Batch 4 — Setup
- `requirements.txt`
- `README.md`

## Run on your laptop

1. Install Python 3.10+.
2. Open terminal in this folder.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Generate sample datasets:

```bash
python generate_sample_data.py
```

5. Train the model:

```bash
python train_model.py
```

6. Start the website:

```bash
python app.py
```

7. Open:

```text
http://127.0.0.1:5000
```

## Admin

Open:

```text
http://127.0.0.1:5000/admin
```

Upload:
- `data/static_colleges.csv`
- `data/monthly_surveys.csv`

Then retrain.

## Real data replacement

Replace only the CSV files inside `data/` with your real college data and verified six-month survey data. Keep the same column names, then retrain.
"# ML-CCC" 

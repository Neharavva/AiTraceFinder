# TraceFinder – Forensic Scanner Identification

## Aim
To identify the scanner device used to scan a document by analyzing scanner-specific noise and frequency patterns.

## Features
- Image preprocessing
- Feature extraction (Laplacian + FFT)
- Scanner source prediction
- Streamlit UI

## How to Run
```bash
python train_model.py
python evaluate.py
streamlit run app.py

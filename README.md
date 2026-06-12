# IR Project 2026

Information Retrieval System using BEIR Quora Dataset

## Implemented Features

- Dataset Loading
- Text Preprocessing
- Inverted Index
- TF-IDF Retrieval
- BM25 Retrieval

## Environment

Python 3.11

## Installation

```bash
conda create -n i_r_system python=3.11
conda activate i_r_system
pip install -r requirements.txt
```

## Run

```bash
python load_dataset.py
python process_dataset.py
python build_index.py
```
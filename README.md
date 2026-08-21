# Smart Academic Text Processing Using Lightweight Transformer Models

An NLP pipeline that simplifies and summarizes NCERT Grade 8 Science textbook content using lightweight pretrained transformer models, built as an MCA final-year project.

## Overview

The pipeline takes textbook chapter text and produces:
- A **simplified** version (easier reading level) using T5
- An **abstractive summary** using BART

Both outputs are evaluated against the original text using standard NLP evaluation metrics (ROUGE, BLEU, METEOR, BERTScore, SARI, Flesch-Kincaid readability).

## Architecture

| Stage | Model | Task |
|---|---|---|
| Simplification | `t5-base` (T5ForConditionalGeneration) | Text simplification |
| Summarization | `facebook/bart-large-cnn` (BartForConditionalGeneration) | Abstractive summarization |

Pipeline stages: dataset loading → preprocessing/cleaning → simplification → summarization → evaluation.

## Dataset

NCERT Grade 8 Science textbook ("Curiosity" edition). **Note:** the source textbook PDF is not included in this repo due to copyright — see Setup below for how to source it yourself.

## Results

| Metric | Score |
|---|---|
| ROUGE-1 | 0.395 |
| ROUGE-2 | 0.087 |
| ROUGE-L | 0.235 |
| BLEU | 0.148 |
| METEOR | 0.305 |
| BERTScore (F1) | 0.227 |
| SARI | 38.97 |
| Flesch-Kincaid grade | 7.43 → 7.26 (−0.17) |
| Compression ratio | 89.0% |

## Tech Stack

- Python 3.12
- `transformers` >= 4.38.0
- `torch` >= 2.1.0
- `streamlit` >= 1.32.0 (demo UI)
- `sentencepiece` >= 0.2.0

## Project Structure

```
├── app.py                  # Streamlit demo UI
├── pipeline_runner.py       # End-to-end pipeline execution
├── modules/
│   ├── dataset_loader.py
│   ├── preprocessing.py
│   ├── simplification.py    # T5-based simplification
│   └── summarization.py     # BART-based summarization
├── requirements.txt
```

## Setup & Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## About

Final-year MCA project, Andhra University, Visakhapatnam.

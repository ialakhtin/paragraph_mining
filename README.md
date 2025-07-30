# Parallel Sentence Mining

This repository contains a pipeline for mining high-quality parallel sentence pairs from large, unaligned bilingual corpora using multilingual embeddings, approximate nearest neighbor search, and lexical-level filtering.

## Features
* Sentence embedding using [LaBSE](https://huggingface.co/sentence-transformers/LaBSE)

* FAISS-based nearest neighbor search

* Margin scoring for similarity

* Token-level filtering with lemmatization and bilingual dictionary overlap

* Evaluation tools with precision, recall, and F₀.₅ score

* Baseline comparison with LASER (CCMatrix-style)

## 📦 Installation

```bash
git clone https://github.com/your-username/parallel-sentence-mining.git
cd parallel-sentence-mining

# (Recommended) create a virtual environment
# Use python3.10 or lower
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```
You will also need to install [LASER](https://github.com/facebookresearch/LASER) if you want to test this model

## Data
All datasets are available in [WMT19](https://www.statmt.org/wmt19/translation-task.html)
1. Evaluation Datasets
    * Yandex corpus (must be loaded manually to the `data/1mcorpus`)

    * News Commentary

2. Fine-Tuning Data
    * CommonCrawl

3. Optional Baseline
    * ParaCrawl

## Preprocessing Pipeline
Each dataset undergoes the following steps:

1. Lowercasing and normalization (punctuation, quotation marks, etc.)

2. Filtering out lines with invalid characters

3. Language verification

4. Matching numeric tokens across pairs

5. Deduplication

6. Removal of outlier lengths (top 1%)

To start preprocessing, run `prepare_dataset.ipynb`

## Dictionary Construction
A bilingual dictionary is used for token-level filtering:

* Top 160k Russian words lemmatized via pymorphy3

* Top 100k English words lemmatized via nltk

* Lemmas translated bidirectionally via [Argos Translate](https://github.com/argosopentech/argos-translate)

To construct dictionary you need to load russian and english vocabularies to the `data/validator` as `russian_words.txt` and `english_words.txt` respectively.

I loaded english [here](https://github.com/david47k/top-english-wordlists) and russian [here](https://github.com/danakt/russian-words)

After that, run

```bash
python3 prepare_validator.py
```

## Launch
To run model on your own datasets:

1. Construct dictionary (see prev. part)
2. Collect english sentences and russian sentences
3. (Optional) use `Normalizer` and `StrFilter` in `prepare_dataset.ipynb` to clean your data
4. In `pair_mining.ipynb` run last cell (for a custom dataset)

## Evaluation
To evaluate the model:

1. A mixed test set is created by removing pairs from 2/3 of sentences.

2. The model predicts matching pairs between two shuffled sets.

3. Evaluation is done against the original gold set using:

    * Precision

    * Recall

    * $F_{0.5}$ score

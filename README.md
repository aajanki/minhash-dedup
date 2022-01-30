# Detecting near-duplicated text documents

This is an experiment on detecting near-duplicated text documents
using the MinHash LSH algorithm.

## Setup

1. Prepare a Python environment:
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Download ItemInfo_train.csv.zip and ItemPairs_train.csv.zip from the [Avite duplicate ads](https://www.kaggle.com/c/avito-duplicate-ads-detection/data) Kaggle competition into the data/duplicate-ads subdirectory.

## Run the experiments

```
python -m dedup.analyzedata
python -m dedup.plot_profile
python -m dedup.plot_histograms
```

The output will be saved in a subdirectory called results.

## References

Jure Leskovec, Anand Rajaraman, Jeff Ullman: [Mining of Massive Datasets](http://www.mmds.org/), Chapter 3

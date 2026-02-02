# Data Directory

This directory contains the French corpus data used for training Mini-GPT.

## Directory Structure

```
data/
├── raw/                    # Raw corpus files
│   └── french_corpus.txt   # French Wikipedia or other corpus
├── processed/              # Processed data
│   ├── corpus_statistics.json
│   └── tokenized_corpus.pkl
└── vocab/                  # Vocabulary files
    └── bpe_vocab.json
```

## Data Sources

### Primary: Wikipedia-fr
- **Source**: French Wikipedia dump
- **Size**: 10,000 articles (limited for 4-hour workshop)
- **Format**: Plain text, one article per line
- **Download**: Automatically downloaded via `datasets` library

### Alternative: Sample Corpus
- If Wikipedia download fails, a sample French corpus is created automatically
- Contains common French sentences for testing

## Usage

The corpus is automatically loaded in the notebook:
```python
from training.config import CorpusConfig
from training.data_loader import load_corpus

config = CorpusConfig(
    source="data/raw/french_corpus.txt",
    dataset_name="wikipedia-fr",
    max_lines=10000
)

texts = load_corpus(config)
```

## Statistics

After loading, corpus statistics are saved to:
- `data/processed/corpus_statistics.json`

This includes:
- Number of texts
- Character and word counts
- Vocabulary size
- Frequency distributions

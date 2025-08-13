# Running the C4 AAL Classifier

## Prerequisites

1. Download TwitterAAE model files from https://github.com/slanglab/twitteraae
2. Place model files in `./model/` directory:
   - `model_vocab.txt`
   - `model_count_table.txt`

## Installation

```bash
pip install datasets torch numpy numba tqdm tdigest
```

## Basic Usage

```bash
python code/label_c4.py --output-file results.jsonl --tdigest-file distributions.json
```

## Command Line Options

- `--output-file`: JSONL file for classified records (required)
- `--tdigest-file`: JSON file for score distributions (optional)
- `--model-dir`: Directory with model files (default: `./model`)
- `--max-records`: Limit number of records to process
- `--save-interval`: Save state every N records (default: 1000)
- `--savestate-file`: Resume state file (default: `savestate.json`)

## Example with Options

```bash
python code/label_c4.py \
  --output-file c4_classified.jsonl \
  --tdigest-file c4_distributions.json \
  --max-records 100000 \
  --save-interval 5000
```

## Output Files

- **JSONL output**: Contains classified text records with AAL scores
- **TDigest JSON**: Contains score distributions for each class (AA, H, A, W)
- **State file**: Allows resuming interrupted runs

## Generating Plots

### Vocabulary Overlap Plot
```bash
python code/vocabulary_overlap_plot.py --input-file code/results.jsonl --max-docs 5000
```

### Distribution Plots from TDigest
```bash
python code/plot_distributions.py --tdigest-file code/distributions.json
```

### Lyrics-C4 Vocabulary Overlap
```bash
python code/lyrics_c4_overlap.py --lyrics-file /path/to/updated_rappers.csv --max-lyrics 1000
```
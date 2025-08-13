#!/usr/bin/env python3
"""
Vocabulary Overlap Analysis: Generate overlap plot from C4 classification results.

Processes the JSONL output from label_c4.py to create vocabulary overlap visualization
similar to the lyrics analysis code.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import Counter
from pathlib import Path
import scienceplots

# Set up plotting style
plt.style.use('science')

def load_classified_data(jsonl_path):
    """Load classified data from JSONL file."""
    records = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            records.append(json.loads(line))
    return pd.DataFrame(records)

def create_word_counter(text):
    """
    Create word counter from text, matching the lyrics analysis approach.
    
    Extracts only alphabetic characters and spaces, converts to lowercase,
    and splits into individual words before creating a Counter object.
    """
    clean_text = "".join([e for e in text if (e.isalpha() or e.isspace())]).lower()
    return Counter(clean_text.split())

def compute_vocabulary_overlaps(df, sample_size=10000):
    """
    Compute vocabulary overlaps between high-AAL and all documents.
    
    Creates word counters for all documents, filters high-AAL documents
    (threshold > 0.3, matching lyrics analysis), then computes maximum
    vocabulary overlap using set intersection between each document
    and all high-AAL documents.
    """
    print("Creating word counters for all documents...")
    df['word_counter'] = df['text'].map(create_word_counter)
    
    high_aal = df[df['score_AA'] > 0.3].sample(min(sample_size, len(df[df['score_AA'] > 0.3])))
    print(f"Selected {len(high_aal)} high-AAL documents")
    
    high_aal_counters = high_aal['word_counter'].tolist()
    
    print("Computing vocabulary overlaps...")
    overlaps = []
    
    for idx, row in df.iterrows():
        doc_counter = row['word_counter']
        max_overlap = max([len(doc_counter & high_counter) for high_counter in high_aal_counters])
        overlaps.append(max_overlap)
        
        if (idx + 1) % 1000 == 0:
            print(f"Processed {idx + 1} documents")
    
    return overlaps

def create_overlap_plot(df, overlaps, output_path='vocabulary_overlap.png'):
    """
    Create the vocabulary overlap plot.
    
    Creates a 2D histogram showing the relationship between AAL scores
    and vocabulary overlap counts. Uses custom bin edges with logarithmic
    scaling similar to the original lyrics analysis approach. Removes
    zero-overlap entries for cleaner visualization.
    """
    df_plot = df.copy()
    df_plot['overlap'] = overlaps
    
    df_plot = df_plot[df_plot['overlap'] > 0]
    
    plt.figure(figsize=(7, 5))
    
    x_edges = np.logspace(np.log10(df_plot['score_AA'].min()), 
                         np.log10(df_plot['score_AA'].max()), num=100)
    
    offset = 10
    y_edges = (list(np.linspace(1, offset)) + 
              list(np.logspace(np.log10(df_plot['overlap'].min() + offset), 
                              np.log10(df_plot['overlap'].max()), num=100)))
    
    plt.hist2d(df_plot['score_AA'], df_plot['overlap'] + 0.00001, 
              bins=[x_edges, y_edges], norm=mcolors.LogNorm(), cmap='plasma')
    
    plt.xscale('linear')
    plt.yscale('log')
    plt.ylabel("Total Unique Words in Intersection")
    plt.title('Vocabulary Overlap with Closest High-AAL Document')
    plt.xlabel("AAL Posterior Label")
    plt.colorbar(label='log10(N)')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    print(f"Plot saved to {output_path}")

def main():
    """
    Main analysis pipeline.
    
    Loads classified C4 data, computes vocabulary overlaps between all
    documents and high-AAL reference documents, then creates visualization
    showing the relationship between AAL scores and vocabulary overlap counts.
    Provides progress tracking and summary statistics.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-file', default='results.jsonl',
                       help='JSONL file with classification results')
    parser.add_argument('--output-plot', default='vocabulary_overlap.png',
                       help='Output plot filename')
    parser.add_argument('--sample-size', type=int, default=10000,
                       help='Sample size for high-AAL documents')
    parser.add_argument('--max-docs', type=int, default=None,
                       help='Maximum documents to process (for testing)')
    
    args = parser.parse_args()
    
    print("Loading classification results...")
    df = load_classified_data(args.input_file)
    
    if args.max_docs:
        df = df.head(args.max_docs)
        print(f"Limited to first {args.max_docs} documents")
    
    print(f"Loaded {len(df)} classified documents")
    print(f"AAL score range: {df['score_AA'].min():.3f} - {df['score_AA'].max():.3f}")
    
    overlaps = compute_vocabulary_overlaps(df, args.sample_size)
    
    print(f"Overlap range: {min(overlaps)} - {max(overlaps)} words")
    print(f"Mean overlap: {np.mean(overlaps):.1f} words")
    
    create_overlap_plot(df, overlaps, args.output_plot)

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Lyrics-C4 Vocabulary Overlap: Compute overlap between lyrics CSV and high-AAL C4 documents.

Replicates the lyrics analysis approach but compares lyrics from CSV against 
high-AAL documents from C4 classification results.
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

def load_lyrics_data(csv_path):
    """Load lyrics data from CSV file."""
    df = pd.read_csv(csv_path)
    
    # Group by song similar to original lyrics analysis
    by_song = df.groupby("Unnamed: 0").agg({'lyric': lambda x: "\n ".join(x)}).reset_index()
    return by_song

def create_word_counter(text):
    """Create word counter from text, matching the lyrics analysis approach."""
    # Extract only alphabetic characters and spaces, convert to lowercase, split
    clean_text = "".join([e for e in text if (e.isalpha() or e.isspace())]).lower()
    return Counter(clean_text.split())

def compute_lyrics_c4_overlaps(lyrics_df, c4_df, max_c4_docs=None):
    """Compute vocabulary overlaps: for each C4 document, find closest lyrics."""
    print("Creating word counters for lyrics...")
    lyrics_df['word_counter'] = lyrics_df['lyric'].map(create_word_counter)
    lyrics_counters = lyrics_df['word_counter'].tolist()
    
    print("Processing C4 documents...")
    c4_sample = c4_df.head(max_c4_docs) if max_c4_docs else c4_df
    print(f"Processing {len(c4_sample)} C4 documents")
    
    print("Creating word counters for C4 documents...")
    c4_sample = c4_sample.copy()
    c4_sample['word_counter'] = c4_sample['text'].map(create_word_counter)
    
    print("Computing closest lyrics overlap for each C4 document...")
    overlaps = []
    
    for idx, row in c4_sample.iterrows():
        c4_counter = row['word_counter']
        # Find maximum overlap with any lyrics document (closest song)
        max_overlap = max([len(c4_counter & lyric_counter) for lyric_counter in lyrics_counters])
        overlaps.append(max_overlap)
        
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1} C4 documents")
    
    return overlaps, c4_sample

def create_overlap_plot(lyrics_df, overlaps, c4_sample, output_path='lyrics_c4_overlap.png'):
    """Create the vocabulary overlap plot matching the original analysis."""
    # Add overlap data to C4 dataframe
    c4_plot = c4_sample.copy()
    c4_plot['closest'] = overlaps
    
    # Remove rows with zero overlap for cleaner visualization
    matched = c4_plot[['score_AA', 'closest']].dropna()
    matched = matched[matched['closest'] > 0]
    
    plt.figure(figsize=(7, 5))
    
    # Create custom bin edges similar to original lyrics analysis
    x_edges = np.logspace(np.log10(matched['score_AA'].min()), 
                         np.log10(matched['score_AA'].max()), num=100)
    
    offset = 10
    y_edges = (list(np.linspace(1, offset)) + 
              list(np.logspace(np.log10(matched['closest'].min() + offset), 
                              np.log10(matched['closest'].max()), num=100)))
    
    plt.hist2d(matched['score_AA'], matched['closest'] + 0.00001, 
              bins=[x_edges, y_edges], norm=mcolors.LogNorm(), cmap='plasma')
    
    plt.xscale('linear')
    plt.yscale('log')
    plt.ylabel("Total Unique Words in Intersection")
    plt.title('Vocabulary Overlap with Closest Lyrics Document')
    plt.xlabel("AAL Posterior Label")
    plt.colorbar(label='log10(N)')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")

def main():
    """Main analysis pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--c4-file', default='code/results.jsonl',
                       help='JSONL file with C4 classification results')
    parser.add_argument('--lyrics-file', required=True,
                       help='CSV file with lyrics data')
    parser.add_argument('--output-plot', default='lyrics_c4_overlap.png',
                       help='Output plot filename')
    parser.add_argument('--max-c4-docs', type=int, default=5000,
                       help='Maximum C4 documents to process (for testing)')
    parser.add_argument('--max-lyrics', type=int, default=None,
                       help='Maximum lyrics to process (for testing)')
    
    args = parser.parse_args()
    
    print("Loading C4 classification results...")
    c4_df = load_classified_data(args.c4_file)
    print(f"Loaded {len(c4_df)} C4 classified documents")
    
    print("Loading lyrics data...")
    lyrics_df = load_lyrics_data(args.lyrics_file)
    print(f"Loaded {len(lyrics_df)} songs")
    
    if args.max_lyrics:
        lyrics_df = lyrics_df.head(args.max_lyrics)
        print(f"Limited to first {args.max_lyrics} songs")
    
    overlaps, c4_sample = compute_lyrics_c4_overlaps(lyrics_df, c4_df, args.max_c4_docs)
    
    print(f"Overlap range: {min(overlaps)} - {max(overlaps)} words")
    print(f"Mean overlap: {np.mean(overlaps):.1f} words")
    print(f"C4 docs with >0 overlap: {sum(1 for x in overlaps if x > 0)}/{len(overlaps)}")
    print(f"C4 docs with >100 overlap: {sum(1 for x in overlaps if x > 100)}/{len(overlaps)}")
    
    create_overlap_plot(lyrics_df, overlaps, c4_sample, args.output_plot)

if __name__ == "__main__":
    main()
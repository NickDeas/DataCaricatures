#!/usr/bin/env python3
"""
TDigest Distribution Plotter: Generate distribution plots from TDigest JSON files.

Based on the plotting logic from the corpuslens plots.py file.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from tdigest import TDigest
import scienceplots

# Set up plotting style
plt.style.use('science')

def load_tdigest_from_json(json_path):
    """Load TDigest objects from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    tdigests = []
    labels = ['AAL', 'H', 'A', 'W']  # AA, Hispanic, Asian, White
    key_mapping = {'AAL': 'AA', 'H': 'H', 'A': 'A', 'W': 'W'}
    
    for label in labels:
        key = f'tdigest_{key_mapping[label]}'
        if key in data:
            # Reconstruct TDigest from dictionary
            tdigest = TDigest()
            tdigest.update_from_dict(data[key])
            tdigests.append(tdigest)
        else:
            print(f"Warning: {key} not found in data")
            tdigests.append(None)
    
    return tdigests, labels

def plot_distributions(tdigests, labels, output_file='c4_distributions.png', title_suffix='C4'):
    """Create distribution plot similar to plots.py."""
    plt.figure(figsize=(12, 8))
    
    for label, tdigest in zip(labels, tdigests):
        if tdigest is None:
            continue
            
        centroids = tdigest.centroids_to_list()
        if not centroids:
            print(f"Warning: No centroids for {label}")
            continue
            
        # Extract x (means) and y (cumulative counts normalized)
        x = [c['m'] for c in centroids]
        y = np.array([c['c'] for c in centroids]) / tdigest.n
        
        # Apply smoothing if we have enough points
        if len(y) >= 10:
            y_smooth = savgol_filter(y, window_length=min(10, len(y)), polyorder=min(5, len(y)-1))
        else:
            y_smooth = y
        
        plt.plot(x, y_smooth, label=label)
    
    plt.legend(labels, loc='upper right')
    plt.xlabel(f'TwitterAAE Label Posterior {title_suffix}')
    plt.ylabel('Log Probability')
    plt.yscale('log')
    plt.axvline(x=0.3, color='black', linestyle='--', alpha=0.7)
    plt.title(f'TwitterAAE Label Density - {title_suffix}')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Distribution plot saved to {output_file}")

def main():
    """Main plotting pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tdigest-file', required=True,
                       help='JSON file containing TDigest data')
    parser.add_argument('--output-file', default='c4_distributions.png',
                       help='Output plot filename')
    parser.add_argument('--title', default='C4',
                       help='Title suffix for the plot')
    
    args = parser.parse_args()
    
    print(f"Loading TDigest data from {args.tdigest_file}...")
    tdigests, labels = load_tdigest_from_json(args.tdigest_file)
    
    print(f"Found {len([t for t in tdigests if t is not None])} valid TDigests")
    for label, tdigest in zip(labels, tdigests):
        if tdigest is not None:
            print(f"  {label}: {tdigest.n} samples, {len(tdigest.centroids_to_list())} centroids")
    
    plot_distributions(tdigests, labels, args.output_file, args.title)

if __name__ == "__main__":
    main()
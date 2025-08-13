#!/usr/bin/env python3
"""
Dataset Scanner: Clean, self-contained script for classifying C4 dataset text by
domain.

Processes the Allen AI C4 dataset, classifies text using a the TwitterAAE model,
and saves results according to the detection threshold.

"""

# %%
import argparse
import json
import random
import sys
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import torch
from datasets import load_dataset
from numba import jit
from tqdm import tqdm
from tdigest import TDigest

# %%
DEFAULT_MODEL_DIR = "./model"
DEFAULT_SAVE_INTERVAL = 1000
DEFAULT_DATASET_SEED = 0
DEFAULT_SKIP_POINT = 0
DEFAULT_SAMPLE_THRESHOLD = 0.7

# %%
class AALClassifier:
    """Self-contained AAL classifier with CVB0 inference.

    Adapted from  https://github.com/slanglab/twitteraae 

    To download the model files, clone their repo and move those files under
    your selected model directory
    """
    
    def __init__(self, model_dir: str = DEFAULT_MODEL_DIR):
        self.model_dir = Path(model_dir)
        self.vocab_file = self.model_dir / "model_vocab.txt"
        self.model_file = self.model_dir / "model_count_table.txt"
        
        # Model state
        self.K = 0
        self.wordprobs = None
        self.w2num = None
        self.vocab = None
        self.N_wk = None
        self.N_w = None
        self.N_k = None
        self._load_model()
    
    def _load_model(self):
        """Load model files. Idempotent."""
        if self.wordprobs is not None:
            return
        
        if not self.model_file.exists() or not self.vocab_file.exists():
            raise FileNotFoundError(f"Model files not found in {self.model_dir}")
        
        self.N_wk = np.loadtxt(self.model_file)
        self.N_w = self.N_wk.sum(1)
        self.N_k = self.N_wk.sum(0)
        self.K = len(self.N_k)
        self.wordprobs = (self.N_wk + 1) / self.N_k
        
        self.vocab = [line.split("\t")[-1].strip() for line in open(self.vocab_file)]
        self.w2num = {w: i for i, w in enumerate(self.vocab)}
        
        assert len(self.vocab) == self.N_wk.shape[0]
    
    @staticmethod
    @jit(nopython=True)
    def _infer_cvb0_optimized(token_indices, wordprobs, alpha, numpasses, K):
        """Optimized CVB0 inference using Numba.
        
        Read more about Collapsed Variational Bayes models in https://arxiv.org/abs/1409.4757
        """
        doclen = len(token_indices)
        Qs = np.zeros((doclen, K), dtype=np.float64)
        
        # Initialize with likelihoods
        for i in range(doclen):
            w_idx = token_indices[i]
            Qs[i, :] = wordprobs[w_idx, :]
            normalizer = np.sum(Qs[i, :])
            if normalizer > 0:
                Qs[i, :] /= normalizer
        
        lik = Qs.copy()
        Q_k = np.sum(Qs, axis=0)
        
        for _ in range(1, numpasses):
            for i in range(doclen):
                Q_k -= Qs[i, :]
                Qs[i, :] = lik[i, :] * (Q_k + alpha)
                normalizer = np.sum(Qs[i, :])
                if normalizer > 0:
                    Qs[i, :] /= normalizer
                Q_k += Qs[i, :]
        
        normalizer = np.sum(Q_k)
        if normalizer > 0:
            Q_k /= normalizer
        return Q_k
    
    def predict_raw(self, tokens, alpha=1, numpasses=5):
        """Predict AAL distribution for tokens without filtering."""
        if not tokens:
            return None
            
        invocab_tokens = [w.lower() for w in tokens if w.lower() in self.w2num]
        
        if len(invocab_tokens) == 0:
            return None
        
        # convert to indices for optimized inference
        token_indices = np.array([self.w2num[w] for w in invocab_tokens], dtype=np.int_)
        
        return self._infer_cvb0_optimized(
            token_indices, self.wordprobs, alpha=alpha, numpasses=numpasses, K=self.K
        )
    
    def predict(self, tokens, alpha=1, numpasses=5, vocab_thresh_min=1, vocab_thresh_ratio=0.2):
        """Predict AAL distribution for tokens."""
        if not tokens:
            return None
            
        invocab_tokens = [w.lower() for w in tokens if w.lower() in self.w2num]
        
        # Check vocabulary coverage thresholds
        if len(invocab_tokens) < vocab_thresh_min:
            return None
        if len(invocab_tokens) / len(tokens) < vocab_thresh_ratio:
            return None
        
        # convert to indices for optimized inference
        token_indices = np.array([self.w2num[w] for w in invocab_tokens], dtype=np.int_)
        
        return self._infer_cvb0_optimized(
            token_indices, self.wordprobs, alpha=alpha, numpasses=numpasses, K=self.K
        )

# %%
class DatasetScanner:
    """Main scanner class for processing C4 dataset."""
    
    def __init__(self, classifier, save_interval=DEFAULT_SAVE_INTERVAL, 
                 sample_threshold=DEFAULT_SAMPLE_THRESHOLD, output_file=None,
                 tdigest_file=None):
        self.classifier = classifier
        self.classes = ('AA', 'H', 'A', 'W')  
        self.save_interval = save_interval
        self.sample_threshold = sample_threshold
        self.output_file = output_file
        self.tdigest_file = tdigest_file
        
        # State tracking for resume capability
        self.processed_count = 0
        self.total_classified = 0
        
        # TDigest for storing full distributions
        self.tdigests = [TDigest() for _ in range(len(self.classes))]
        
        # Output file handle
        self.output_handle = None
        if self.output_file:
            self.output_handle = open(self.output_file, 'a', encoding='utf-8')
    
    def load_state(self, savestate_path):
        """Load existing state from files."""
        savestate_path = Path(savestate_path)
        
        if savestate_path.exists():
            state = json.loads(savestate_path.read_text())
            self.processed_count = state.get('processed_count', 0)
            self.total_classified = state.get('total_classified', 0)
    
    def save_state(self, savestate_path):
        """Save current state to file."""
        savestate_path = Path(savestate_path)
        
        state = {
            'processed_count': self.processed_count,
            'total_classified': self.total_classified
        }
        savestate_path.write_text(json.dumps(state))
        
        # Save TDigest distributions to JSON
        if self.tdigest_file:
            tdigest_data = {
                f'tdigest_{class_name}': tdigest.to_dict() 
                for class_name, tdigest in zip(self.classes, self.tdigests)
            }
            with open(self.tdigest_file, 'w') as f:
                json.dump(tdigest_data, f)
    
    def print_stats(self):
        """Print current processing statistics."""
        print(f"Processed: {self.processed_count:,} records")
        print(f"Successfully classified: {self.total_classified:,} records")
        if self.processed_count > 0:
            success_rate = (self.total_classified / self.processed_count) * 100
            print(f"Classification success rate: {success_rate:.1f}%")
    
    def process_record(self, record):
        """Process a single dataset record and write to JSONL output."""
        try:
            text = str(record.get('text', ''))
            url = record.get('url', '')
            
            if not text or not url:
                return False
            
            tokens = text.split()
            
            # Get raw prediction without filtering for TDigest
            raw_prediction = self.classifier.predict_raw(tokens)
            if raw_prediction is not None:
                for i, score in enumerate(raw_prediction):
                    self.tdigests[i].update(float(score))
            
            # Get filtered prediction for output
            prediction = self.classifier.predict(tokens)
            
            if prediction is None:
                return False
            
            try:
                parsed_url = urlparse(url)
                domain = parsed_url.netloc
            except Exception:
                return False
            
            if not domain:
                return False
            
            output_record = {
                'text': text,
                'url': url,
                'domain': domain,
                'processed_count': self.processed_count
            }
            
            for class_name, score in zip(self.classes, prediction):
                output_record[f'score_{class_name}'] = float(score)
            
            predicted_class_idx = np.argmax(prediction)
            output_record['predicted_class'] = self.classes[predicted_class_idx]
            output_record['max_score'] = float(prediction[predicted_class_idx])
            
            if self.output_handle:
                json.dump(output_record, self.output_handle, ensure_ascii=False)
                self.output_handle.write('\n')
                self.output_handle.flush()  # ensure data is written immediately
            
            return True
            
        except Exception as e:
            print(f"Error processing record: {e}", file=sys.stderr)
            return False
    
    def scan_dataset(self, dataset, savestate_path, max_records=None):
        """Main scanning loop with progress tracking."""
        self.load_state(savestate_path)
        
        # resume point logic
        if self.processed_count > 0:
            print(f"Resuming from record {self.processed_count:,}")
            dataset = dataset.skip(self.processed_count)
        
        # progress bar
        pbar = tqdm(dataset, desc="Scanning dataset", unit=" records")
        
        try:
            for record in pbar:
                # label AAL
                success = self.process_record(record)
                self.processed_count += 1
                if success:
                    self.total_classified += 1
                
                # progress bar update
                pbar.set_postfix({
                    'classified': self.total_classified,
                    'success_rate': f"{(self.total_classified/self.processed_count)*100:.1f}%" if self.processed_count > 0 else "0%"
                })
                
                if self.processed_count % self.save_interval == 0:
                    self.save_state(savestate_path)
                    if self.processed_count % (self.save_interval * 10) == 0:
                        print("\n" + "="*50)
                        self.print_stats()
                        print("="*50)
                
                if max_records and self.processed_count >= max_records:
                    print(f"\nReached max records limit: {max_records:,}")
                    break
                    
        except KeyboardInterrupt:
            print("\nInterrupted by user. Saving state...")
        finally:
            self.save_state(savestate_path)
            self.print_stats()
            if self.output_handle:
                self.output_handle.close()

# %%
def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(description=__doc__)
    
    parser.add_argument('--model-dir', default=DEFAULT_MODEL_DIR,
                        help=f'Directory containing TwitterAAE model files (default: {DEFAULT_MODEL_DIR})')
    
    parser.add_argument('--dataset-seed', type=int, default=DEFAULT_DATASET_SEED,
                        help=f'Random seed for dataset shuffling (default: {DEFAULT_DATASET_SEED})')
    parser.add_argument('--skip-point', type=int, default=DEFAULT_SKIP_POINT,
                        help=f'Number of records to skip (default: {DEFAULT_SKIP_POINT})')
    parser.add_argument('--max-records', type=int, default=None,
                        help='Maximum number of records to process (default: unlimited)')
    
    parser.add_argument('--save-interval', type=int, default=DEFAULT_SAVE_INTERVAL,
                        help=f'Save state every N records (default: {DEFAULT_SAVE_INTERVAL})')
    parser.add_argument('--sample-threshold', type=float, default=DEFAULT_SAMPLE_THRESHOLD,
                        help=f'Confidence threshold for saving samples (default: {DEFAULT_SAMPLE_THRESHOLD})')
    
    parser.add_argument('--output-file', required=True,
                        help='JSONL output file for classified records')
    parser.add_argument('--tdigest-file', 
                        help='JSON output file for TDigest distributions')
    parser.add_argument('--savestate-file', default='savestate.json',
                        help='File to save processing state (default: savestate.json)')
    
    parser.add_argument('--random-seed', type=int, default=0,
                        help='Random seed for reproducibility (default: 0)')
    
    args = parser.parse_args()
    
    torch.manual_seed(args.random_seed)
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    
    print("Initializing AAL classifier...")
    classifier = AALClassifier(args.model_dir)
    
    dataset = load_dataset("allenai/c4", "en.noblocklist", streaming=True)
    dataset = dataset.shuffle(seed=args.dataset_seed)
    train_dataset = dataset['train']
    
    if args.skip_point > 0:
        train_dataset = train_dataset.skip(args.skip_point)
    
    print("Starting dataset scan...")
    scanner = DatasetScanner(
        classifier=classifier,
        save_interval=args.save_interval,
        sample_threshold=args.sample_threshold,
        output_file=args.output_file,
        tdigest_file=args.tdigest_file
    )
    
    scanner.scan_dataset(
        dataset=train_dataset,
        savestate_path=args.savestate_file,
        max_records=args.max_records
    )
    
    print("Scan completed!")

# %%
if __name__ == "__main__":
    main()
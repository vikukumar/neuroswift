"""
examples/train_with_ssd.py
==========================
God-Level Demonstration: Training on multi-gigabyte datasets with low RAM.

This script uses MmapDataset to stream tokens directly from an SSD/HDD, 
allowing you to train a powerful model on a machine with as little as 4GB RAM.
"""
from pathlib import Path
import torch
from neuroswift import NeuroSwiftLM, NeuroSwiftConfig, WordTokenizer, AutoTrainer

def main():
    data_dir = Path("examples/data")
    output_dir = Path("artifacts/ssd_model")
    
    print("--- NeuroSwift SSD-Streaming Demo ---")
    
    # Initialize trainer with SSD-Streaming enabled explicitly
    trainer = AutoTrainer(
        data_dir=data_dir,
        output_dir=output_dir,
        use_ssd=True, # Force SSD-backed streaming (Mamba-2 style)
        epochs_per_cycle=1,
        save_every=100,
        ternary_mode=True # Enable BitNet-style addition-only training (1.58-bit)
    )
    
    # Run one cycle
    # For a real multi-GB dataset, just point data_dir to your 100GB folder
    loss = trainer.train_once()
    print(f"\nTraining cycle complete. Final loss: {loss:.4f}")
    print(f"Model saved to: {output_dir}")

if __name__ == "__main__":
    main()

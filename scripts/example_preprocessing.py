#!/usr/bin/env python3
"""
Example script showing how to use the Shiba dataset preprocessing.
"""

import os
import sys

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.preprocess_shiba_dataset import ShibaDataset


def example_usage():
    """Example of how to use the ShibaDataset class."""

    # Example parameters
    data_dir = "data"  # Path to your data directory
    time_indices = [0, 1, 2, 3, 4]  # Time indices to include
    use_IS = 0  # Use thermal positions (0) or IS positions (1)
    e_pot = "th"  # Use thermal potential energy
    r_c = 1.5  # Cutoff radius
    set_type = "train"  # or "test"
    N_train = 100  # Number of training samples

    print("Creating ShibaDataset...")

    # Create dataset instance
    dataset = ShibaDataset(
        root=data_dir,
        listTimesIdx=time_indices,
        IS=use_IS,
        e_pot=e_pot,
        r_c=r_c,
        set_type=set_type,
        N_train=N_train,
    )

    print(f"Dataset created with {len(dataset.raw_file_names)} raw files")
    print(f"Will process into {len(dataset.processed_file_names)} processed files")

    # Process the dataset
    print("Processing dataset...")
    dataset.process()

    print("Processing completed!")

    # Calculate statistics
    try:
        mean_dev_per_type, mean_dev_epot, mean_dev_deltaR = dataset.calc_stats()
        print("\nStatistics calculated successfully!")
        print(f"Potential energy mean/std: {mean_dev_epot}")
        print(f"Delta R cage mean/std: {mean_dev_deltaR}")
    except Exception as e:
        print(f"Could not calculate statistics: {e}")

    # Example of loading a single sample
    if len(dataset) > 0:
        print("\nLoading first sample...")
        sample = dataset.get(0)
        print(f"Sample loaded with {sample.x.shape[0]} nodes and {sample.edge_index.shape[1]} edges")
        print(f"Target shape: {sample.y.shape}")


if __name__ == "__main__":
    example_usage()

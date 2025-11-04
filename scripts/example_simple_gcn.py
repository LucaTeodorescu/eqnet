#!/usr/bin/env python3
"""
Example usage script for the simple GCN model.

This script demonstrates how to:
1. Load a sample data file
2. Create and inspect the model
3. Run a forward pass
4. Train the model briefly
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.simple_gcn import create_simple_gcn


def main():
    print("=== Simple GCN Model Example ===\n")

    # Load a sample data file
    data_path = "data/processed/isoconfig_N4096T0.44_1.pt"
    print(f"Loading sample data from: {data_path}")

    try:
        data = torch.load(data_path, weights_only=False)
        print("✓ Data loaded successfully")
        print(f"  - Number of nodes: {data.num_nodes}")
        print(f"  - Number of edges: {data.num_edges}")
        print(f"  - Node features shape: {data.x.shape}")
        print(f"  - Target shape: {data.y.shape}")
        print(f"  - Number of node types: {int(data.x.max().item() + 1)}")
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return

    # Create model
    num_node_types = int(data.x.max().item() + 1)
    model = create_simple_gcn(
        num_node_types=num_node_types,
        hidden_dim=32,  # Smaller for demo
        output_dim=10,
        dropout_p=0.1,
    )

    print("\n✓ Model created successfully")
    print(f"  - Number of parameters: {model.get_num_parameters():,}")
    print("  - Model architecture:")
    print(f"    {model}")

    # Test forward pass
    print("\n=== Testing Forward Pass ===")
    model.eval()

    with torch.no_grad():
        try:
            predictions = model(data)
            print("✓ Forward pass successful")
            print(f"  - Input shape: {data.x.shape}")
            print(f"  - Output shape: {predictions.shape}")
            print(f"  - Output range: [{predictions.min().item():.3f}, {predictions.max().item():.3f}]")
        except Exception as e:
            print(f"✗ Forward pass failed: {e}")
            return

    # Test training step
    print("\n=== Testing Training Step ===")
    model.train()

    try:
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Forward pass
        predictions = model(data)
        targets = data.y

        # Compute loss
        loss = criterion(predictions, targets)
        print("✓ Training step successful")
        print(f"  - Loss: {loss.item():.6f}")

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("✓ Backward pass successful")

    except Exception as e:
        print(f"✗ Training step failed: {e}")
        return

    print("\n=== Summary ===")
    print("✓ Model is ready for training!")
    print("✓ To train on full dataset, run:")
    print("  python scripts/train_simple_gcn.py --data_dir data/processed --epochs 10")
    print("✓ For quick testing, run:")
    print("  python scripts/train_simple_gcn.py --max_files 10 --epochs 5")


if __name__ == "__main__":
    main()

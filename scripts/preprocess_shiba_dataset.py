#!/usr/bin/env python3
"""
Preprocessing script for Shiba glass dataset.

This script processes raw glass simulation data into PyTorch Geometric format
for training GNNs to predict particle mobilities in glassy systems.
"""

import argparse
import glob
import os
from collections.abc import Sequence

import numpy as np
import scipy.spatial
import torch
from torch_geometric.data import Data, Dataset
from torch_scatter import scatter


def get_targets(initial_positions: np.ndarray, trajectory_target_positions: Sequence[np.ndarray]) -> np.ndarray:
    """Returns the averaged particle mobilities from the sampled trajectories.

    Args:
        initial_positions: the initial positions of the particles with shape
            [n_particles, 3].
        trajectory_target_positions: the absolute positions of the particles at the
            target time for all sampled trajectories, each with shape
            [n_particles, 3].

    Returns:
        targets: averaged particle mobilities with shape [n_particles]
    """
    targets = np.mean(
        [np.linalg.norm(t - initial_positions, axis=-1) for t in trajectory_target_positions],
        axis=0,
    )
    return targets.astype(np.float32)


def get_edge_targets(edge_index, edge_norms, target_positions):
    """Compute edge targets like in Shiba 2022.

    Args:
        edge_index: edge connectivity matrix
        edge_norms: initial edge distances
        target_positions: target positions for all trajectories

    Returns:
        edge_targets: change in edge distances
    """
    edge_targets = np.mean(
        [np.linalg.norm(t[edge_index[1]] - t[edge_index[0]], axis=-1) - edge_norms for t in target_positions],
        axis=0,
    )
    return edge_targets


class ShibaDataset(Dataset):
    """Dataset class for Shiba glass simulation data.

    This class processes raw glass simulation data and converts it into
    PyTorch Geometric format for GNN training.
    """

    def __init__(
        self,
        root,
        listTimesIdx,
        IS,
        e_pot,
        r_c,
        set_type,
        N_train=400,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        """
        Args:
            root: root directory containing raw and processed data
            listTimesIdx: list of time indices to regress to
            IS: use inherent structure positions (1) or thermal positions (0)
            e_pot: use potential energy ('IS', 'th', or None)
            r_c: cutoff radius for maximum edge distance
            set_type: 'train' or 'test'
            N_train: number of training samples
            transform: optional transform to apply to data
            pre_transform: optional pre-transform to apply during processing
            pre_filter: optional pre-filter to apply during processing
        """
        self.input_dir = root
        self.listTimesIdx = listTimesIdx
        self.set_type = set_type
        self.N_train = N_train
        self.IS = IS
        self.e_pot = e_pot
        self.r_c = r_c
        self.pot_th = 2.5

        # Train-test split according to Shiba paper
        self.train_idxs = np.arange(1, 401)
        self.test_idxs = np.arange(401, 501)

        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        """Get list of raw file names based on train/test split."""
        raw_files = glob.glob(self.input_dir + "/raw/*")
        if self.set_type == "train":
            train_files = []
            for name in raw_files:
                filename = name.split("/")[-1]
                try:
                    base_name = filename.replace(".npz", "")
                    file_number = int(base_name.split("_")[-1])
                    if file_number in self.train_idxs:
                        train_files.append(filename)
                except (ValueError, IndexError):
                    print(f"Warning: Could not parse filename {filename}")
                    continue

            if self.N_train > len(self.train_idxs):
                raise Exception("N_train larger than train set size")
            raw_files = train_files[: self.N_train]

        elif self.set_type == "test":
            test_files = []
            for name in raw_files:
                filename = name.split("/")[-1]
                try:
                    base_name = filename.replace(".npz", "")
                    file_number = int(base_name.split("_")[-1])
                    if file_number in self.test_idxs:
                        test_files.append(filename)
                except (ValueError, IndexError):
                    print(f"Warning: Could not parse filename {filename}")
                    continue
            raw_files = test_files

        else:
            raise Exception("Not allowed type")
        return raw_files

    @property
    def processed_file_names(self):
        """Get list of processed file names."""
        return [file[:-4] + ".pt" for file in self.raw_file_names]

    def download(self):
        """Download data if not present."""
        pass

    def calc_stats(self):
        """Calculate statistics for normalization."""
        ys = np.array([data.y.numpy() for data in self])  # node targets size [400, 4096, 10]
        types = self[0].x
        Ntargets = ys.shape[2]

        Ntypes = types.max() + 1
        mean_dev_per_type = np.zeros((Ntypes, Ntargets, 2))
        mean_dev_epot = np.zeros(2)
        mean_dev_deltaR = np.zeros(2)

        for part_type in range(Ntypes):
            selected_particles = np.where(types == part_type)[0]
            mean_dev_per_type[part_type, :, 0] = np.mean(ys[:, selected_particles, :], axis=(0, 1))
            mean_dev_per_type[part_type, :, 1] = np.std(ys[:, selected_particles, :], axis=(0, 1))

        epots = np.array([data.e_pot.numpy() for data in self])  # size [400, 4096]
        mean_dev_epot[0] = np.mean(epots)
        mean_dev_epot[1] = np.std(epots)

        deltaRs = np.array([data.delta_r_cage.numpy() for data in self])  # size [400, 4096]
        mean_dev_deltaR[0] = np.mean(deltaRs)
        mean_dev_deltaR[1] = np.std(deltaRs)

        return mean_dev_per_type, mean_dev_epot, mean_dev_deltaR

    def process(self):
        """Process raw data files into PyTorch Geometric format."""
        # Check if processed files already exist
        existing_files = sum(1 for path in self.processed_paths if os.path.exists(path))
        if existing_files == len(self.processed_paths):
            print(f"All {len(self.processed_paths)} processed files already exist. Skipping processing.")
            return
        elif existing_files > 0:
            print(f"Found {existing_files}/{len(self.processed_paths)} existing processed files. Processing remaining files...")

        print(f"Processing {len(self.raw_paths)} files...")

        successful_files = 0
        failed_files = 0

        for ind, path in enumerate(self.raw_paths):
            print(f"Processing file {ind+1}/{len(self.raw_paths)}: {os.path.basename(path)}")

            # Skip if file already processed
            if os.path.exists(self.processed_paths[ind]):
                print(f"  Skipping {os.path.basename(path)} - already processed")
                successful_files += 1
                continue

            try:
                # Load data with better error handling
                data = np.load(path)

                # Check if required keys exist
                required_keys = [
                    "types",
                    "positions",
                    "positions_IS",
                    "times",
                    "box",
                    "trajectory_target_positions",
                ]
                missing_keys = [key for key in required_keys if key not in data]
                if missing_keys:
                    print(f"Warning: Missing keys in {path}: {missing_keys}")
                    failed_files += 1
                    continue

                # Extract data
                types = data["types"].astype(np.int32)
                positions_th = data["positions"].astype(np.float32)
                positions_IS = data["positions_IS"].astype(np.float32)
                data["times"].astype(np.int32)
                box = data["box"]
                target_positions = data["trajectory_target_positions"]

                # Close the file explicitly
                data.close()

                # Number of targets
                n_idx = len(target_positions)

                # Compute targets for all timescales
                targets = []
                for timIdx in range(n_idx):
                    targets.append(get_targets(positions_th, target_positions[timIdx]))
                targets = np.array(targets)

                Lx = box[0]

                edge_index_list = []
                edge_attr_list = []
                e_pot_list = []
                pair_pot_list = []
                edge_targets_list = []

                # Process both thermal and IS positions
                for i, positions in enumerate((positions_th, positions_IS)):
                    # Generate the graph using k-d tree
                    distance_upper_bound = np.array([self.pot_th, self.r_c]).max()
                    tree = scipy.spatial.cKDTree(positions, boxsize=Lx + 1e-8)
                    _, col = tree.query(positions, k=4096, distance_upper_bound=distance_upper_bound + 1e-8)
                    col = col[:, 1:]  # receivers padded
                    row = np.array([np.ones(len(c)) * i for (i, c) in enumerate(col)], dtype=int)  # senders padded
                    cf = col.flatten()
                    rf = row.flatten()
                    mask = cf < tree.n  # remove padding

                    edge_index = np.array([rf[mask], cf[mask]], dtype=int)
                    edge_attr = positions[edge_index[1]] - positions[edge_index[0]]

                    # Enforce periodic boundary conditions
                    edge_attr[(edge_attr > Lx / 2.0)] -= Lx
                    edge_attr[(edge_attr < -Lx / 2.0)] += Lx
                    edge_norms = np.sum(edge_attr**2, axis=-1) ** (1 / 2)

                    # Compute potential energy
                    energy_scales = np.array([1, 1.5, 0.5])  # epsilon_AA, epsilon_AB, epsilon_BB
                    sigmas = np.array([1, 0.8, 0.88])  # sigma_AA, sigma_AB, sigma_BB
                    edge_types = types[edge_index[0]] + types[edge_index[1]]  # 0 = AA, 1 = AB, 2 = BB

                    edge_epsilon = energy_scales[edge_types]
                    edge_sigmas = sigmas[edge_types]
                    pairwise_potentials = 4 * edge_epsilon * ((edge_sigmas / edge_norms) ** 12 - (edge_sigmas / edge_norms) ** 6)

                    if distance_upper_bound > self.pot_th:
                        pairwise_potentials[edge_norms > self.pot_th] = 0.0

                e_pot_tensor = scatter(torch.tensor(pairwise_potentials), torch.tensor(edge_index[0]), dim=0)
                e_pot = e_pot_tensor.numpy()

                if distance_upper_bound > self.r_c:
                    # Restrict edges to cutoff radius
                    edge_index = edge_index[:, edge_norms <= self.r_c]
                    edge_attr = edge_attr[edge_norms <= self.r_c]
                    pairwise_potentials = pairwise_potentials[edge_norms <= self.r_c]

                    # Compute edge targets
                    initial_rel_positions = positions_th[edge_index[1]] - positions_th[edge_index[0]]
                    initial_distances = np.sum((initial_rel_positions) ** 2, axis=-1) ** (1 / 2)
                    edge_targets = []
                    for timIdx in range(n_idx):
                        edge_targets.append(get_edge_targets(edge_index, initial_distances, target_positions[timIdx]))
                    edge_targets = np.array(edge_targets)

                    edge_index_list.append(edge_index)
                    edge_attr_list.append(edge_attr)
                    e_pot_list.append(e_pot)
                    pair_pot_list.append(pairwise_potentials)
                    edge_targets_list.append(edge_targets)

                # Create PyTorch Geometric Data object
                data_out = Data(
                    x=torch.from_numpy(types.reshape((-1, 1))),
                    y=torch.from_numpy(targets).T,
                    edge_index_th=torch.from_numpy(edge_index_list[0]),
                    edge_attr_th=torch.from_numpy(edge_attr_list[0]),
                    pos_th=torch.from_numpy(positions_th),
                    e_pot_th=torch.from_numpy(e_pot_list[0].reshape((-1, 1))),
                    pair_pot_th=torch.from_numpy(pair_pot_list[0].reshape((-1, 1))),
                    edge_targets_th=torch.from_numpy(edge_targets_list[0]).T,
                    edge_index_IS=torch.from_numpy(edge_index_list[1]),
                    edge_attr_IS=torch.from_numpy(edge_attr_list[1]),
                    pos_IS=torch.from_numpy(positions_IS),
                    e_pot_IS=torch.from_numpy(e_pot_list[1].reshape((-1, 1))),
                    pair_pot_IS=torch.from_numpy(pair_pot_list[1].reshape((-1, 1))),
                    edge_targets_IS=torch.from_numpy(edge_targets_list[1]).T,
                    delta_r_cage=torch.from_numpy((np.sum((positions_th - positions_IS) ** 2, axis=-1) ** (1 / 2)).reshape((-1, 1))),
                )

                if self.pre_filter is not None:
                    data_out = self.pre_filter(data_out)

                if self.pre_transform is not None:
                    data_out = self.pre_transform(data_out)

                torch.save(data_out, self.processed_paths[ind])
                successful_files += 1

            except Exception as e:
                print(f"Error processing {path}: {e}")
                failed_files += 1
                continue

        print(f"Processing completed: {successful_files} successful, {failed_files} failed")

    def len(self):
        """Return number of processed files."""
        return len(self.processed_file_names)

    def get(self, idx):
        """Get a single data sample."""
        data_raw = torch.load(self.processed_paths[idx], weights_only=False)
        x = data_raw.x
        y = data_raw.y[:, self.listTimesIdx]

        if self.IS == 1:
            edge_index = data_raw.edge_index_IS
            edge_attr = data_raw.edge_attr_IS
            pos = data_raw.pos_IS
            edge_targets = data_raw.edge_targets_IS[:, self.listTimesIdx]
        else:
            edge_index = data_raw.edge_index_th
            edge_attr = data_raw.edge_attr_th
            pos = data_raw.pos_th
            edge_targets = data_raw.edge_targets_th[:, self.listTimesIdx]

        if self.e_pot == "IS":
            e_pot = data_raw.e_pot_IS
            pair_pot = data_raw.pair_pot_IS
        elif self.e_pot == "th":
            e_pot = data_raw.e_pot_th
            pair_pot = data_raw.pair_pot_th
        else:
            e_pot = torch.tensor(0)
            pair_pot = torch.tensor(0)

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            pos=pos,
            e_pot=e_pot,
            pair_pot=pair_pot,
            edge_targets=edge_targets,
            delta_r_cage=data_raw.delta_r_cage,
            filename=self.processed_paths[idx],
        )

        return data


def main():
    """Main function to run preprocessing."""
    parser = argparse.ArgumentParser(description="Preprocess Shiba glass dataset")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to data directory containing raw/ subdirectory",
    )
    parser.add_argument(
        "--set_type",
        type=str,
        choices=["train", "test"],
        required=True,
        help="Type of dataset to process",
    )
    parser.add_argument("--N_train", type=int, default=400, help="Number of training samples (default: 400)")
    parser.add_argument("--r_c", type=float, default=1.5, help="Cutoff radius for edges (default: 1.5)")
    parser.add_argument(
        "--IS",
        type=int,
        choices=[0, 1],
        default=0,
        help="Use inherent structure positions (1) or thermal positions (0)",
    )
    parser.add_argument(
        "--e_pot",
        type=str,
        choices=["IS", "th", "none"],
        default="th",
        help="Use potential energy from IS, thermal, or none",
    )
    parser.add_argument(
        "--time_indices",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
        help="Time indices to include in targets",
    )

    args = parser.parse_args()

    # Convert e_pot string to appropriate format
    e_pot = args.e_pot if args.e_pot != "none" else None

    print(f"Processing {args.set_type} dataset...")
    print(f"Data directory: {args.data_dir}")
    print(f"Number of samples: {args.N_train}")
    print(f"Cutoff radius: {args.r_c}")
    print(f"Use IS positions: {bool(args.IS)}")
    print(f"Potential energy: {e_pot}")
    print(f"Time indices: {args.time_indices}")

    # Create dataset and process
    dataset = ShibaDataset(
        root=args.data_dir,
        listTimesIdx=args.time_indices,
        IS=args.IS,
        e_pot=e_pot,
        r_c=args.r_c,
        set_type=args.set_type,
        N_train=args.N_train,
    )

    print("Preprocessing completed successfully!")

    # Calculate and print statistics
    try:
        mean_dev_per_type, mean_dev_epot, mean_dev_deltaR = dataset.calc_stats()
        print("\nDataset statistics:")
        print(f"Mean/std per particle type: {mean_dev_per_type.shape}")
        print(f"Potential energy mean/std: {mean_dev_epot}")
        print(f"Delta R cage mean/std: {mean_dev_deltaR}")
    except Exception as e:
        print(f"Could not calculate statistics: {e}")


if __name__ == "__main__":
    main()

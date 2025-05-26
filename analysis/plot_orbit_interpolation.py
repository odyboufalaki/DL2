import pathlib
import torch
from collections import defaultdict
from utils.utils import plot_interpolation_curves

def load_loss_matrices():
    matrix_dir = pathlib.Path("analysis/resources/interpolation/matrices")
    matrices = defaultdict(dict)
    
    # Track metadata from first file to verify consistency
    first_file = True
    expected_num_runs = None
    expected_perturbation = None
    
    for file_path in matrix_dir.glob("*.pt"):
        # Parse filename components
        # Format: loss_matrix-{naive/reconstruction}-{method}-{transformation}-numruns={num_runs}-perturbation={perturbation}.pt
        parts = file_path.stem.split("-")
        
        file_metadata = {
            'matrix_type': parts[1],      # naive or reconstruction
            'method': parts[2],           # linear_assignment, scalegmn or neural_graphs
            'transformation': parts[3],    # P, D, or PD
            'num_runs': int(parts[4].split("=")[1]),
            'perturbation': float(parts[5].split("=")[1])
        }

        # Verify consistency of num_runs and perturbation across all files
        if first_file:
            expected_num_runs = file_metadata['num_runs']
            expected_perturbation = file_metadata['perturbation']
            first_file = False
        else:
            assert file_metadata['num_runs'] == expected_num_runs, f"Inconsistent num_runs: {file_metadata['num_runs']} vs {expected_num_runs}"
            assert file_metadata['perturbation'] == expected_perturbation, f"Inconsistent perturbation: {file_metadata['perturbation']} vs {expected_perturbation}"

        # Load the matrix
        matrix = torch.load(file_path)
        
        # Store in nested dictionary
        # Create nested dictionaries if they don't exist
        if file_metadata['transformation'] not in matrices:
            matrices[file_metadata['transformation']] = {}
        if file_metadata['method'] not in matrices[file_metadata['transformation']]:
            matrices[file_metadata['transformation']][file_metadata['method']] = {}
        
        # Store the matrix
        matrices[file_metadata['transformation']][file_metadata['method']][file_metadata['matrix_type']] = matrix

    metadata = {
        'num_runs': expected_num_runs,
        'perturbation': expected_perturbation
    }
    return matrices, metadata


def main():
    matrices, metadata = load_loss_matrices()
    print("Plotting experiments for transformations:", matrices.keys())
    for transformation in matrices.keys():
        # Assert that the original matrices are the same for all methods
        # Calculate standard deviation between the three methods
        naive_matrices = [
            matrices[transformation]["linear_assignment"]["naive"],
            matrices[transformation]["scalegmn"]["naive"],
            matrices[transformation]["neural_graphs"]["naive"]
        ]
        std_dev = torch.stack(naive_matrices).std(dim=0)
        print(f"Standard deviation of Naive interpolations (should be 0) {transformation}: {std_dev.mean().item():.4f}")

        # Defines curves to plot
        curves = [
            (matrices[transformation]["linear_assignment"]["naive"], "Naive"),
            # (matrices[transformation]["linear_assignment"]["naive"], "Naive Linear Assignment"),
            # (matrices[transformation]["scalegmn"]["naive"], "Naive ScaleGMN"),
            # (matrices[transformation]["neural_graphs"]["naive"], "Naive Neural Graphs"),
            (matrices[transformation]["linear_assignment"]["reconstruction"], "Linear Assignment"),
            (matrices[transformation]["scalegmn"]["reconstruction"], "ScaleGMN Autoencoder"),
            (matrices[transformation]["neural_graphs"]["reconstruction"], "Neural Graphs Autoencoder"),
        ]
        
        # Plot the interpolation curves
        save_path = (
            f"analysis/resources/interpolation/"
            f"{transformation}_numruns={metadata['num_runs']}_perturbation={metadata['perturbation']}.png"
        )
        plot_interpolation_curves(curves, save_path=save_path)

if __name__ == "__main__":
    main()
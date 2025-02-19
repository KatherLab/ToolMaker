import numpy as np


def calculate_tissue_fraction(npy_path, tissue_type):
    """Calculate the fraction of specific tissue type"""
    prob_matrix = np.load(npy_path)
    multi_classes = np.ones((prob_matrix.shape[0], prob_matrix.shape[1])) * 1

    # Calculate maximum probability class for each pixel
    for i in range(prob_matrix.shape[0]):
        for j in range(prob_matrix.shape[1]):
            if np.sum(prob_matrix[i, j, :]) == 0:
                multi_classes[i][j] = 1
            else:
                multi_classes[i][j] = np.argmax(prob_matrix[i, j, :])

    # Calculate total area (excluding background class 1)
    total_area = np.sum(multi_classes != 1)

    # Calculate area of specific tissue type
    tissue_area = np.sum(multi_classes == tissue_type)

    return tissue_area / total_area if total_area > 0 else 0


def mus_fraction(npy_path):
    """Calculate muscle fraction"""
    return calculate_tissue_fraction(npy_path, 5)


def hypothesis_score(prob_map_path: str) -> float:
    return mus_fraction(prob_map_path)

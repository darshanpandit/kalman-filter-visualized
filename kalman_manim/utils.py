"""Math utilities for Kalman filter visualizations."""

import numpy as np
from numpy.linalg import eigh


def cov_to_ellipse_params(cov: np.ndarray, n_sigma: float = 2.0):
    """Convert a 2x2 covariance matrix to ellipse rendering parameters.

    Uses eigendecomposition: the eigenvectors give the orientation and the
    eigenvalues (scaled by n_sigma) give the semi-axis lengths.

    Parameters
    ----------
    cov : np.ndarray
        2x2 positive-semidefinite covariance matrix.
    n_sigma : float
        Number of standard deviations for the ellipse boundary.
        2.0 ≈ 95% confidence region for a 2D Gaussian.

    Returns
    -------
    dict with keys:
        width  : float  – full width (2 * semi-major or semi-minor along x-eigenvector)
        height : float  – full height
        angle  : float  – rotation angle in radians (counter-clockwise from +x axis)
    """
    eigenvalues, eigenvectors = eigh(cov)  # sorted ascending
    # eigh guarantees real, non-negative eigenvalues for PSD matrices
    eigenvalues = np.maximum(eigenvalues, 0)  # numerical safety

    # Semi-axis lengths = n_sigma * sqrt(eigenvalue)
    semi_axes = n_sigma * np.sqrt(eigenvalues)

    # Rotation angle from the eigenvector corresponding to the larger eigenvalue
    # eigh returns sorted ascending, so eigenvectors[:, 1] is the major axis
    angle = np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1])

    return {
        "width": 2 * semi_axes[1],   # major axis (larger eigenvalue)
        "height": 2 * semi_axes[0],  # minor axis (smaller eigenvalue)
        "angle": angle,
    }


def gaussian_product_1d(mu1: float, var1: float, mu2: float, var2: float):
    """Compute the product of two 1D Gaussians.

    Returns the mean and variance of the resulting Gaussian.
    This is the core of the Kalman update step in 1D.
    """
    var_new = (var1 * var2) / (var1 + var2)
    mu_new = (var2 * mu1 + var1 * mu2) / (var1 + var2)
    return mu_new, var_new


def gaussian_product_2d(mu1: np.ndarray, cov1: np.ndarray,
                        mu2: np.ndarray, cov2: np.ndarray):
    """Compute the product of two 2D Gaussians.

    Returns the mean and covariance of the resulting Gaussian.
    """
    cov1_inv = np.linalg.inv(cov1)
    cov2_inv = np.linalg.inv(cov2)
    cov_new = np.linalg.inv(cov1_inv + cov2_inv)
    mu_new = cov_new @ (cov1_inv @ mu1 + cov2_inv @ mu2)
    return mu_new, cov_new


def gaussian_1d_pdf(x: np.ndarray, mu: float, var: float) -> np.ndarray:
    """Evaluate 1D Gaussian PDF at points x."""
    return (1.0 / np.sqrt(2 * np.pi * var)) * np.exp(-0.5 * (x - mu) ** 2 / var)

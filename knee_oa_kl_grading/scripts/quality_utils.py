"""
quality_utils.py - Shared quality assessment utilities for MRKR knee X-rays.

Exports functions to assess rotation, collimation, sharpness, exposure,
and anatomical completeness of knee X-ray images. Used by both the EDA
script (01_eda_with_quality.py) and the quality filtering preprocessor
(02_filter_by_quality.py).
"""

import numpy as np

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False


# =============================================================================
# Quality assessment functions
# =============================================================================

def assess_positioning_rotation(img_array):
    """Checks for leg rotation by comparing widths at the joint line.

    Parameters
    ----------
    img_array : np.ndarray
        2D grayscale image array.

    Returns
    -------
    float
        Rotation score. Scores > 0.1 may indicate rotation.
    """
    h, w = img_array.shape
    joint_line = img_array[h // 2 : h // 2 + 20, :]
    left_width = np.sum(joint_line[:, : w // 2])
    right_width = np.sum(joint_line[:, w // 2 :])
    eps = 1e-8
    rotation_score = abs(left_width - right_width) / (left_width + right_width + eps)
    return rotation_score


def assess_collimation(img_array):
    """Measures the amount of dark background (air) around the knee.

    Parameters
    ----------
    img_array : np.ndarray
        2D grayscale image array.

    Returns
    -------
    float
        Collimation score. Scores > 0.6 may indicate poor collimation.
        Returns NaN if OpenCV is unavailable.
    """
    if not _CV2_AVAILABLE:
        return np.nan
    _, binary_img = cv2.threshold(img_array, 10, 255, cv2.THRESH_BINARY)
    background_pixels = np.sum(binary_img == 0)
    collimation_score = background_pixels / max(binary_img.size, 1)
    return collimation_score


def assess_sharpness_blur(img_array):
    """Calculates the Laplacian variance as a measure of image sharpness.

    Parameters
    ----------
    img_array : np.ndarray
        2D grayscale image array.

    Returns
    -------
    float
        Laplacian variance. Low variance (< 100) suggests blur.
        Returns NaN if OpenCV is unavailable.
    """
    if not _CV2_AVAILABLE:
        return np.nan
    return cv2.Laplacian(img_array, cv2.CV_64F).var()


def assess_exposure_artifacts(img_array):
    """Flags potentially over or under-exposed images.

    Parameters
    ----------
    img_array : np.ndarray
        2D grayscale image array.

    Returns
    -------
    tuple of (float, float)
        (brightness, noise). Brightness < 30 (underexposed) or > 220
        (overexposed) may be issues.
    """
    brightness = np.mean(img_array)
    noise = np.std(img_array)
    return brightness, noise


def assess_anatomical_completeness(img_array, edge_thickness=5, intensity_threshold=5):
    """Flags images that may be cut off at the edges.

    Parameters
    ----------
    img_array : np.ndarray
        2D grayscale image array.
    edge_thickness : int
        Number of pixels from each edge to sample.
    intensity_threshold : float
        Mean intensity below which an edge is considered truncated.

    Returns
    -------
    bool
        True if truncation suspected at any edge.
    """
    h, w = img_array.shape
    edge_intensities = [
        np.mean(img_array[:, :edge_thickness]),      # Left edge
        np.mean(img_array[:, -edge_thickness:]),     # Right edge
        np.mean(img_array[:edge_thickness, :]),      # Top edge
        np.mean(img_array[-edge_thickness:, :]),     # Bottom edge
    ]
    return any(intensity < intensity_threshold for intensity in edge_intensities)


def compute_quality_metrics(img_array,
                            blur_threshold=100,
                            rotation_threshold=0.1,
                            collimation_threshold=0.6,
                            min_brightness=30,
                            max_brightness=220,
                            edge_thickness=5,
                            edge_intensity_threshold=5):
    """Run all quality assessments on a grayscale image array.

    Parameters
    ----------
    img_array : np.ndarray
        2D grayscale image array.
    blur_threshold : float
        Laplacian variance below this is flagged as blurred.
    rotation_threshold : float
        Rotation score above this is flagged as possible rotation.
    collimation_threshold : float
        Collimation score above this is flagged as poor collimation.
    min_brightness : float
        Mean brightness below this is flagged as underexposed.
    max_brightness : float
        Mean brightness above this is flagged as overexposed.
    edge_thickness : int
        Pixels from edge to sample for truncation check.
    edge_intensity_threshold : float
        Edge mean intensity below this is flagged as truncated.

    Returns
    -------
    dict
        Dictionary of quality metric names to values, including boolean flags.
    """
    brightness, noise = assess_exposure_artifacts(img_array)
    metrics = {
        "brightness": brightness,
        "noise": noise,
        "sharpness_laplacian_var": assess_sharpness_blur(img_array),
        "rotation_score": assess_positioning_rotation(img_array),
        "collimation_score": assess_collimation(img_array),
        "anatomical_truncated": assess_anatomical_completeness(img_array,
                                                               edge_thickness,
                                                               edge_intensity_threshold),
        "underexposed": brightness < min_brightness,
        "overexposed": brightness > max_brightness,
    }
    # Blur and collimation flags only valid when OpenCV is available
    if _CV2_AVAILABLE:
        metrics["blurred"] = metrics["sharpness_laplacian_var"] < blur_threshold
        metrics["poor_collimation"] = metrics["collimation_score"] > collimation_threshold
    metrics["possible_rotation"] = metrics["rotation_score"] > rotation_threshold
    return metrics


def compute_overall_pass(metrics_dict, max_fail_flags=0):
    """Determine whether an image passes quality filtering.

    Parameters
    ----------
    metrics_dict : dict
        Output from compute_quality_metrics().
    max_fail_flags : int
        Maximum number of flagged issues to tolerate.
        0 means any flag = fail. Use -1 to require all checks to pass.

    Returns
    -------
    bool
        True if the image passes quality filtering.
    """
    # Collect boolean flag columns (exclude continuous metrics)
    flag_keys = [k for k in metrics_dict if k in (
        "underexposed", "overexposed", "blurred", "poor_collimation",
        "possible_rotation", "anatomical_truncated"
    ) and k in metrics_dict]
    n_failures = sum(1 for k in flag_keys if metrics_dict.get(k, False))
    if max_fail_flags < 0:
        return n_failures == 0
    return n_failures <= max_fail_flags


def cv2_available():
    """Check whether OpenCV is available for cv2-dependent metrics."""
    return _CV2_AVAILABLE


def cv2_version():
    """Return the OpenCV version string, or None if unavailable."""
    if _CV2_AVAILABLE:
        return cv2.__version__
    return None

import numpy as np

focal_length = 721.5377  # Focal length in pixels (from KITTI dataset)
baseline = 0.532722  # Baseline length in meters (from KITTI dataset)
def depth_to_disparity(depth, focal_length, baseline):
    """
    Converts depth values to disparity values.

    Parameters:
        depth (numpy.ndarray): The depth map (values in meters).
        focal_length (float): The focal length of the camera in pixels.
        baseline (float): The baseline length of the stereo camera setup in meters.

    Returns:
        numpy.ndarray: The computed disparity map (values in pixels).
    """
    # Avoid division by zero
    valid_mask = depth > 0
    disparity = np.zeros_like(depth)
    disparity[valid_mask] = (focal_length * baseline) / depth[valid_mask]
    return disparity

def compute_d1_error(gt_disp, pred_disp):
    """
    Computes the D1 error percentage and generates an error map.

    Parameters:
        gt_disp (numpy.ndarray): Ground truth disparity map.
        pred_disp (numpy.ndarray): Predicted disparity map.

    Returns:
        float: The D1 error percentage.
        numpy.ndarray: A binary error map, where 1 indicates pixels with errors and 0 otherwise.
    """

    # Valid pixel mask (ground truth disparity > 0)
    mask = gt_disp > 0

    # Calculate absolute error
    abs_diff = np.abs(gt_disp - pred_disp)

    # Error conditions
    error_mask = (abs_diff > 3) & (abs_diff > 0.05 * gt_disp)

    # Compute D1 error percentage
    error_pixels = np.sum(error_mask & mask)
    total_pixels = np.sum(mask)
    d1_error = (error_pixels / total_pixels) * 100

    # Generate error map
    error_map = np.zeros_like(gt_disp)
    error_map[error_mask & mask] = 1  # Mark error pixels

    return d1_error, error_map

def error_to_color(error_norm, error_mask, valid_mask):
    """
    Generates a color visualization of the error map.

    Parameters:
        error_norm (numpy.ndarray): Normalized error values (range [0, 1]).
        error_mask (numpy.ndarray): Binary mask indicating erroneous pixels.
        valid_mask (numpy.ndarray): Binary mask indicating valid pixels.

    Returns:
        numpy.ndarray: A color image representing the error map.
                       - Blue channel: Correct predictions.
                       - Red channel: Errors scaled by magnitude.
                       - Black: Occluded or invalid pixels.
    """
    # Initialize color image
    H, W = error_norm.shape
    color_image = np.zeros((H, W, 3), dtype=np.float32)

    # Correctly predicted pixels (blue channel)
    color_image[..., 2] = (~error_mask) & valid_mask  # 蓝色通道

    # Incorrectly predicted pixels (red tones based on error magnitude)
    color_image[..., 0] = error_norm * error_mask     # 红色通道

    # Occluded and invalid pixels (black)
    color_image[~valid_mask] = 0

    return color_image
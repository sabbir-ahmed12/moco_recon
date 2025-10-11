import os
import yaml
import logging
import numpy as np
import nibabel as nib

# Get the logger
logger = logging.getLogger(__name__)

# Load the configuration file
def load_config(config_path):
    """
    Load the global configuration file.

    Parameters:
    -----------
        config_path : str
            Path of the configuration file.
    
    Returns:
    --------
        config : dict
            Dictionary containing the global parameters.
    """
    logger.info(f"Loading the global config file from: {config_path}.")
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

    except FileNotFoundError:
        logger.error(f"Global configuration file not found in: {global_conf}")
        raise FileNotFoundError(f"Global configuration file not found: {global_conf}")

    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML: {e}")

    return config


def minmax_normalize(x, min_val=0.0, max_val=1.0):
    """
    Perform min-max normalization of an array, handling NaN values.
    
    Args:
        x: Input array
        min_val: Minimum value of output range
        max_val: Maximum value of output range
        
    Returns:
        Normalized array with NaN values preserved
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Create mask of non-NaN values
    mask = ~np.isnan(x)
    
    if not np.any(mask):
        # All values are NaN, return original array
        return x
    
    # Calculate min and max only on non-NaN values
    x_min = np.min(x[mask])
    x_max = np.max(x[mask])
    
    if x_max == x_min:
        # Handle case where all non-NaN values are identical
        normalized = np.full_like(x, (min_val + max_val) / 2)
        normalized[~mask] = np.nan  # Preserve NaN values
        return normalized
    
    # Normalize non-NaN values
    normalized = np.empty_like(x)
    normalized[mask] = (x[mask] - x_min) / (x_max - x_min)  # Scale to [0, 1]
    normalized[mask] = normalized[mask] * (max_val - min_val) + min_val  # Scale to [min_val, max_val]
    normalized[~mask] = np.nan  # Preserve NaN values
    
    return normalized
    

def save_nifti_volume(volume, filename=None, save_dir=None):
    """
    Save an input volume as NifTI.

    Args:
        volume (numpy.ndarray) : Input volume
        filename (str) : Name of the file to be saved.
        save_dir (str) : Path of the output directory
    """
    if filename is None:
        filename = "default_name.nii.gz"

    if save_dir is None:
        save_dir = os.getcwd()

    logger.info(f"Saving {filename} at {save_dir}.")
    normalized_output = minmax_normalize(volume, 0, 255)
    nifti_volume = nib.Nifti1Image(normalized_output, np.eye(4))
    output_path = os.path.join(save_dir, filename)
    nib.save(nifti_volume, output_path)


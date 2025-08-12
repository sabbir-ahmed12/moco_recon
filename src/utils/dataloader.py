import os
import logging
import numpy as np

# Get the logger
logger = logging.getLogger(__name__)

# Load npy files
def load_npy_files(processed_dir):
    logger.info(f"Loading the required preprocessed .npy files from {processed_dir} ...")
        
    try:
        ksp = np.load(os.path.join(processed_dir, "ksp.npy"))
        coord = np.load(os.path.join(processed_dir, "coord.npy"))
        dcf = np.load(os.path.join(processed_dir, "dcf.npy"))
        resp = np.load(os.path.join(processed_dir, "resp.npy"))
        tr = np.load(os.path.join(processed_dir, "tr.npy"))
        noise = np.load(os.path.join(processed_dir, "noise.npy"))

        # Logging the shape of the npy files
        logger.info(f"Shape of ksp: {ksp.shape}")
        logger.info(f"Shape of coord: {coord.shape}")
        logger.info(f"Shape of dcf: {dcf.shape}")
        logger.info(f"Shape of resp: {resp.shape}")
        logger.info(f"Shape of tr: {tr.shape}")
        logger.info(f"Shape of noise: {noise.shape}")

    except FileNotFoundError as e:
        logger.error(f"Error loading files {e}.")
        raise FileNotFoundError(e)

    return ksp, coord, dcf, resp, tr, noise
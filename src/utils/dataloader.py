import os
import logging
import numpy as np

# Get the logger
logger = logging.getLogger(__name__)

# Load npy files
def load_npy_files(file_path):
    logger.info(f"Loading the required preprocessed .npy files from {file_path} ...")
    try:
        ksp = os.path.join(file_path, "ksp.npy")
        coord = os.path.join(file_path, "coord.npy")
        dcf = os.path.join(file_path, "dcf.npy")
        resp = os.path.join(file_path, "resp.npy")
        tr = os.path.join(file_path, "tr.npy")
        noise = os.path.join(file_path, "noise.npy")

    except FileNotFoundError as e:
        logger.error(f"Error loading files {e}.")
        raise FileNotFoundError(e)
    
    return ksp, coord, dcf, resp, tr, noise
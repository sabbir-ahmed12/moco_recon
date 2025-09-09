import os 
import yaml
import time
import argparse
import logging
from datetime import datetime

# Set the logging directory
log_dir = os.path.join(os.path.dirname(os.getcwd()), "logs")
os.makedirs(log_dir, exist_ok=True)

# Configure the logger
log_filename = os.path.join(log_dir, datetime.now().strftime("%Y-%m-%d_%H-%M-%S.log"))
log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(filename=log_filename, level=logging.INFO, format=log_format, datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)


# Load the internal modules
from utils.dataloader import load_npy_files
from utils.misc import load_config
from utils.auto_fov import auto_fov

def main(config_path):
    start_time = time.time()

    # Load the global configuration parameters
    config = load_config(config_path)

    # Get the processed directory
    processed_dir = config["directories"]["processed"]

    # Convert the MRI_Raw.h5 file into npy files
    if config["preprocessing"]["convert_h5"]:
        # Loading the convert_ute function
        from utils.convert_h5_to_npy import convert_ute

        # Set the path to import raw and to save the npy files
        h5_path = os.path.join(config["directories"]["raw"], "MRI_Raw.h5")

        # Extract the required files and save as npy files
        convert_ute(h5_path, output_dir=processed_dir)

    # Load the npy files
    ksp, coord, dcf, resp, tr, noise = load_npy_files(processed_dir)

    stop_time = time.time()
    logger.info(f"Total time taken: {(stop_time - start_time)/3600:.2f} hours.")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reconstruct image using different motion compensated algorithms."
        )
    parser.add_argument("--config_path", type=str)

    args = parser.parse_args()
    main(args.config_path)

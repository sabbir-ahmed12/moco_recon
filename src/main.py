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


def main(config_path):
    logger.info(f"Loading the global config file from: {config_path}.")
    
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

    except FileNotFoundError:
        logger.error(f"Global configuration file not found in: {global_conf}")
        raise FileNotFoundError(f"Global configuration file not found: {global_conf}")

    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML: {e}")

    # Convert the MRI_Raw.h5 file into npy files
    if config["preprocessing"]["convert_h5"]:
        # Loading the convert_ute function
        from utils.convert_h5_to_npy import convert_ute
        from utils.dataloader import load_npy_files

        # Set the path to import raw and to save the npy files
        h5_path = os.path.join(config["directories"]["raw"], "MRI_Raw.h5")
        processed_dir = config["directories"]["processed"]

        # Extract the required files and save as npy files
        convert_ute(h5_path, processed_dir)

        # Load the npy files
        encodes = os.listdir(processed_dir)
        for encode in encodes:
            logger.info(f"Now loading encode: {encode} ...")
            encode_path = os.path.join(processed_dir, encode)
            # ksp, coord, dcf, resp, tr, noise = load_npy_files(encode_path)

    else:
        processed_dir = config["directories"]["processed"]

        # Load the npy files
        encodes = os.listdir(processed_dir)
        for encode in encodes:
            logger.info(f"Now loading encode: {encode} ...")
            encode_path = os.path.join(processed_dir, encode)
            # ksp, coord, dcf, resp, tr, noise = load_npy_files(encode_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reconstruct image using different motion compensated algorithms."
        )
    parser.add_argument("--config_path", type=str)

    args = parser.parse_args()
    main(args.config_path)

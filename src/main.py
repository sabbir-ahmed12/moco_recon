import os 
import yaml
import time
import argparse
import logging
from datetime import datetime

# Set the logging directory
log_dir = os.path.join(os.path.dirname(os.getcwd()), 'logs')
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

    # Run the moco-recons 
    if config['reconstructions']['no_gating']:
        from no_gating import no_gating

    if config['reconstructions']['hard_gating']:
        from hard_gating import hard_gating

    if config['reconstructions']['soft_gating']:
        from soft_gating import soft_gating

    if config['reconstructions']['xdgrasp']:
        from xdgrasp import xdgrasp
    
    if config['reconstructions']['imoco']:
        from imoco import imoco

    if config['reconstructions']['mocostorm']:
        from mocostorm import mocostorm



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reconstruct image using motion compensated algorithms."
        )
    parser.add_argument("--config_path", type=str)

    args = parser.parse_args()
    main(args.config_path)

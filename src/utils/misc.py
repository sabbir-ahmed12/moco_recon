import yaml
import logging

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




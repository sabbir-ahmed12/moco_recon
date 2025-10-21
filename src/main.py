import os 
import yaml
import time
import argparse
import logging
import sigpy as sp
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
from utils.misc import load_config, save_nifti_volume
from utils.auto_fov import auto_fov
from no_gating.no_gating import NoGating
from hard_gating.hard_gating import HardGating

def main(raw_path, config_path):
    start_time = time.time()

    # Load the global configuration parameters
    config = load_config(config_path)

    # Set the device
    device = -1
    if config["device"]["gpu"]:
        device = sp.Device(0)

    # Create the necessary directories
    recon_dir = os.path.join(raw_path, 'recons')
    os.makedirs(recon_dir, exist_ok=True)

    processed_dir = os.path.join(recon_dir, 'processed')
    os.makedirs(processed_dir, exist_ok=True)


    # Convert the MRI_Raw.h5 file into npy files
    if config["preprocessing"]["convert_h5"]:
        # Loading the convert_ute function
        from utils.convert_h5_to_npy import convert_ute

        # Set the path to import raw and to save the npy files
        h5_path = os.path.join(raw_path, "MRI_Raw.h5")

        # Extract the required files and save as npy files
        convert_ute(h5_path, output_dir=processed_dir)

    # Check if there are multiple directories (in-case of multiple encodes)
    encode_dirs = os.listdir(processed_dir)

    # Load the npy files
    for encode_dir in encode_dirs:
        processed_file_dir = os.path.join(processed_dir, encode_dir)
        ksp, coord, dcf, resp, tr, noise = load_npy_files(processed_file_dir)

        # Creating directory to save output for each encode
        out_dir = os.path.join(processed_file_dir, 'output')
        os.makedirs(out_dir, exist_ok=True)
    
        # Run No_Gating Reconstruction
        if config['reconstructions']['no_gating']:
            # Create a directory to save the files
            save_dir = os.path.join(out_dir, "no_gating")
            os.makedirs(save_dir, exist_ok=True)
            
            no_gating = NoGating(img_shape=config['output']['img_shape'], device=device)
            output_vol = no_gating.run(ksp, coord, dcf)

            save_nifti_volume(output_vol, filename="no_gating.nii.gz", save_dir=save_dir)
        
        # Run Hard_Gating Reconstruction
        if config['reconstructions']['hard_gating']:
            # Create a directory to save the files
            save_dir = os.path.join(out_dir, "hard_gating")
            os.makedirs(save_dir, exist_ok=True)

            hard_gating = HardGating(img_shape=config['output']['img_shape'], gating_thresh=config['hard_gating']['thresh'], device=device)
            output_vol = hard_gating.run(ksp, coord, dcf, resp)

            save_nifti_volume(output_vol, filename="hard_gating.nii.gz", save_dir=save_dir)

    stop_time = time.time()
    logger.info(f"Total time taken: {(stop_time - start_time)/3600:.2f} hours.")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reconstruct image using different motion compensated algorithms."
        )
    parser.add_argument("-i", "--raw_path", type=str, help="Path to the MRI_Raw.h5 file.")
    parser.add_argument("--config_path", type=str, help="Path to the YAML configuration file.")

    args = parser.parse_args()
    main(args.raw_path, args.config_path)

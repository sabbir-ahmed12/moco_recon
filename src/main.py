import os 
import time
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

# Import the internal modules
from no_gating import no_gating
from hard_gating import hard_gating
from soft_gating import soft_gating
from imoco import imoco
from mocostorm import mocostorm
from xdgrasp import xdgrasp
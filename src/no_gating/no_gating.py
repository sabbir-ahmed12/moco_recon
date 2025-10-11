import logging
import numpy as np
import sigpy as sp
from recon.base import Recon

# Get the logger
logger = logging.getLogger(__name__)

class NoGating(Recon):
    def __init__(self, img_shape=(256, 256, 256), oversamp=1.25, kernel_width=4, device=-1):
        self.img_shape = img_shape
        self.oversamp_factor = oversamp
        self.kernel_width = kernel_width
        self.device = device

    def run(self, ksp, coord, dcf):
        logger.info(f"Performing no_gating reconstructions ...")
        coord = sp.to_device(coord, device=self.device)
        ksp = ksp * dcf
        num_coils = ksp.shape[0]
        xp = sp.Device(self.device).xp
        with sp.Device(self.device):
            img = 0
            for coil in range(0, num_coils):
                logger.info(f"Performing no_gating reconstruction for coil {coil}.")
                ksp_coil = sp.to_device(ksp[coil], device=self.device)
                img_coil = sp.nufft_adjoint(ksp_coil, coord, oshape=self.img_shape, oversamp=self.oversamp_factor, width=self.kernel_width)
                img = img + sp.to_device(img_coil * xp.conj(img_coil), device=-1)
            
            img = np.abs(np.sqrt(img))
        
        del img_coil, ksp_coil, dcf, coord, ksp
        img = np.transpose(img, (2, 1, 0))
        
        return img


    
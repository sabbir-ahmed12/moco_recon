import os
import time
import copy
import logging
import numpy as np
import sigpy as sp
from recon.base import Recon

# Get the logger
logger = logging.getLogger(__name__)


class HardGating(Recon):
    def __init__(self, img_shape=(256, 256, 256), 
                gating_thresh=50, 
                gating_weight=1.0, 
                oversamp=1.25, 
                flip=False, 
                kernel_width=2.5, 
                device=-1
                ):
        self.img_shape = img_shape
        self.gating_thresh = gating_thresh
        self.gating_weight = gating_weight
        self.flip = flip
        self.oversamp_factor = oversamp
        self.kernel_width = kernel_width
        self.device = device

    
    def __get_threshold_mask(self, resp, margin=5):
        
        margin = margin  # Remove 5% at both ends to threshold robustly
        # Estimate the standard deviation of the data using median based estimator
        sigma = 1.4628 * np.median(np.abs(resp - np.median(resp)))   # float
        # Standardize the signal with approx. unit variance and zero median and flips the signal
        resp = -1 * (resp - np.median(resp)) / sigma
        # Find the cut-off values beyond which data points are considered too low or too high
        thresh_extreme = [np.percentile(resp, margin), np.percentile(resp, 100 - margin)]
        idx = (resp >= thresh_extreme[0]) & (resp < thresh_extreme[1])
        idx_exclude = (resp < thresh_extreme[0]) | (resp >= thresh_extreme[1])
        # Store the samples which are within the cut-off range
        resp_temp = resp[idx]

        # Get the robust midpoint (ref. value) to distinguish between low and high regions
        thresh = np.percentile(resp_temp, self.gating_thresh)

        if self.flip:
            resp *= 1
        
        # Create a binary mask of size resp
        mask = np.where(resp < thresh, 1, 0)
        mask[idx_exclude] = 0

        return mask

    
    def __get_gated_array(self, mask, ksp, coord, dcf):
        idx = mask == 1
        ksp = ksp[:, idx]
        coord = coord[idx]
        dcf = dcf[idx]
        
        return ksp, coord, dcf

    
    def run(self, ksp, coord, dcf, resp):
        start_time = time.time()

        ksp = copy.deepcopy(ksp)
        coord = copy.deepcopy(coord)
        dcf = copy.deepcopy(dcf)
        resp = copy.deepcopy(resp)

        mask = self.__get_threshold_mask(resp)
        gated_ksp, gated_coord, gated_dcf = self.__get_gated_array(mask, ksp, coord, dcf)

        # Free the memory
        del ksp, coord, dcf, resp

        logger.info(f"Performing hard_gating reconstructions ...")
        gated_coord = sp.to_device(gated_coord, device=self.device)
        gated_ksp = gated_ksp * gated_dcf
        num_coils = gated_ksp.shape[0]
        xp = sp.Device(self.device).xp

        with sp.Device(self.device):
            img = 0
            for coil in range(0, num_coils):
                logger.info(f"Performing hard_gating reconstruction for coil {coil}.")
                ksp_coil = sp.to_device(gated_ksp[coil], device=self.device)
                img_coil = sp.nufft_adjoint(ksp_coil, gated_coord, oshape=self.img_shape, oversamp=self.oversamp_factor, width=self.kernel_width)
                img = img + sp.to_device(img_coil * xp.conj(img_coil), device=-1)
            
            img = np.abs(np.sqrt(img))
        
        del img_coil, ksp_coil, gated_dcf, gated_coord, gated_ksp
        img = np.transpose(img, (2, 1, 0))
        
        stop_time = time.time()
        logger.info(f"Finished hard_gating reconstruction! Took: {(stop_time - start_time)/3600:.2f} hours.")

        return img
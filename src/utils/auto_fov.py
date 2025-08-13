import os
import logging
import numpy as np
import sigpy as sp
from scipy import ndimage
from PIL import Image
from skimage import measure, transform

# Get the logger
logger = logging.getLogger(__name__)

# Load the internal modules
from utils.misc import minmax_normalize

# Get the largest connected component (object)
def largest_cc(mask):
    labels = measure.label(mask)
    assert labels.max() != 0
    largest_cc = (labels == np.argmax(np.bincount(labels.flat)[1:]) + 1)

    return largest_cc


def auto_fov(ksp, 
             coord, 
             dcf, 
             output_dir,
             num_readouts=100,
             thresh=0.4,
             radial=False,
             device=-1):
    """Automatic estimation of field of view (FOV). FOV is estimated
    by thresholding a low resolution gridded image.
        - Firstly, a large FOV scout is reconstructed and the main anatomoy is extracted from it.
        - Secondly, a tight centered box is measured around the anatomy.
        - Finally, the coordinates are rescaled to make sure the final recons's FOV just fits that box.

    Parameters:
    -----------
        ksp : np.ndarray
            k-space measurements of shape (num_coil, num_traj, num_readouts)
            where: 
                - num_coil is the number of channels, 
                - num_traj is the number of trajectories, 
                - num_readouts is the number of readouts.
        
        coord : np.ndarray
            k-space coordinates of shape (num_traj, num_readouts, num_dim)
            where - num_dim is the k-space coordinates shape.
        
        dcf : np.ndarray
            Density compensation factor of shape (num_traj, num_readouts)
        
        num_readouts : int
            Number of read-out points
        
        thresh : float
            Threshold between 0 and 1
        
        device  : sigpy.device
            Computing device

    Returns:
    --------
        coord : np.ndarray
            Modified coordinates of k-space.
    """
    logger.info(f"Estimating FOV automatically ...")

    # Create a directory to store the necessary files
    autofov_diagnostics = os.path.join(output_dir, "autofov_diagnostics")
    os.makedirs(autofov_diagnostics, exist_ok=True)

    # Set the device
    device = sp.Device(device)
    # Get the numpy/cupy based on the device
    xp = device.xp

    with device:
        if  radial:
            readout_center = ksp.shape[2] // 2
            readout_range = slice(
                readout_center - num_readouts // 2, 
                readout_center + num_readouts // 2, 1)
            logger.info(f"Readout range: {readout_range}")
        
        else:
            readout_range = slice(0, num_readouts, 1)
            logger.info(f"Readout range: {readout_range}")

        logger.info(f"Estimated FOV: {sp.estimate_shape(coord)}")

        ksp_cropped = ksp[:, :, readout_range]
        coord_cropped = coord[:, readout_range, :]
        dcf_cropped = dcf[:, readout_range]

        # Double the FOV to make sure the anatomy is not cropped
        coord_x2 = sp.to_device(coord_cropped * 2, device)
        num_coils = len(ksp_cropped)

        # Estimate the image shape after cropping
        img_shape = sp.estimate_shape(coord_cropped)
        logger.info(f"Cropped image shape: {img_shape}")

        # Estimate the image shape after doubling the coordinate
        img_x2_shape = sp.estimate_shape(coord_x2)
        logger.info(f"Cropped image shape (after doubling coordinates): {img_x2_shape}")

        # Get the center index of the image
        img_x2_center = [i // 2 for i in img_x2_shape]

        img_x2 = sp.nufft_adjoint(sp.to_device(dcf_cropped * ksp_cropped, device), 
                                  coord_x2, 
                                  [num_coils, *img_x2_shape]
                                  )
        # Get the RSS image combining all coils
        img_x2 = xp.sum(xp.abs(img_x2) ** 2, axis=0) ** 0.5
        img_x2 = sp.to_device(img_x2)
        # Smooth the image to reduce salt-and-pepper noise
        img_x2 = ndimage.median_filter(img_x2, (3, 3, 3))

        # Normalize the image [0, 255] and save a slice from each view
        img_cor = minmax_normalize(img_x2[:, img_x2.shape[1] // 2, :], 0, 255)
        img_cor = Image.fromarray(transform.resize(img_cor, (256, 256)))
        # Convert to gray scale (8-bit pixels, black and white)
        img_cor = img_cor.convert("L")
        img_cor.save(os.path.join(autofov_diagnostics, "autofov_lowres_coronal.jpg"))

        img_sag = minmax_normalize(img_x2[:, :, img_x2.shape[2] // 2], 0, 255)
        img_sag = Image.fromarray(transform.resize(img_sag, (256, 256)))
        img_sag = img_sag.convert("L")
        img_sag.save(os.path.join(autofov_diagnostics, "autofov_lowres_sagittal.jpg"))

        img_ax = minmax_normalize(img_x2[img_x2.shape[0] // 2, :, :], 0, 255)
        img_ax = Image.fromarray(transform.resize(img_ax, (256, 256)))
        img_ax = img_ax.convert("L")
        img_ax.save(os.path.join(autofov_diagnostics, "autofov_lowres_axial.jpg"))

        # Get the absolute threshold value from fraction
        thresh *= img_x2.max()
        box_c = img_x2 > thresh
        # Keep only the largest connected component -> a clean foreground mask
        box_c = largest_cc(box_c).astype(float)

        mask_cor = minmax_normalize(box_c[:, box_c.shape[1] // 2, :], 0, 255)
        mask_cor = Image.fromarray(transform.resize(mask_cor, (256, 256)))
        # Convert to black and white (1-bit pixel) image
        mask_cor = mask_cor.convert("1")
        mask_cor.save(os.path.join(autofov_diagnostics, "autofov_mask_coronal.jpg"))

        mask_sag = minmax_normalize(box_c[:, :, box_c.shape[2] // 2], 0, 255)
        mask_sag = Image.fromarray(transform.resize(mask_sag, (256, 256)))
        mask_sag = mask_sag.convert("1")
        mask_sag.save(os.path.join(autofov_diagnostics, "autofov_mask_sagittal.jpg"))

        mask_ax = minmax_normalize(box_c[box_c.shape[0] // 2, :, :], 0, 255)
        mask_ax = Image.fromarray(transform.resize(mask_ax, (256, 256)))
        mask_ax = mask_ax.convert("1")
        mask_ax.save(os.path.join(autofov_diagnostics, "autofov_mask_axial.jpg"))

        # Get the index arrays for non-zero voxels (one array per dimension)
        box_c_idx = np.nonzero(box_c)
        # Get the centered symmetric box that tightly covers the object (anatomy)
        box_c_shape = np.array([int(np.abs(box_c_idx[i] - img_x2_center[i]).max()) * 2
                                for i in range(img_x2.ndim)
                             ])

        # Calculate the box size relative to the original grid       
        img_scale = box_c_shape / img_shape
        logger.info(f"Scaling factor: {img_scale}.")

        # Save the scaling factor
        np.save(os.path.join(autofov_diagnostics, "fov_scale_factor.txt"), img_scale)

        # For radial, inflate the scale(x2) to add conservative fudge for circle-in-sqaure
        if radial:
            img_scale *= 2

        # Rescale (<1 = reduce FOV i.e. zooms in on the object and vice-versa)
        coord *= img_scale
        logger.info(f"Auto FOV output shape: {sp.estimate_shape(coord)}.")

        # Reconstruct the image again at the new (smaller) FOV
        coord_cropped = coord[:, readout_range, :]
        coord_cropped = sp.to_device(coord_cropped, device)
        img_shape = sp.estimate_shape(coord_cropped)
        logger.info(f"Performing NUFFT adjoint on cropped coordinate.")
        img_cropped = sp.nufft_adjoint(sp.to_device(dcf_cropped * ksp_cropped, device), coord_cropped, [num_coils, *img_x2_shape])
        # Get the RSS image combining all coils
        img = xp.sum(xp.abs(img_cropped) ** 2, axis=0) ** 0.5
        img = sp.to_device(img)

        # Save the new image
        img_cor = minmax_normalize(img[:, img.shape[1] // 2, :], 0, 255)
        img_cor = Image.fromarray(transform.resize(img_cor, (256, 256)))
        img_cor = img_cor.convert("L")
        img_cor.save(os.path.join(autofov_diagnostics, "autofov_cropped_coronal.jpg"))

        img_sag = minmax_normalize(img[:, :, img.shape[2] // 2], 0, 255)
        img_sag = Image.fromarray(transform.resize(img_sag, (256, 256)))
        img_sag = img_sag.convert("L")
        img_sag.save(os.path.join(autofov_diagnostics, "autofov_cropped_sagittal.jpg"))

        img_ax = minmax_normalize(img[img.shape[0] // 2, :, :], 0, 255)
        img_ax = Image.fromarray(transform.resize(img_ax, (256, 256)))
        img_ax = img_ax.convert("L")
        img_ax.save(os.path.join(autofov_diagnostics, "autofov_cropped_axial.jpg"))

        return coord









            
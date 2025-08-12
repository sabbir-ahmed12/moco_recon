import os
import h5py
import logging
import numpy as np


# Get the logger for logging
logger = logging.getLogger(__name__)

def convert_ute(h5_path, 
                output_dir, 
                spoke_downsample_factor=1.0,
                pre_whiten=False, 
                apodise=False,
                compress_coils=False
                ):
    """
    Convert MRI_Raw.h5 file to ksp.npy, coord.npy, dcf.npy, tr.npy, noise.npy 
    and resp.npy files.

    Parameters:
    -----------
    h5_path (str): path of the MRI_Raw.h5 file. 
    output_dir (str): path to save the output files.
    """
    logger.info(f"Converting {h5_path} file to npy files ...")

    noise = 0
    os.makedirs(output_dir, exist_ok=True)

    with h5py.File(h5_path, "r") as hf:
        logger.info(f"Reading the MRI_Raw.h5 file ...")
        try:
            num_encodes = np.squeeze(hf["Kdata"].attrs["Num_Encodings"])
            num_coils = np.squeeze(hf["Kdata"].attrs["Num_Coils"])
            num_frames = np.squeeze(hf["Kdata"].attrs["Num_Frames"])

            trajectory_type = [
                np.squeeze(hf["Kdata"].attrs["trajectory_typeX"]),
                np.squeeze(hf["Kdata"].attrs["trajectory_typeY"]),
                np.squeeze(hf["Kdata"].attrs["trajectory_typeZ"]),
            ]

            dft_needed = [
                np.squeeze(hf["Kdata"].attrs["dft_neededX"]),
                np.squeeze(hf["Kdata"].attrs["dft_neededY"]),
                np.squeeze(hf["Kdata"].attrs["dft_neededZ"]),
            ]

            logger.info(f"Number of frames: {num_frames}.")
            logger.info(f"Number of coils: {num_coils}.")
            logger.info(f"Number of encodings: {num_encodes}.")

        except Exception:
            logger.error("Missing H5 Attributes ...")

            num_coils = 0
            while f"KData_E0_C{num_coils}" in hf["Kdata"]:
                num_coils += 1

            num_encodes = 0
            while f"KData_E{num_encodes}_C0" in hf["Kdata"]:
                num_encodes += 1

        for encode in range(num_encodes):
            logger.info(f"Processing encode {encode} ...")
            encode_dir = os.path.join(output_dir, f"encode_{encode}")
            os.makedirs(encode_dir, exist_ok=True)

            try:
                time = np.squeeze(hf["Gating"][f"time"])
            except Exception:
                time = np.squeeze(hf["Gating"][f"TIME_E{encode}"])

            order = np.argsort(time)
            d_time = time[order]

            try:
                resp = np.squeeze(hf["Gating"][f"resp"])
            except Exception:
                resp = np.squeeze(hf["Gating"][f"RESP_E{encode}"])
            resp = resp[order]

            try:
                ecg = np.squeeze(hf["Gating"][f"ecg"])
            except Exception:
                ecg = np.squeeze(hf["Gating"][f"ECG_E{encode}"])
            
            ecg = ecg[order]

            coord = []
            for i in ["Z", "Y", "X"]:
                coord.append(hf["Kdata"][f"K{i}_E{encode}"][0][order])
            coord = np.stack(coord, axis=-1)

            dcf = np.array(hf["Kdata"][f"KW_E{encode}"][0][order])

            ksp = []
            for c in range(num_coils):
                real = hf["Kdata"][f"KData_E{encode}_C{c}"]["real"][0][order]
                imag = hf["Kdata"][f"KData_E{encode}_C{c}"]["imag"][0][order]
                ksp.append(real + 1j * imag)
            ksp = np.stack(ksp, axis=0)

            try:
                noise = hf["Kdata"]["Noise"]["real"] + 1j * hf["Kdata"]["Noise"]["imag"]
                
                if pre_whiten:
                    logger.error("Prewhitening not implemented in this snippet.")
                else:
                    ksp /= np.abs(ksp).max()

            except Exception as err:
                print(f"Noise processing error: {err}")
                ksp /= np.abs(ksp).max()

            if apodise:
                kr = np.sqrt(np.sum(coord ** 2, axis=2))
                kmax = np.max(kr)
                fermi = 1 / (1 + np.exp((kr - kmax / ap_alpha) / ap_beta))
                dcf *= fermi

            if compress_coils:
                logger.error("Coil compression not implemented in this snippet.")

            total_spokes = ksp.shape[1]
            num_spokes = int(total_spokes // spoke_downsample_factor)
            ksp = ksp[:, :num_spokes, :]
            coord = coord[:num_spokes, :, :]
            dcf = dcf[:num_spokes, :]
            resp = resp[:num_spokes]

            tr = d_time[1] - d_time[0]
            logger.info(f"Time of repetition: {tr}.")

            # === Save the npy files ===
            np.save(os.path.join(encode_dir, "ksp.npy"), ksp)
            np.save(os.path.join(encode_dir, "coord.npy"), coord)
            np.save(os.path.join(encode_dir, "dcf.npy"), dcf)
            np.save(os.path.join(encode_dir, "resp.npy"), resp / resp.max())
            np.save(os.path.join(encode_dir, "tr.npy"), np.array([tr]))
            np.save(os.path.join(encode_dir, "noise.npy"), noise)

            logger.info(f"Saved data for encode {encode} in {encode_dir}.")

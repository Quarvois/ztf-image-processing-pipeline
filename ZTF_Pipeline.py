from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import random as rd
from typing import Optional, List, Tuple, Dict

import numpy as np
import pandas as pd 
import copy
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import gaussian_filter, binary_dilation, generate_binary_structure
from reproject import reproject_interp
import matplotlib.pyplot as plt
from astropy.stats import SigmaClip
from photutils.background import Background2D, MedianBackground
from photutils.aperture import CircularAperture, aperture_photometry
















# ----------------------------
# CONFIG
# ----------------------------

@dataclass
class MaskConfig:
    # Sources Detection (simple threshold)
    detect_sigma: float = 3.0       # treshold = median + detect_sigma * sigma_bruit
    smooth_sigma_pix: float = 1.5   # smoothing to avoid detecting noise as sources
    dilate_pix: int = 8             # Dilatation to cover halos around sources (stars/galaxies)

    # Saturation / edges
    edge_margin: int = 10           #Thickness of the image edge to be ignored by default.
    saturate_dilate_pix: int = 3   #Very strong expansion specific to saturated pixels


@dataclass
class BackgroundConfig:
    box_size: int = 32         #The size of the squares (in pixels) used to locally estimate the sky background.
    filter_size: int = 3        #The size of the smoothing filter applied to the sky background map to avoid abrupt transitions.
    sigma_clip: float = 3.0     #The threshold for rejecting outlier pixels when calculating the background.


@dataclass
class PipelineConfig:
    mask: MaskConfig = field(default_factory=MaskConfig)
    bkg: BackgroundConfig = field(default_factory=BackgroundConfig)
    pixel_scale: float = 1.01
    dtype: type = np.float32 #32bit to save some RAM 





















# ----------------------------
# TREATMENT AND TOOLS
# ----------------------------

class SingleFrame:
    """
    Represents a single ZTF FITS image and provides methods for astronomical 
    preprocessing, including masking, background estimation, and PSF matching.
    """

    def __init__(self, fits_path: str | Path, config: PipelineConfig):
        """
        Initializes the SingleFrame object by loading data and WCS from a FITS file.
        Args:
            fits_path: Path to the .fits file.
            config: Configuration object containing parameters for masking and background.
        """
        self.path = Path(fits_path)
        self.cfg = config
        with fits.open(self.path) as hdul:
            hdu = hdul[0]
            self.data = np.array(hdu.data, dtype=self.cfg.dtype)
            self.header = hdu.header.copy()
            
        self.wcs = WCS(self.header)
        self.seeing = self.header.get("SEEING", None)
        self.zp = self.header.get("MAGZP", None)
        self.saturate = self.header.get("SATURATE", None)
        
        raw_date = self.header.get('SHUTOPEN', self.header.get('OBSMJD', 'Unknown'))
        self.date_str = str(raw_date)[:19].replace('T', ' ')
        
        
    
    def to_hdu(self) -> fits.PrimaryHDU:
            """
            Génère un objet PrimaryHDU en synchronisant le WCS actuel 
            et les métadonnées (ZP, SEEING) dans le header.
            """
            new_header = self.header.copy()

            # We generate a new header from the current WCS to ensure all WCS keywords are correct
            wcs_header = self.wcs.to_header()
            new_header.update(wcs_header)           
            # We also update the ZP and SEEING in the header to reflect any changes made during processing
            if self.zp is not None:
                new_header["MAGZP"] = self.zp
            if self.seeing is not None:
                new_header["SEEING"] = self.seeing
            return fits.PrimaryHDU(data=self.data, header=new_header)





    def save(self, out_path: str | Path, overwrite: bool = True) -> Path:
        """
        Save the current image data with a header that reflects the current WCS and metadata to a new FITS file.
        """
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self.to_hdu().writeto(out_path, overwrite=overwrite)
        return out_path





    @staticmethod
    def _robust_sigma(x: np.ndarray) -> float:
        """
        Calculates a noise estimate (sigma) that is resistant to outliers (like stars).
        It uses the Median Absolute Deviation (MAD) scaled to match a normal distribution.

        Args:
            x: Input image array.
        Returns:
            The robust standard deviation of the background noise.
        """
        med = np.nanmedian(x)
        mad = np.nanmedian(np.abs(x - med))
        return 1.4826 * mad




    # ---------- Mask ----------
    def mask_edges(self) -> np.ndarray:
        """
        Creates a boolean mask for the image boundaries to avoid artifacts near the CCD edges.
        Returns:
            A boolean array where True indicates a pixel within the defined margin.
        """
        m = np.zeros(self.data.shape, dtype=bool)
        e = self.cfg.mask.edge_margin
        if e > 0:
            m[:e, :] = True
            m[-e:, :] = True
            m[:, :e] = True
            m[:, -e:] = True
        return m




    def mask_saturation(self) -> np.ndarray:
        """
        Identifies and masks saturated pixels (blooms) based on the SATURATE header value.
        The mask is dilated to cover the surrounding affected area.
        Returns:
            A boolean mask covering saturated regions.
        """
        if self.saturate is None:
            return np.zeros(self.data.shape, dtype=bool)

        m = self.data >= float(self.saturate)
        if np.any(m) and self.cfg.mask.saturate_dilate_pix > 0:
            struct = generate_binary_structure(2, 2)
            m = binary_dilation(m, structure=struct, iterations=self.cfg.mask.saturate_dilate_pix)
        
        self.data[m] = np.nan 
        return m




    def mask_sources_simple(self) -> np.ndarray:
        """
        Detects bright sources (stars/galaxies) using a simple thresholding method.
        The image is smoothed, then pixels above N-sigma are flagged and dilated.
        Returns:
            A boolean mask covering detectable astronomical sources.
        """
        img = self.data

        # PSF homogenization
        img_s = gaussian_filter(img, sigma=self.cfg.mask.smooth_sigma_pix)
        med = np.nanmedian(img_s)
        sig = self._robust_sigma(img_s)
        thresh = med + self.cfg.mask.detect_sigma * sig
        m = img_s > thresh

        # Dilatation to cover halos
        if self.cfg.mask.dilate_pix > 0:
            struct = generate_binary_structure(2, 2)
            for _ in range(self.cfg.mask.dilate_pix):
                m = binary_dilation(m, structure=struct)

        return m




    def build_mask(self) -> np.ndarray:
        """
        Combines edge, saturation, and source masks into a single master mask.
        This mask is used to ignore non-background pixels during statistical calculations.
        Returns:
            The final unified boolean mask (True = pixel to be ignored).
        """
        m = np.zeros(self.data.shape, dtype=bool)
        m |= self.mask_edges()
        m |= self.mask_saturation()
        m |= self.mask_sources_simple()
        return m

    
    
    
    def estimate_background(self, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calculates a 2D map of the sky background using the photutils library.
        
        This version relies exclusively on Background2D for professional-grade 
        background estimation, ensuring robust handling of masked areas.
        """
        sigma_clip = SigmaClip(sigma=self.cfg.bkg.sigma_clip) # It ignores values far from the mean to avoid star contamination
        
        bkg = Background2D(
            self.data,
            box_size=self.cfg.bkg.box_size,
            filter_size=(self.cfg.bkg.filter_size, self.cfg.bkg.filter_size),
            sigma_clip=sigma_clip,
            bkg_estimator=MedianBackground(),
            mask=mask,
        )                                                   # This divides the image into boxes and calculates the median sky in each
        return np.array(bkg.background, dtype=self.cfg.dtype)





    def subtract_background(self, bkg_map: np.ndarray) -> None:
        """
        Subtracts a provided background map from the image data to center the flux at zero.
        Args:
            bkg_map: The 2D background model to subtract.
        """
        self.data = np.array(self.data - bkg_map, dtype=self.cfg.dtype)





    # ---------- Photometry / PSF / Align ----------
    def rescale_to_zp(self, zp_target: float) -> None:
        """
        Normalizes the image flux to a common Magnitude Zero-Point (ZP).
        This ensures that a source has the same numerical value across different frames.
        Args:
            zp_target: The target zero-point for scaling.
        Raises:
            ValueError: If the original MAGZP is missing from the FITS header.
        """
        if self.zp is None:
            raise ValueError("MAGZP missing from header.")
        scale = 10 ** ((zp_target - float(self.zp)) / 2.5)
        self.data = np.array(self.data * scale, dtype=self.cfg.dtype)
        self.zp = zp_target





    def psf_homogenize_to(self, seeing_target: float) -> None:
        """
        Degrades the image resolution (seeing) to match a specific target value.
        This is done via Gaussian convolution to ensure the Reference and Science 
        images have matching star shapes before subtraction.
        Args:
            seeing_target: The desired FWHM (in arcseconds) of the output image.
        """
        if self.seeing is None:
            raise ValueError("SEEING absent du header.")
        seeing_from = float(self.seeing)
        seeing_to = float(seeing_target)
        if seeing_to <= seeing_from:
            return

        fwhm_diff = np.sqrt(seeing_to**2 - seeing_from**2)
        sigma_pix = fwhm_diff / (2.3548 * self.cfg.pixel_scale)  # 2.3548 --> Convert FWHM to sigma in pixels
        self.data = gaussian_filter(self.data, sigma=sigma_pix).astype(self.cfg.dtype)
        self.seeing = seeing_target





    def reproject_to(self, target_wcs: WCS, target_shape: Tuple[int, int]) -> None:
        """
        Resamples and aligns the image onto a new WCS grid (geometric registration).
        Essential for ensuring every pixel in the Science image corresponds 
        to the exact same sky coordinate as the Reference image.
        Args:
            target_wcs: The World Coordinate System to project onto.
            target_shape: The dimensions of the output image.
        """
        arr, _ = reproject_interp((self.data, self.wcs), target_wcs, shape_out=target_shape, order="bilinear")
        self.data = np.array(arr, dtype=self.cfg.dtype)
        self.wcs = target_wcs
        
        
    
    
    

    def get_aperture_flux(self, coords: np.ndarray, r: float = 5.0) -> pd.DataFrame:
        """
        Do aperture photometry at specified pixel coordinates to extract fluxes for lightcurve construction.
        
        Args:
            coords: Array-like with shape (N, 2) containing the pixel coordinates of the centers of the apertures.
            r: Radius of the circular aperture in pixels.
            
        Returns:
            Pandas DataFRame with columns ['id', 'xcenter', 'ycenter', 'aperture_sum', 'date', 'zp'] where:
                - 'id' is a unique identifier for each aperture 
        """
        apertures = CircularAperture(coords, r=r)
        # Flux
        phot_table = aperture_photometry(self.data, apertures)
        # Error estimation: sigma_sky * sqrt(N_pix)
        sig_sky = self._robust_sigma(self.data) 
        n_pix = apertures.area
        # Error = sigma_sky * sqrt(N_pix) because the noise adds in quadrature, and we assume the background noise dominates.
        # We can add the poisson noise of the source if we want, but in difference images the background noise is usually dominant, so we keep it simple for now.
        flux_err = sig_sky * np.sqrt(n_pix)
        
        df = phot_table.to_pandas()
        df['flux_err'] = flux_err
        df['date'] = self.date_str
        df['mjd'] = self.header.get('OBSMJD', 0)
        df['zp'] = self.zp
        
        return df




            
    def summary(self):
        """Returns a concise summary of the image properties for quick inspection."""
        print(f"--- Fichier : {self.path.name} ---")
        print(f"Dimensions : {self.data.shape}")
        print(f"ZP         : {self.zp}")
        print(f"SEEING     : {self.seeing} arcsec")
        print(f"Filtre     : {self.header.get('FILTER')}")
        print(f"Saturation : {self.saturate} ADU")
        print(f"Dtype      : {self.data.dtype}")
        print(f"Exposition : {self.header.get('EXPTIME')} s")
        
        
        
        




















# --------------------------------------------
# FOLDER TREATMENT & REFERENCE CONSTRUCTION
# --------------------------------------------

class ZTFFolderPipeline:
    """
    Orchestrates the processing of a directory containing ZTF FITS files.
    It manages the alignment (WCS), the creation of a deep reference image, 
    and the preparation of science frames for temporal analysis (supernova hunting).
    """

    def __init__(self, folder: str | Path, config: PipelineConfig, pattern: str = "*.fits"):
        """
        Initializes the pipeline by scanning a folder for specific FITS files.
        Args:
            folder: Path to the directory containing the data.
            config: PipelineConfig instance holding all processing parameters.
            pattern: Glob pattern to filter files (default is all .fits).
        
        Raises:
            FileNotFoundError: If no files matching the pattern are found.
        """
        self.folder = Path(folder)
        self.cfg = config
        self.pattern = pattern
        self.files = sorted(self.folder.glob(self.pattern))
        if not self.files:
            raise FileNotFoundError(f"Aucun FITS trouvé dans {self.folder} avec pattern={pattern}")
        self.target_wcs: Optional[WCS] = None
        self.target_shape: Optional[Tuple[int, int]] = None


    def plot_seeing_hist(
        self,
        folder: str | Path,
        pattern: str = "*.fits",
        bins: int = 30,
        seeing_key: str = "SEEING",
        ignore_nonfinite: bool = True,
        show: bool = True,
    ) -> dict:
        """
        Plots the distribution of the 'SEEING' values across all FITS files in a folder.
        This is a crucial step to understand the quality of the dataset and to set appropriate thresholds for reference image construction.
        
        Args:
            folder: Directory containing the FITS files to analyze.
        """
        folder = Path(folder)
        files = sorted(folder.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No FITS files found in {folder} with pattern={pattern}")

        seeings = []
        skipped = 0

        for p in files:
            try:
                with fits.open(p, memmap=True) as hdul:
                    val = hdul[0].header.get(seeing_key, None)
            except Exception:
                skipped += 1
                continue

            if val is None:
                skipped += 1
                continue

            try:
                v = float(val)
            except Exception:
                skipped += 1
                continue

            if ignore_nonfinite and (not np.isfinite(v)):
                skipped += 1
                continue

            seeings.append(v)

        if len(seeings) == 0:
            raise ValueError(
                f"No valid {seeing_key} values found. "
                f"(files={len(files)}, skipped={skipped})"
            )

        x = np.asarray(seeings, dtype=float)
        mean = float(np.mean(x))
        median = float(np.median(x))
        std = float(np.std(x, ddof=1)) if x.size >= 2 else 0.0

        plt.figure(figsize=(9, 5))
        plt.hist(x, bins=bins, color="darkblue", edgecolor="k", alpha=0.9)
        plt.axvline(mean, linestyle="--", label=f"Mean = {mean:.3f}", color="r")
        plt.axvline(median, linestyle="--", label=f"Median = {median:.3f}", color="g")
        plt.axvline(mean - std, linestyle=":", label=f"±1σ = {std:.3f}", color="r")
        plt.axvline(mean + std, linestyle=":", color="r")

        plt.xlabel(f"{seeing_key} (arcsec)")
        plt.ylabel("Files Count")
        plt.title(f"Distribution of {seeing_key} — N={x.size} (skipped={skipped})")
        plt.legend()
        plt.tight_layout()

        if show:
            plt.show()

        return {
            "N": int(x.size),
            "skipped": int(skipped),
            "mean": mean,
            "median": median,
            "std": std,
            "min": float(np.min(x)),
            "max": float(np.max(x)),
        }
    


    def set_target_from_file(self, fits_path: str | Path) -> None:
        """
        Defines the 'Master Geometry' (WCS and image shape) based on a specific file.
        All other images in the pipeline will be reprojected to match this frame.
        Args:
            fits_path: Path to the FITS file to be used as the geometric reference.
        """
        f = SingleFrame(fits_path, self.cfg)
        self.target_wcs = f.wcs
        self.target_shape = f.data.shape




    def _ensure_target(self) -> None:
        """
        Internal safety check to ensure a target WCS and shape are defined.
        If not already set, it defaults to using the first file in the dataset.
        """
        if self.target_wcs is None or self.target_shape is None:
            # par défaut: premier fichier, mais tu peux améliorer: choisir la "meilleure" image
            self.set_target_from_file(self.files[0])




    def prepare_frame(self, fits_path: str | Path, zp_target: Optional[float], seeing_target: Optional[float]) -> SingleFrame:
        """
        Executes the full preprocessing sequence on a single FITS file to make it 
        analysis-ready.

        The sequence is: 
        1. Alignment (Reprojection) -> 2. Masking -> 3. Background Subtraction 
        -> 4. Flux Scaling (ZP) -> 5. Resolution Matching (Seeing).

        Args:
            fits_path: Path to the raw FITS file.
            zp_target: The Magnitude Zero-Point to reach.
            seeing_target: The FWHM (arcsec) to reach via PSF homogenization.

        Returns:
            A preprocessed SingleFrame object.
        """
        self._ensure_target()

        f = SingleFrame(fits_path, self.cfg)
        f.reproject_to(self.target_wcs, self.target_shape)

        mask = f.build_mask()
        bkg = f.estimate_background(mask=mask)
        f.subtract_background(bkg)

        if zp_target is not None:
            f.rescale_to_zp(zp_target)

        if seeing_target is not None:
            f.psf_homogenize_to(seeing_target)

        return f

    


    def build_reference(
        self,
        zp_target: float,
        seeing_target: float,
        max_frames: Optional[int] = None,
        save_path: Optional[str | Path] = None,
        overwrite: bool = True,
        show_ref: bool = True,
    ) -> SingleFrame:
        """
        Creates a high Signal-to-Noise Ratio (SNR) reference image by stacking 
        multiple frames.

        It filters available images to only include those with a 'SEEING' quality 
        better than the target. It then uses a median stack to remove transient 
        objects (like asteroids or cosmic rays) and reveal the static sky.

        Args:
            zp_target: Target Zero-Point for the stack.
            seeing_target: Maximum allowed seeing; also the target for homogenization.
            max_frames: Limit the number of images to stack (to save time/memory).
            save_path: Path to save the resulting reference FITS file.
            overwrite: If True, replaces existing file at save_path.
            show_ref: If True, displays the final reference image using matplotlib.

        Returns:
            A SingleFrame object representing the deep reference image.
            
        Raises:
            ValueError: If no images meet the seeing quality criteria.
        """
        self._ensure_target()
        
        files_to_process = list(self.files) 
        rd.shuffle(files_to_process)
        
        stack = []
        skipped_count = 0
        
        for p in files_to_process:
            # Limit the number of frames to stack if max_frames is set (for testing or memory constraints)
            if max_frames is not None and len(stack) >= max_frames:
                break
                
            # Seeing request check
            with fits.open(p) as hdul:
                current_seeing = hdul[0].header.get("SEEING", 6) # 6 si inconnu
            
            # Seeing check
            if current_seeing >= seeing_target:
                skipped_count += 1
                continue
            
            # If the image passes the seeing check, we prepare it and add to the stack
            fr = self.prepare_frame(p, zp_target=zp_target, seeing_target=seeing_target)
            stack.append(fr.data)

        if not stack:
            raise ValueError(f"No image found with seeing <= {seeing_target}")

        print(f"Reference built with {len(stack)} images ({skipped_count} skipped).")

        # Median compute
        ref = np.nanmedian(np.stack(stack, axis=0), axis=0).astype(self.cfg.dtype)

        ref_frame = SingleFrame(self.files[0], self.cfg)
        ref_frame.reproject_to(self.target_wcs, self.target_shape)
        ref_frame.data = ref
        ref_frame.zp = zp_target
        ref_frame.seeing = seeing_target
        
        # Add metadata about the stacking process
        ref_frame.header["NCOMBINE"] = len(stack)
        ref_frame.header["METHOD"] = "MEDIAN_STACK"

        if show_ref:
            plt.figure(figsize=(10, 8))
            vmin, vmax = np.nanpercentile(ref, [1, 99])
            plt.imshow(ref, origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
            plt.colorbar(label='ADU')
            plt.title(f"Reference (N={len(stack)}, Target Seeing: {seeing_target})")
            plt.show()

        if save_path is not None:
            ref_frame.save(save_path, overwrite=overwrite)
            print(f"Reference image saved successfully: {save_path}")
            
        print(f"median of ref = {np.nanmedian(ref):.2f} ADU")
        print(f"min ADU : {np.nanmin(ref):.2f}")
        return ref_frame
































# --------------------------------------------
# DIFFERENCE IMAGE CONSTRUCTION
# --------------------------------------------

class ZTFDifferencePipeline:
    def __init__(self, folder_pipeline: ZTFFolderPipeline, reference_frame: SingleFrame):
        """
        Initialise le pipeline de soustraction.
        Le dossier de sortie 'diffimg' sera créé dynamiquement pour chaque image.
        """
        self.pipe = folder_pipeline
        self.reference = reference_frame
        self.inventory = self._build_inventory()





    def _build_inventory(self) -> pd.DataFrame:
        """Scan les headers pour extraire les dates (SHUTOPEN ou OBSMJD)."""
        data = []
        for p in self.pipe.files:
            try:
                with fits.open(p) as hdul:
                    header = hdul[0].header
                    raw_date = header.get('SHUTOPEN', header.get('OBSMJD', None))
                    if raw_date:
                        dt = pd.to_datetime(raw_date)
                    else:
                        dt = pd.NaT # Not a Time 
                        
                    data.append({'path': Path(p), 'date': dt})
            except Exception as e:
                print(f"Erreur lecture {p.name}: {e}")
        
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date']) 
        return df






    def subtract_range(self, start_date: str, end_date: str, save: bool = True, force: bool = False):
        """Subtracts all science frames within a specified date range from the reference image."""
        t_start = pd.to_datetime(start_date)
        t_end = pd.to_datetime(end_date)
        mask = (self.inventory['date'] >= t_start) & (self.inventory['date'] <= t_end)
        targets = self.inventory[mask]
        
        results = []
        
        for _, row in targets.iterrows():
            sci_path = row['path']
            diff_path = sci_path.parent.parent / "diffimg" / f"diff_{sci_path.name}"
            
            if not force and diff_path.exists():
                results.append(SingleFrame(diff_path, self.pipe.cfg))
                continue
            # We prepare the image, but we keep the original seeing of this frame
            sci_frame = self.pipe.prepare_frame(sci_path, zp_target=self.reference.zp, seeing_target=None)
            
            s_ref = float(self.reference.seeing)
            s_sci = float(sci_frame.seeing)

            # Convolution of the image with the lower seeing
            if s_sci < s_ref:
                sci_frame.psf_homogenize_to(s_ref)
                data_sci = sci_frame.data
                data_ref = self.reference.data
            elif s_ref < s_sci:
                ref_tmp = copy.deepcopy(self.reference)
                ref_tmp.psf_homogenize_to(s_sci)
                data_sci = sci_frame.data
                data_ref = ref_tmp.data
            else:
                data_sci = sci_frame.data
                data_ref = self.reference.data

            # Subtraction
            diff_data = data_sci - data_ref
            
            # Object output
            diff_frame = SingleFrame(sci_path, self.pipe.cfg)
            diff_frame.data = diff_data.astype(self.pipe.cfg.dtype)
            diff_frame.wcs = sci_frame.wcs
            diff_frame.zp = sci_frame.zp
            diff_frame.seeing = sci_frame.seeing 
            
            if save:
                diff_frame.save(diff_path, overwrite=True)
            results.append(diff_frame)
            
        return results
































# --------------------------------------------
# LIGHT CURVE EXTRACTION
# -------------------------------------------                      

class LightCurveExtractor:
    def __init__(self, diff_frames: List[SingleFrame]):
        """
        Take a list of SingleFrame (difference images).
        """
        self.frames = diff_frames

    def extract_at(self, x: float, y: float, r: float = 5.0) -> pd.DataFrame:
        """
        Generates the light curve for a given (x, y) position.
        """
        results = []
        coords = np.array([[x, y]])

        for frame in self.frames:
            df_step = frame.get_aperture_flux(coords, r=r)
            results.append(df_step)
            
        lc = pd.concat(results).sort_values('mjd').reset_index(drop=True)
        lc = lc.rename(columns={'aperture_sum': 'flux'})
        lc['snr'] = lc['flux'] / lc['flux_err']
        

        return lc[['mjd', 'date', 'flux', 'flux_err', 'snr', 'zp']]

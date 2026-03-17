# ZTF Image Processing Pipeline

> **A complete Python pipeline for difference imaging and supernova light curve extraction from Zwicky Transient Facility data.**

**Author:** Quentin Arvois  
**Context:** Final-year Master research project (TER) – Université Clermont Auvergne  
**Supervisor:** Philippe Rosnet & Marie Aubert (LPCA, UCA)  
**Field:** Observational astrophysics · Image processing · Time-domain astronomy

---

## Table of Contents

1. [Overview](#1-overview)
2. [Pipeline Architecture](#2-pipeline-architecture)
3. [Repository Structure](#3-repository-structure)
4. [Installation](#4-installation)
5. [Quick Start](#5-quick-start)
6. [Detailed Pipeline Workflow](#6-detailed-pipeline-workflow)
   - [6.1 Data Acquisition – FITS Download from IRSA](#61-data-acquisition--fits-download-from-irsa)
   - [6.2 Configuration](#62-configuration)
   - [6.3 Masking](#63-masking)
   - [6.4 Background Estimation & Subtraction](#64-background-estimation--subtraction)
   - [6.5 Photometric Rescaling](#65-photometric-rescaling)
   - [6.6 PSF Homogenization](#66-psf-homogenization)
   - [6.7 Astrometric Alignment](#67-astrometric-alignment)
   - [6.8 Reference Image Construction](#68-reference-image-construction)
   - [6.9 Difference Imaging](#69-difference-imaging)
   - [6.10 Aperture Photometry & Light Curve Extraction](#610-aperture-photometry--light-curve-extraction)
7. [Visual Results](#7-visual-results)
8. [Class Reference](#8-class-reference)
9. [Configuration Reference](#9-configuration-reference)
10. [Validation & Results](#10-validation--results)
11. [Dependencies](#11-dependencies)
12. [Notes & Known Limitations](#12-notes--known-limitations)

---

## 1. Overview

This repository contains a fully custom Python image-processing pipeline designed to extract differential photometry of transient astronomical sources (supernovae, novae, etc.) from calibrated ZTF science images.

The pipeline was built from scratch as part of a final-year Master research project at the Université Clermont Auvergne. It covers the entire chain from raw FITS data acquisition through to a validated supernova light curve, implementing each processing step independently using standard Python astronomy libraries.

**Key goals:**
- Reproduce and understand each step of a state-of-the-art difference imaging pipeline
- Validate the extracted light curves against the official ZTF photometry pipeline
- Build a modular, reusable and well-documented codebase

---

## 2. Pipeline Architecture

The pipeline is composed of four main processing layers, each implemented as a dedicated Python class:

```
┌─────────────────────────────────────────────────────────────┐
│                    sciimg_FITS_Request                       │
│         IRSA/ZTF Data Download  →  Local FITS files          │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                      SingleFrame                             │
│  Per-image preprocessing:                                     │
│  Masking · Background · ZP Rescaling · PSF · Reproject       │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                   ZTFFolderPipeline                          │
│  Dataset orchestration:                                       │
│  File scanning · Frame preparation · Reference construction  │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                 ZTFDifferencePipeline                        │
│  Difference imaging:                                          │
│  Science − Reference → Diff images by date range             │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                  LightCurveExtractor                         │
│ Aperture photometry on diff images → Light curve (DataFrame) │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Repository Structure

```
ztf-image-processing-pipeline/
│
├── ZTF_Pipeline.py                  # Core pipeline (all classes)
├── sciimg_FITS_Request.ipynb        # FITS data download from IRSA
├── Notebook_ZTF_Pipeline.ipynb      # Full pipeline demo notebook
├── Light_Curve_Extraction.ipynb     # Light curve extraction notebook
│
└── docs/
    ├── Comparison_of_ZTF17aadlxmv_Fluxes_3.png  # Light curve validation
    ├── Background2D_Test.png                      # Background estimation tests
    ├── Mask_Test.png                              # Masking tests
    ├── Plot3D_SmoothingSigma.png                  # 3D source detection
    └── Smoothing_impact.png                       # PSF homogenization demo
```

---

## 4. Installation

**Python 3.10+ required.**

Clone the repository and install the dependencies:

```bash
git clone https://github.com/Quarvois/ztf-image-processing-pipeline.git
cd ztf-image-processing-pipeline
pip install -r requirements.txt
```

**Core dependencies:**

```
numpy
scipy
pandas
matplotlib
astropy
reproject
photutils
```

You can also install them manually:

```bash
pip install numpy scipy pandas matplotlib astropy reproject photutils
```

---

## 5. Quick Start

```python
from ZTF_Pipeline import PipelineConfig, ZTFFolderPipeline, ZTFDifferencePipeline, LightCurveExtractor

# 1. Configure the pipeline
cfg = PipelineConfig()

# 2. Point to a folder of ZTF sciimg.fits files
pipe = ZTFFolderPipeline("ztf_data/664_c10_q4_zr/sciimg", config=cfg)

# 3. Build the deep reference image (static sky)
ref = pipe.build_reference(
    zp_target=26.0,
    seeing_target=3.0,
    max_frames=50,
    save_path="reference.fits",
    show_ref=True
)

# 4. Run difference imaging over a date range
diff_pipe = ZTFDifferencePipeline(pipe, ref)
diff_frames = diff_pipe.subtract_range("2020-01-01", "2020-06-01", save=True)

# 5. Extract the light curve at the transient position (pixel coords)
lc = LightCurveExtractor(diff_frames)
df = lc.extract_at(x=589, y=1414, r=5.0)

print(df.head())
# Columns: mjd | date | flux | flux_err | snr | zp
```

---

## 6. Detailed Pipeline Workflow

### 6.1 Data Acquisition – FITS Download from IRSA

**Notebook:** `sciimg_FITS_Request.ipynb`

Before running the pipeline, ZTF science images (`sciimg.fits`) must be downloaded from the IRSA archive. The `download_ztf_sciimg()` function queries the [IRSA IBE service](https://irsa.ipac.caltech.edu/ibe/) for images matching a specific ZTF field configuration over a given time window.

```python
from sciimg_FITS_Request import download_ztf_sciimg

download_ztf_sciimg(
    fields=664,
    ccds=10,
    qids=4,
    filters="zr",
    tmin="2019-01-01T00:00:00",
    tmax="2023-05-01T23:59:59",
    root_out="ztf_data"
)
```

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `fields`  | ZTF field ID (integer or list) |
| `ccds`    | CCD number (1–16) |
| `qids`    | Quadrant ID (1–4) |
| `filters` | Filter code: `"zr"`, `"zg"`, or `"zi"` |
| `tmin` / `tmax` | UTC date range (ISO format) |
| `root_out` | Output directory |

Files are organized automatically as: `ztf_data/{field}_c{ccd}_q{qid}_{filter}/sciimg/`

Existing files are skipped (safe to re-run). A typical 4-year baseline for one quadrant yields ~600 science images.

---

### 6.2 Configuration

All processing parameters are controlled through three dataclasses:

```python
from ZTF_Pipeline import PipelineConfig, MaskConfig, BackgroundConfig

cfg = PipelineConfig(
    mask=MaskConfig(
        detect_sigma=3.0,       # Source detection threshold (σ above background)
        smooth_sigma_pix=1.5,   # Pre-smoothing for source detection
        dilate_pix=8,           # Dilation radius around detected sources (pixels)
        edge_margin=10,         # Ignored border width (pixels)
        saturate_dilate_pix=3   # Extra dilation for saturated blooms
    ),
    bkg=BackgroundConfig(
        box_size=32,            # Block size for local background estimation
        filter_size=3,          # Smoothing filter on the background map
        sigma_clip=3.0          # Sigma-clipping for outlier rejection
    ),
    pixel_scale=1.01,           # ZTF pixel scale in arcsec/pixel
    dtype=np.float32            # Data type (float32 to conserve RAM)
)
```

---

### 6.3 Masking

**Class:** `SingleFrame`  
**Methods:** `mask_edges()`, `mask_saturation()`, `mask_sources_simple()`, `build_mask()`

Before any background estimation, a combined boolean mask is built to identify pixels that should not contribute to sky statistics. Three types of bad pixels are identified:

**Edge mask** — strips of `edge_margin` pixels around the CCD boundary are excluded to avoid readout artifacts.

**Saturation mask** — pixels at or above the `SATURATE` header value are flagged and dilated to cover the surrounding bloom.

**Source mask** — the image is first smoothed with a Gaussian (σ = `smooth_sigma_pix`) to suppress noise, then pixels exceeding `median + detect_sigma × σ_MAD` are flagged as sources and dilated by `dilate_pix` to cover PSF wings and halos.

> **Important design choice:** the mask is used *only* by the background estimator. Pixel values are never replaced by NaN during normal pipeline execution. This avoids introducing holes in difference images and ensures source flux is preserved for photometry.

```python
frame = SingleFrame("image.fits", cfg)
mask = frame.build_mask()  # boolean array, True = exclude from background
```

![Masking at two different sigma thresholds](docs/Mask_Test.png)

---

### 6.4 Background Estimation & Subtraction

**Method:** `SingleFrame.estimate_background()`, `SingleFrame.subtract_background()`

A 2D spatially varying background model is computed using `photutils.Background2D` with median statistics and sigma-clipping. The image is divided into blocks of `box_size × box_size` pixels; within each block, the source mask excludes bright objects and the robust median is computed. The resulting coarse map is then smoothed by a `filter_size × filter_size` median filter to produce a continuous background map.

```python
mask = frame.build_mask()
bkg_map = frame.estimate_background(mask=mask)
frame.subtract_background(bkg_map)
```

The `box_size` parameter controls the spatial resolution of the background model. Smaller values capture fine gradients but risk being contaminated by sources; larger values produce smoother but less accurate maps.

![Background map comparison for different box sizes](docs/Background2D_Test.png)

---

### 6.5 Photometric Rescaling

**Method:** `SingleFrame.rescale_to_zp()`

ZTF images from different epochs have slightly varying photometric zero points stored in the FITS header keyword `MAGZP`. To ensure consistent flux values across the time series, all images are rescaled to a common target zero point using the standard magnitude–flux relation:

```
flux_scaled = flux × 10^((ZP_target - ZP_original) / 2.5)
```

```python
frame.rescale_to_zp(zp_target=26.0)
```

This is applied to both science frames and the reference image so that the difference `science − reference` yields flux values in a consistent photometric system.

---

### 6.6 PSF Homogenization

**Method:** `SingleFrame.psf_homogenize_to()`

The seeing (atmospheric blur, expressed as FWHM in arcseconds) varies from night to night. Before subtraction, both the science and reference images must have matching PSF widths to avoid residuals from star profiles in the difference image.

The required convolution kernel is computed from the quadrature difference of the two FWHMs:

```
σ_conv = sqrt(FWHM_target² - FWHM_original²) / (2.3548 × pixel_scale)
```

The image with smaller FWHM is convolved with a Gaussian of this width. The image with larger FWHM is left unchanged.

```python
frame.psf_homogenize_to(seeing_target=3.0)  # degrade to 3.0 arcsec FWHM
```

![Effect of PSF homogenization on a star profile](docs/Smoothing_impact.png)

---

### 6.7 Astrometric Alignment

**Method:** `SingleFrame.reproject_to()`

All science images must share the exact same pixel grid (WCS, orientation, sampling) before stacking or subtraction. Each image is reprojected onto a reference WCS grid using `reproject.reproject_interp` with bilinear interpolation.

```python
# Set the master geometry from a reference file
pipe.set_target_from_file("reference_frame.fits")

# Reproject any frame to that geometry
frame.reproject_to(target_wcs=pipe.target_wcs, target_shape=pipe.target_shape)
```

By default the pipeline uses the WCS of the first file in the dataset as the master geometry. Pixels outside the WCS overlap receive NaN values, which are replaced by 0 in the final difference images (they carry no astrophysical information).

---

### 6.8 Reference Image Construction

**Class:** `ZTFFolderPipeline`  
**Method:** `build_reference()`

The reference image represents the static sky: the host galaxy, field stars, and any persistent background. It is built by median-stacking a set of well-selected science frames.

The procedure automatically:
1. Reads the `SEEING` keyword from each FITS header
2. Sorts all available images by seeing quality (best first)
3. Discards frames with `SEEING ≥ seeing_target`
4. Runs the full preprocessing sequence (alignment, masking, background subtraction, ZP rescaling, PSF homogenization) on each selected frame
5. Computes the median stack — this suppresses transient objects and cosmic rays

```python
ref = pipe.build_reference(
    zp_target=26.0,
    seeing_target=3.0,    # Only use images with FWHM < 3.0 arcsec
    max_frames=50,        # Cap at 50 best frames
    save_path="ref.fits",
    show_ref=True
)
```

You can inspect the seeing distribution of your dataset before choosing the threshold:

```python
stats = pipe.plot_seeing_hist(folder="ztf_data/664_c10_q4_zr/sciimg")
# Returns: {'N': 631, 'mean': 2.61, 'median': 2.55, 'std': 0.38, ...}
```

---

### 6.9 Difference Imaging

**Class:** `ZTFDifferencePipeline`  
**Method:** `subtract_range()`

For each science frame in the requested date range, the pipeline:

1. Preprocesses the science frame (alignment, masking, background, ZP rescaling)
2. Determines which image (science or reference) has the lower seeing
3. Convolves the sharper image to match the PSF of the blurrier one
4. Subtracts: `diff = science − reference`
5. Replaces NaN border pixels with 0
6. Saves the difference image as a FITS file in a `diffimg/` subdirectory

```python
diff_pipe = ZTFDifferencePipeline(folder_pipeline=pipe, reference_frame=ref)

diff_frames = diff_pipe.subtract_range(
    start_date="2020-01-01",
    end_date="2020-06-01",
    save=True,    # Save diff FITS to disk
    force=False   # Skip if already computed
)
```

The `force=False` option makes re-runs fast — already-computed difference images are loaded directly from disk.

---

### 6.10 Aperture Photometry & Light Curve Extraction

**Class:** `LightCurveExtractor`  
**Method:** `extract_at()`

Circular aperture photometry is performed on each difference image at a fixed pixel position `(x, y)` using `photutils.aperture_photometry`. The flux uncertainty is estimated from a local sky annulus (inner radius `r+2`, outer radius `r+12` pixels) using a robust MAD-based noise estimator, accounting for variable backgrounds beneath the transient.

```python
lc = LightCurveExtractor(diff_frames)
df = lc.extract_at(x=589.0, y=1414.0, r=5.0)

# df columns: mjd | date | flux | flux_err | snr | zp
df.to_csv("lightcurve_SN2020xyz.csv", index=False)
```

| Column | Description |
|--------|-------------|
| `mjd` | Modified Julian Date of observation |
| `date` | UTC date string |
| `flux` | Differential aperture flux (ADU) |
| `flux_err` | Flux uncertainty (local noise × √N_pixels) |
| `snr` | Signal-to-noise ratio |
| `zp` | Photometric zero point |

---

## 7. Visual Results

### 3D Source Detection
Threshold-based detection with Gaussian smoothing prior to masking. Sources above `median + N × σ_MAD` are identified and dilated to cover their halos.

![3D visualization of source detection](docs/Plot3D_SmoothingSigma.png)

### Masking Comparison
Two different sigma thresholds (σ=5.0 and σ=3.0) with their respective dilation radii. A lower threshold masks more sources at the cost of masking more sky area.

![Mask A and B comparison](docs/Mask_Test.png)

### PSF Homogenization
Before (left, seeing 2.42") and after (right, seeing degraded to 4.42") Gaussian convolution. Star profiles are broadened to match the reference PSF.

![PSF homogenization effect](docs/Smoothing_impact.png)

### Background Estimation
Background maps computed with different block sizes (4 to 64 pixels). Larger blocks produce smoother maps; smaller blocks capture local gradients but risk source contamination.

![Background 2D estimation test](docs/Background2D_Test.png)

### Light Curve Validation
Comparison of the light curve extracted by this pipeline (blue, `AstroTools`) against the official ZTF pipeline photometry (orange) for the transient `ZTF17aadlxmv`. Both pipelines recover the same supernova shape and peak brightness, validating the difference imaging and photometry implementation.

![Light curve comparison](docs/Comparison_of_ZTF17aadlxmv_Fluxes_3.png)

---

## 8. Class Reference

### `SingleFrame`
Represents a single ZTF FITS image. Provides all per-frame operations.

| Method | Description |
|--------|-------------|
| `__init__(fits_path, config)` | Load data, WCS, seeing, ZP, saturation level from header |
| `build_mask()` | Returns combined boolean mask (edges + saturation + sources) |
| `mask_edges()` | Boolean mask for CCD edge margins |
| `mask_saturation()` | Boolean mask for saturated pixels (dilated) |
| `mask_sources_simple()` | N-sigma thresholding for source detection |
| `estimate_background(mask)` | 2D background model via `photutils.Background2D` |
| `subtract_background(bkg_map)` | In-place background subtraction |
| `rescale_to_zp(zp_target)` | Photometric normalization to a target zero point |
| `psf_homogenize_to(seeing_target)` | Gaussian convolution to degrade PSF |
| `reproject_to(target_wcs, target_shape)` | WCS-based geometric alignment |
| `get_aperture_flux(coords, r)` | Circular aperture photometry at given pixel positions |
| `save(out_path)` | Save processed frame as FITS |
| `to_hdu()` | Returns an `astropy.io.fits.PrimaryHDU` with updated WCS header |
| `summary()` | Print key image properties (shape, ZP, seeing, filter, etc.) |

---

### `ZTFFolderPipeline`
Orchestrates processing of a full directory of FITS files.

| Method | Description |
|--------|-------------|
| `__init__(folder, config, pattern)` | Scan directory, collect file list |
| `set_target_from_file(fits_path)` | Set master WCS and shape from a given file |
| `plot_seeing_hist(folder)` | Plot and return seeing statistics for all files |
| `prepare_frame(fits_path, zp_target, seeing_target)` | Full preprocessing pipeline on one file |
| `build_reference(zp_target, seeing_target, ...)` | Build deep reference image by median stacking |

---

### `ZTFDifferencePipeline`
Handles difference image production over a time range.

| Method | Description |
|--------|-------------|
| `__init__(folder_pipeline, reference_frame)` | Initialize with a prepared folder pipeline and reference |
| `subtract_range(start_date, end_date, save, force)` | Produce difference images for all frames in date range |

---

### `LightCurveExtractor`
Extracts aperture photometry from a list of difference images.

| Method | Description |
|--------|-------------|
| `__init__(diff_frames)` | Initialize with a list of `SingleFrame` objects |
| `extract_at(x, y, r)` | Perform aperture photometry at `(x, y)` with radius `r` |

---

## 9. Configuration Reference

### `MaskConfig`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `detect_sigma` | `3.0` | Detection threshold in units of σ_MAD above median |
| `smooth_sigma_pix` | `1.5` | Gaussian smoothing σ (pixels) before thresholding |
| `dilate_pix` | `8` | Binary dilation radius for source mask (pixels) |
| `edge_margin` | `10` | Width of ignored CCD border (pixels) |
| `saturate_dilate_pix` | `3` | Extra dilation for saturated pixel blooms |

### `BackgroundConfig`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `box_size` | `32` | Block size for local background estimation (pixels) |
| `filter_size` | `3` | Smoothing filter size applied to background map |
| `sigma_clip` | `3.0` | Sigma-clipping threshold for outlier rejection |

### `PipelineConfig`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mask` | `MaskConfig()` | Masking configuration |
| `bkg` | `BackgroundConfig()` | Background configuration |
| `pixel_scale` | `1.01` | ZTF pixel scale in arcsec/pixel |
| `dtype` | `np.float32` | Data type for image arrays (float32 saves RAM) |

---

## 10. Validation & Results

The pipeline was validated against the official ZTF forced photometry service on the transient `ZTF17aadlxmv` (a supernova observed in the ZTF r-band, field 664, CCD 10, quadrant 4).

The extracted light curve recovers:
- The correct peak timing and amplitude (~65,000 ADU at ZP=30)
- The general decline shape of the supernova light curve
- Consistent flux levels at quiescence before and after the event

Residual discrepancies with the ZTF reference (particularly at late times) are attributed to differences in background estimation strategy and aperture definition, and are currently under investigation.

---

## 11. Dependencies

| Library | Version | Usage |
|---------|---------|-------|
| `numpy` | ≥1.24 | Array operations |
| `scipy` | ≥1.10 | Gaussian filtering, binary dilation |
| `pandas` | ≥2.0 | Light curve tables, date handling |
| `matplotlib` | ≥3.7 | Diagnostics and plots |
| `astropy` | ≥5.3 | FITS I/O, WCS, time conversion, tables |
| `reproject` | ≥0.11 | WCS-based image reprojection |
| `photutils` | ≥1.9 | Background estimation, aperture photometry |
| `requests` | ≥2.28 | IRSA download script |

---

## 12. Notes & Known Limitations

- **Windows path separators:** The FITS download script (`sciimg_FITS_Request.ipynb`) uses backslash separators in folder names. This may require adaptation on Linux/macOS.
- **Memory usage:** Processing full 3072×3072 pixel ZTF quadrant images is memory-intensive. Using `dtype=np.float32` (default) halves RAM usage compared to float64. Processing hundreds of frames may require 8–16 GB of RAM.
- **Seeing keyword:** The pipeline relies on the `SEEING` keyword in ZTF FITS headers. If absent, frames are assigned a default value of 999 arcsec and automatically excluded from the reference stack.
- **No optimal image subtraction (OIS/ZOGY):** The current subtraction is a direct pixel-by-pixel difference after Gaussian PSF matching. More advanced methods (e.g., ZOGY or Alard-Lutz) could reduce PSF-mismatch residuals in crowded fields.
- **Pixel coordinate input:** The `LightCurveExtractor.extract_at()` method currently takes pixel coordinates. A future improvement would accept (RA, Dec) directly and perform WCS-based coordinate conversion automatically.

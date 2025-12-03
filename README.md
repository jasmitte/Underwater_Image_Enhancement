# An Enhanced Multi-Stage Approach for Dehazing Underwater Images

This repository contains a Python implementation of the image enhancement pipeline described in the 2024 paper, **"An Enhanced Multi-Stage Approach for Dehazing Underwater Images"** by Murugan et al. The project faithfully implements the three-stage process outlined in the paper to correct for color distortion, low contrast, and haze in underwater photographs.

---
**Disclaimer:** This project is a re-implementation of the research work described in the referenced papers. The author of this repository is not affiliated with the original authors of the papers. This implementation represents a best-effort attempt to accurately reproduce the described methodology, but it is not guaranteed to be an exact or fully accurate replication of the authors' original work or results.
---

![Pipeline Visualization](figures/pipeline_visualization.png)  <!-- Placeholder for a pipeline visualization image -->

## Features

*   **Faithful Three-Stage Pipeline:** Implements the exact methodology from the An_Enhanced paper (CLAHE → ZID → Color Correction).
*   **Zero-Shot Dehazing:** Utilizes a Zero-shot Image Dehazing (ZID) model that optimizes on each image individually, requiring no pre-training on paired datasets.
*   **Correct Paper Implementation:** Preprocessing uses ONLY CLAHE (not multiple techniques), matching Section III.A of the paper.
*   **Zero-UMSIE Network Architectures:** J-Net, T-Net, and A-Net implemented as detailed in the referenced Zero-UMSIE framework.
*   **Comprehensive Post-Processing:** Color corrections (white balance, gamma, saturation) applied AFTER dehazing as specified in Section III.B.
*   **Flexible CLI:** Command-line interface for single images or batch processing of entire directories.
*   **Configurable:** All pipeline parameters easily adjustable through CLI or Python API.
*   **Quantitative Evaluation:** Supports PSNR/SSIM metric calculation with ground-truth reference images. 
## Methodology and Architecture

The enhancement process faithfully implements the three-stage workflow from the paper "An Enhanced Multi-Stage Approach for Dehazing Underwater Images" (Murugan et al., 2024). This implementation follows the exact methodology described in the paper, with network architectures based on the Zero-UMSIE framework referenced in the original work.

### Stage 1: Pre-processing (Section III.A)
As specified in the paper, this stage performs **ONLY** contrast enhancement to prepare the image for dehazing:

*   **Contrast Limited Adaptive Histogram Equalization (CLAHE):** Applied in LAB color space to enhance contrast while preserving color information

**Important:** The paper explicitly states to use "histogram equalization **or** CLAHE" - meaning only ONE contrast enhancement method. All other color corrections (white balance, gamma correction, saturation adjustment) are performed in Stage 3 (post-processing) **AFTER** dehazing, not before.

### Stage 2: Zero-Shot Image Dehazing (ZID) (Section III.A)
The core of the pipeline is the ZID model, which decomposes each underwater image into three fundamental components using the underwater imaging model:

**I(x) = J(x)·t(x) + A·(1 - t(x))**

Where:
- **I(x)** = Observed underwater image
- **J(x)** = Scene radiance (clean image)
- **t(x)** = Transmission map (3-channel, wavelength-dependent)
- **A** = Global background light (1-channel)

This is achieved by jointly optimizing three neural networks (J-Net, T-Net, and A-Net) using a composite loss function. The model is "zero-shot" because it optimizes on each input image individually without requiring pre-training on paired datasets.

#### Network Architectures (Based on Zero-UMSIE)

The An_Enhanced paper references these network structures, which are detailed in the Zero-UMSIE framework:

*   **J-Net (Scene Radiance Estimation Network):**
    - 5 convolutional layers (3×3 kernels, 64 channels each)
    - 4 instance normalization layers
    - 4 ReLU activation layers
    - Sigmoid output layer (normalizes to [0,1])
    - **No downsampling** to preserve fine details
    - **Output:** 3-channel RGB scene radiance

*   **T-Net (Transmission Map Estimation Network):**
    - Similar architecture to J-Net
    - **Output:** 3-channel RGB transmission map
    - Accounts for wavelength-selective attenuation underwater

*   **A-Net (Global Background Light Estimation Network):**
    - 6 convolutional layers
    - 5 LeakyReLU activation layers
    - Sigmoid output layer
    - Based on Retinex decomposition concept
    - **Output:** 1-channel grayscale global background light

#### Loss Functions

The loss function enforces physical and statistical priors about underwater image formation:
*   **Reconstruction Loss (`L_rec`):** MSE loss ensuring the dehazing equation holds
*   **Atmospheric Light Loss (`L_atm`):** Variational inference for global light estimation
*   **Dark Channel Prior Loss (`L_dark`):** Statistics-based loss for scene radiance
*   **Laplacian Regularization Loss (`L_reg`):** Smoothness prior for transmission and atmospheric light

### Stage 3: Post-processing (Section III.B - Color Correction)
After dehazing, this final stage applies color corrections to produce the final enhanced image. As specified in the An_Enhanced paper, the techniques include:

*   **White Balance Correction:** Neutralizes color casts by adjusting RGB channel means
*   **Color Enhancement:** Saturation adjustments to restore vibrant underwater colors
*   **Gamma Correction:** Final brightness/contrast adjustments
*   **CLAHE (Optional):** Additional contrast enhancement if needed

**Note:** These color corrections were intentionally excluded from Stage 1 (preprocessing) and are applied here AFTER dehazing, as specified in the paper's methodology.

## Installation

1.  Clone the repository:
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  Create and activate a Python virtual environment (recommended):
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  Install the package:
    ```bash
    pip install .
    ```

## Usage

### Command-Line Interface (CLI)

The easiest way to use the model is through the provided CLI.

**To enhance a single image:**

```bash
python -m zero_shot_dehaze.cli --input path/to/your/image.jpg --output outputs/
```

**To batch-process a directory of images:**

```bash
python -m zero_shot_dehaze.cli --input path/to/image/directory --output outputs/
```

**To evaluate with reference images:**

If you have ground-truth (clean) images, you can calculate PSNR and SSIM metrics by providing a reference path.

```bash
python -m zero_shot_dehaze.cli --input /path/to/raw_images --output outputs/ --reference /path/to/reference_images
```

### Key CLI Arguments

*   `--input`: Path to the input image or directory.
*   `--output`: Path to the output directory.
*   `--reference`: (Optional) Path to the ground-truth image or directory for metric calculation.
*   `--max-iterations`: Optimization steps for the ZID solver (default: 600). Reduce for faster previews.
*   `--device`: Force a specific device (e.g., `cpu`, `cuda`, `cuda:0`).
*   `--no-clahe`: Disable CLAHE in both pre and post-processing stages for a different aesthetic.

### Library Usage

You can also import and use the enhancement pipeline directly in your Python code.

```python
from pathlib import Path
from zero_shot_dehaze.pipeline import MultistageUnderwaterEnhancer, EnhancerConfig
from zero_shot_dehaze.zid import ZIDConfig
from zero_shot_dehaze.utils import load_image, save_image

# Configure the pipeline
config = EnhancerConfig()
config.zid.max_iterations = 400
config.zid.device = "cuda"  # or "cpu"

# Initialize the enhancer
enhancer = MultistageUnderwaterEnhancer(config)

# Load an image and enhance it
input_image_path = Path("input.jpg")
image_array = load_image(input_image_path)
enhanced_array = enhancer.enhance_array(image_array)

# Save the result
output_path = Path("outputs/enhanced_image.png")
output_path.parent.mkdir(exist_ok=True)
save_image(enhanced_array, output_path)

print(f"Enhanced image saved to {output_path}")
```

## Running Tests

A simple test suite is included to verify that the end-to-end pipeline executes correctly.

```bash
python -m pytest
```

## Citation

This project is an implementation of the work described in the following papers. Please consider citing the original authors if you use this code in your research.

**Primary Paper (Methodology):**
> T. K. Murugan, S. Sharma, A. Ganguly, A. Banerjee, and K. Kejriwal, "An Enhanced Multi-Stage Approach for Dehazing Underwater Images," *IEEE Access*, vol. 12, pp. 156803-156822, 2024. DOI: 10.1109/ACCESS.2024.3486456

**Network Architectures (J-Net, T-Net, A-Net):**
> T. Liu, K. Zhu, W. Cao, B. Shan, and F. Guo, "Zero-UMSIE: a zero-shot underwater multi-scale image enhancement method based on isomorphic features," *Optics Express*, vol. 32, no. 23, pp. 40398-40415, 2024. DOI: 10.1364/OE.538120
# An Enhanced Multi-Stage Approach for Dehazing Underwater Images

This repository contains a Python implementation of the image enhancement pipeline described in the 2024 paper, **"An Enhanced Multi-Stage Approach for Dehazing Underwater Images"** by Murugan et al. The project faithfully implements the three-stage process outlined in the paper to correct for color distortion, low contrast, and haze in underwater photographs.

![Pipeline Visualization](https://i.imgur.com/example.png)  <!-- Placeholder for a pipeline visualization image -->

## Features

*   **Multi-Stage Enhancement Pipeline:** Implements the full three-stage enhancement process for superior results.
*   **Zero-Shot Dehazing:** Utilizes a Zero-shot Image Dehazing (ZID) model that adapts to each image individually, requiring no prior training dataset.
*   **Faithful Implementation:** The core ZID loss functions, including the Laplacian smoothness regularization, are implemented as described in the paper.
*   **Comprehensive Pre- and Post-Processing:** Includes all the color and contrast correction techniques from the paper, such as CLAHE, gray-world balancing, and saturation boosting.
*   **Flexible CLI:** A command-line interface is provided to easily enhance single images or batch-process entire directories.
*   **Configurable:** All pipeline parameters are easily configurable through the CLI or via Python.
*   **Quantitative Evaluation:** Supports PSNR/SSIM metric calculation if ground-truth reference images are available.
no 
## Methodology and Architecture

The enhancement process mirrors the workflow from the paper, which is divided into three main stages. This implementation enhances the original paper's concepts by using deeper, more powerful neural network architectures for the core dehazing task.

### Stage 1: Pre-processing
This stage prepares the image for dehazing by performing initial color and contrast adjustments. The implemented techniques include:
*   Gray-World Balancing
*   Red Channel Restoration
*   Contrast Limited Adaptive Histogram Equalization (CLAHE)
*   Gamma Correction

### Stage 2: Zero-Shot Image Dehazing (ZID)
The core of the pipeline is the ZID model, which is optimized on a per-image basis to separate the hazy input into a clean image (`J`), a transmission map (`t`), and an atmospheric light estimate (`A`). This is achieved by jointly optimizing three neural networks (J-Net, T-Net, and A-Net) using a composite loss function.

The loss function is composed of several terms designed to enforce physical and statistical priors about the image formation process:
*   **Reconstruction Loss (`L_rec`):** An MSE loss that ensures the dehazing equation (`I = J*t + A*(1-t)`) holds true, forcing the model to learn the physics of the scene.
*   **Atmospheric Light Loss (`L_atm`):** Ensures the predicted atmospheric light is close to a prior estimated from the brightest pixels in the hazy image.
*   **Dark Channel Prior Loss (`L_dark`):** A statistics-based loss based on the assumption that most non-sky patches in natural images have at least one very dark color channel.
*   **Laplacian Regularization Loss (`L_reg`):** A smoothness prior applied to the transmission map and atmospheric light to prevent noisy outputs.

#### Architectural Details
The neural networks in this implementation have been designed to be significantly deeper and more complex to enhance their representational capacity.

*   **J-Net and T-Net (`UNet`):** These networks utilize a deeper `UNet` architecture. This model features an additional downsampling/upsampling level and includes **skip connections** between corresponding encoder and decoder stages. These connections facilitate the flow of fine-grained details from early layers to later layers, which is crucial for reconstructing high-fidelity images and preventing detail loss.

*   **A-Net (`AtmosphericNet`):** The atmospheric light estimation network incorporates `ResidualBlock`s into its feature extractor, making it deeper. This design provides the network with increased capacity to analyze the entire image and estimate a more accurate global atmospheric light color, while adhering to the model's assumption of a spatially constant light source.

### Stage 3: Post-processing
After dehazing, this final stage polishes the image to produce the final output. The techniques include:
*   White Balance Correction
*   Saturation Boosting
*   CLAHE
*   Auto-Contrast Adjustment

## Installation

1.  Clone the repository:
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  Create and activate a Python virtual environment:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Command-Line Interface (CLI)

The easiest way to use the model is through the provided CLI.

**To enhance a single image:**

```bash
python -m src.cli --input path/to/your/image.jpg --output outputs/
```

**To batch-process a directory of images:**

```bash
python -m src.cli --input path/to/image/directory --output outputs/
```

**To evaluate with reference images:**

If you have ground-truth (clean) images, you can calculate PSNR and SSIM metrics by providing a reference path.

```bash
python -m src.cli --input /path/to/raw_images --output outputs/ --reference /path/to/reference_images
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
from src.pipeline import MultistageUnderwaterEnhancer, EnhancerConfig
from src.zid import ZIDConfig
from src.utils import load_image, save_image

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

This project is an implementation of the work described in the following paper. Please consider citing the original authors if you use this code in your research.

> T. K. Murugan, S. Sharma, A. Ganguly, A. Banerjee, and K. Kejriwal, "An Enhanced Multi-Stage Approach for Dehazing Underwater Images," *IEEE Access*, 2024.

## Installation

1.  Clone the repository:
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  Create and activate a Python virtual environment:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Command-Line Interface (CLI)

The easiest way to use the model is through the provided CLI.

**To enhance a single image:**

```bash
python -m src.cli --input path/to/your/image.jpg --output outputs/
```

**To batch-process a directory of images:**

```bash
python -m src.cli --input path/to/image/directory --output outputs/
```

**To evaluate with reference images:**

If you have ground-truth (clean) images, you can calculate PSNR and SSIM metrics by providing a reference path.

```bash
python -m src.cli --input /path/to/raw_images --output outputs/ --reference /path/to/reference_images
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
from src.pipeline import MultistageUnderwaterEnhancer, EnhancerConfig
from src.zid import ZIDConfig
from src.utils import load_image, save_image

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

This project is an implementation of the work described in the following paper. Please consider citing the original authors if you use this code in your research.

> T. K. Murugan, S. Sharma, A. Ganguly, A. Banerjee, and K. Kejriwal, "An Enhanced Multi-Stage Approach for Dehazing Underwater Images," *IEEE Access*, 2024.
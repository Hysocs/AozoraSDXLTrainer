# Simplified SDXL Fine-Tuning Script

This repository contains a single-file Python script for fine-tuning Stable Diffusion XL (SDXL) models. It is designed for simplicity and efficiency, focusing on training cross-attention layers, feed-forward networks, and text embeddings while keeping other weights frozen.

The script includes automatic latent caching, dataset bucketing, and detailed verification steps to ensure a stable and correct training process.

## Features

-   **Single-File Simplicity**: All logic is contained in one Python script for easy management.
-   **Targeted Training**: Optimizes only specified parts of the UNet (attention, feed-forward) and text encoders (token embeddings), which is much faster than full fine-tuning.
-   **Efficient Optimizer**: Uses `bitsandbytes` 8-bit Adam optimizer (`Adafactor`) to reduce VRAM usage.
-   **Automatic Latent Caching**: Pre-computes and caches VAE latents to disk, dramatically speeding up the training loop by eliminating the need for repeated VAE encoding.
-   **Automatic Bucketing**: Scans your dataset and groups images into appropriate buckets to handle various aspect ratios efficiently.
-   **Resume Training**: Automatically finds and resumes from the latest saved checkpoint.
-   **Min-SNR Weighting**: Implements Min-SNR gamma weighting for improved training stability, especially on `v-prediction` models.
-   **Save Verification**: Includes multiple sanity checks and a detailed report to verify that the model weights are being updated correctly and saved properly.

## 1. Setup and Installation

### Prerequisites
-   An NVIDIA GPU with at least 12GB of VRAM (16GB+ recommended).
-   Python 3.10 or later.
-   CUDA Toolkit installed.

### Installation Steps
1.  **Clone the repository:**
```bash
git clone https://github.com/Hysocs/AozoraSDXLTrainer
cd https://github.com/Hysocs/AozoraSDXLTrainer
```

2.  **Create a virtual environment (recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3.  **Install the required Python packages:**
    The script depends on several libraries. Install them using pip:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install diffusers transformers safetensors bitsandbytes accelerate tqdm
```
    *Note: Make sure your `torch` and `bitsandbytes` versions are compatible with your CUDA installation.*

## 2. Data Preparation

The script recursively scans a root data directory for images and their corresponding caption files.

1.  **Create your dataset folder.** Inside this folder, you can have as many subfolders as you like.
2.  For each image, create a text file with the **exact same name** (but with a `.txt` extension) in the same directory. This text file should contain the caption for the image.

### Example Directory Structure

Your `INSTANCE_DATA_DIR` should be structured like this:

```
./DatasetV1/
├── project_A/
│   ├── image_001.png
│   ├── image_001.txt
│   ├── image_002.jpg
│   └── image_002.txt
│
├── project_B/
│   ├── another_image.webp
│   └── another_image.txt
│
└── character_sheets/
    ├── concept.png
    └── concept.txt
```
-   **Image Captions:** The script will read the caption from the `.txt` file. If a `.txt` file is not found, it will use the image's filename as a fallback caption (with underscores replaced by spaces).
-   **Supported Image Formats:** `.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`.

## 3. How to Use the Script

### Step 1: Place Your Base Model

Download your desired SDXL base model in `.safetensors` format and place it in the project's root directory. The script defaults to looking for `./MIDNIGHT_NAI-XL_vPredV1_Baked_VAE.safetensors`.

### Step 2: Configure the Training Job

Open the training script (`train_sdxl.py`) in a text editor. All user-configurable settings are located in the `CONFIGURATION` section at the top of the file.

```python
# ====================================================================================
# CONFIGURATION
# ====================================================================================
SINGLE_FILE_CHECKPOINT_PATH = "./MODEL_Vpred_or_eps.safetensors"
INSTANCE_DATA_DIR = "./DatasetV1/"
OUTPUT_DIR = Path("./sdxl_cross_attn_simplified_output_optimized")
FORCE_RECACHE_LATENTS = False

# --- Training Parameters ---
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 64
UNET_LEARNING_RATE = 3e-6
TEXT_ENCODER_LEARNING_RATE = 1.5e-6
MAX_TRAIN_STEPS = 15400
LR_WARMUP_PERCENT= 0.40

# --- Advanced Settings ---
RESUME_TRAINING = False
SAVE_EVERY_N_STEPS = 5000
# ... etc.
# ====================================================================================
```

Modify these values to match your setup and training goals.

### Step 3: Run the Script

Once configured, simply run the script from your terminal:

```bash
python train_sdxl.py
```

-   **First Run (Caching):** The first time you run the script, it will pre-compute and cache the image latents. This may take some time depending on the size of your dataset. You will see a progress bar for this process.
-   **Subsequent Runs (Training):** After caching is complete, the main training loop will begin. The script will print progress, loss, and learning rates to the console.

### Output Files

-   **Checkpoints**: Intermediate models and optimizer states are saved in the `OUTPUT_DIR/checkpoints/` directory every `SAVE_EVERY_N_STEPS`.
-   **Final Model**: The final trained model will be saved in the `OUTPUT_DIR` with a name like `YourBaseModel_trained_stepXXXX.safetensors`.

## Configuration Options Explained

| Parameter                       | Description                                                                                                                              |
| ------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `SINGLE_FILE_CHECKPOINT_PATH`   | Path to your base SDXL model in `.safetensors` format.                                                                                   |
| `INSTANCE_DATA_DIR`             | Path to the root folder containing your training images and captions.                                                                    |
| `OUTPUT_DIR`                    | Path to the directory where final models and checkpoints will be saved.                                                                  |
| `FORCE_RECACHE_LATENTS`         | Set to `True` to delete all existing latent caches and regenerate them. Useful if you've changed your dataset or bucketing settings.      |
| `BATCH_SIZE`                    | Number of images processed in a single forward/backward pass. Set to 1 if VRAM is limited.                                               |
| `GRADIENT_ACCUMULATION_STEPS`   | Number of steps to accumulate gradients before an optimizer step. **Effective Batch Size = `BATCH_SIZE` * `GRADIENT_ACCUMULATION_STEPS`**. |
| `UNET_LEARNING_RATE`            | The learning rate for the UNet parameters.                                                                                               |
| `TEXT_ENCODER_LEARNING_RATE`    | The learning rate for the text encoder token embeddings.                                                                                 |
| `MAX_TRAIN_STEPS`               | The total number of training steps to run.                                                                                               |
| `LR_SCHEDULER_TYPE`             | The learning rate scheduler. Can be `"cosine"` or `"linear"`.                                                                            |
| `LR_WARMUP_PERCENT`             | The percentage of `MAX_TRAIN_STEPS` to use for a learning rate warmup.                                                                   |
| `RESUME_TRAINING`               | Set to `True` to automatically find the latest checkpoint in `OUTPUT_DIR` and resume training.                                            |
| `SAVE_EVERY_N_STEPS`            | How often to save a full model checkpoint.                                                                                               |
| `MIXED_PRECISION`               | Use `"bfloat16"` (recommended for Ampere/Ada GPUs) or `"float16"` for training.                                                          |

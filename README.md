# Behavior Cloning (BC) Training on LIBERO Spatial

This project implements a native Behavior Cloning (BC) approach for training on LIBERO Spatial tasks using the `imitation` library. It provides a straightforward implementation for multi-task learning across all LIBERO Spatial subtasks.

## Overview

This implementation uses the `imitation` library's BC algorithm to train a policy that can handle all 10 subtasks in the LIBERO Spatial benchmark. Unlike the official LIBERO framework which uses Transformer-based architectures, this implementation uses a simpler approach suitable for CPU training.

## Project Structure

```
BC/
├── data/                          # Demonstration data directory
│   └── libero_spatial/           # LIBERO Spatial task demonstrations (HDF5 files)
├── models/                       # Trained model checkpoints
├── scripts/                      # Training and evaluation scripts
│   ├── train_libero_spatial.py  # Main BC training script (imitation library)
│   ├── train_libero_spatial_official.py  # LIBERO official framework training
│   └── evaluate_libero_spatial.py
├── LIBERO/                       # LIBERO framework (if cloned)
├── run_evaluate.sh               # Training script using imitation BC
├── run_train_libero_official.sh  # Training script using LIBERO framework
├── run_libero_bc.sh              # Alternative LIBERO training script
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Features

- **Native BC Implementation**: Uses `imitation` library's BC algorithm
- **Multi-task Learning**: Trains on all LIBERO Spatial subtasks simultaneously
- **CPU-friendly**: Simpler architecture suitable for CPU training
- **LIBERO Official Support**: Also includes scripts for LIBERO official framework

## Installation

### 1. Create Virtual Environment

```bash
python -m venv bc-libero-env
source bc-libero-env/bin/activate  # On Windows: bc-libero-env\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Setup LIBERO (Optional)

If you want to use the official LIBERO framework:

```bash
# Clone LIBERO repository
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git

# Install LIBERO
cd LIBERO
pip install -e .
cd ..
```

## Usage

### Training with Imitation Library (Native BC)

This is the main implementation using the `imitation` library:

```bash
# Activate environment
source bc-libero-env/bin/activate

# Run training
bash run_evaluate.sh

# Or directly:
python scripts/train_libero_spatial.py \
    --task libero_spatial \
    --data ./data/libero_spatial \
    --epochs 50 \
    --batch-size 32 \
    --lr 0.0001 \
    --device cpu \
    --use-libero-config
```

**Parameters:**
- `--task`: Task suite name (default: `libero_spatial`)
- `--data`: Path to demonstration data directory or file
- `--epochs`: Number of training epochs (default: 50)
- `--batch-size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 0.0001)
- `--device`: Device to use (`cpu` or `cuda`)
- `--use-libero-config`: Use LIBERO recommended config (AdamW + L2 regularization)
- `--use-scheduler`: Enable learning rate scheduler (limited support)
- `--use-augmentation`: Enable data augmentation (not supported in current version)

### Training with LIBERO Official Framework

For full features including data augmentation and learning rate scheduling:

```bash
bash run_train_libero_official.sh
```

This uses the official LIBERO framework with Transformer-based policies.

## Implementation Details

### Native BC Approach (`train_libero_spatial.py`)

- **Framework**: `imitation.algorithms.bc.BC`
- **Optimizer**: AdamW (with L2 regularization via `l2_weight`)
- **Policy**: Automatically selected by `imitation` library based on observation/action spaces
- **Data Format**: HDF5 files with LIBERO format (`data/demo_X/obs/` and `data/demo_X/actions/`)
- **Multi-task**: All subtasks are combined into a single training dataset

**Advantages:**
- Simpler implementation
- Faster on CPU (lighter model)
- Easy to understand and modify

**Limitations:**
- No built-in data augmentation
- Limited learning rate scheduler support
- Less feature-rich than official framework

### LIBERO Official Framework (`train_libero_spatial_official.py`)

- **Framework**: LIBERO's official training pipeline
- **Policies**: BCTransformerPolicy, BCRNNPolicy, or BCViLTPolicy
- **Features**: Full data augmentation, learning rate scheduling, etc.

**Advantages:**
- Full feature set
- Better performance potential
- Official support

**Limitations:**
- More complex
- Slower on CPU (Transformer models)
- Requires LIBERO framework setup

## Data Format

The training script expects LIBERO format HDF5 files:

```
data/libero_spatial/
├── task1_demo.hdf5
├── task2_demo.hdf5
└── ...

Each HDF5 file structure:
data/
  demo_0/
    obs/
      agentview_rgb: (T, H, W, 3)
      eye_in_hand_rgb: (T, H, W, 3)
      joint_states: (T, ...)
      gripper_states: (T, ...)
    actions: (T, action_dim)
  demo_1/
  ...
```

## Training Output

- **Models**: Saved to `models/bc_libero_spatial_final/`
- **Logs**: Training progress printed to console
- **Checkpoints**: Optional checkpoints during training

## Notes

1. **CPU Training**: The native BC implementation is more suitable for CPU training due to simpler architecture
2. **Data Augmentation**: Not supported in the native BC implementation; use official framework for augmentation
3. **Learning Rate Scheduler**: Limited support in native BC; official framework has full scheduler support
4. **Multi-task**: Both implementations support training on all LIBERO Spatial subtasks simultaneously

## Comparison

| Feature | Native BC (imitation) | LIBERO Official |
|---------|----------------------|-----------------|
| Framework | `imitation` library | LIBERO framework |
| Model Architecture | Auto-selected (simpler) | Transformer/RNN/ViLT |
| Data Augmentation | ❌ Not supported | ✅ Supported |
| Learning Rate Scheduler | ⚠️ Limited | ✅ Full support |
| CPU Training Speed | ✅ Faster | ⚠️ Slower |
| GPU Training | ✅ Supported | ✅ Supported |
| Multi-task Learning | ✅ Supported | ✅ Supported |

## References

- [LIBERO Project](https://libero-project.github.io)
- [Imitation Learning Library](https://imitation.readthedocs.io/)
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)

## License

This implementation is provided for research and educational purposes.

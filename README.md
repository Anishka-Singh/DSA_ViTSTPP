# NuwaDynamics Repository

## Overview
The NuwaDynamics repository focuses on video prediction and analysis using state-of-the-art deep learning methods. It includes implementations of various models, data loaders, metrics, and utilities to support research and development in spatiotemporal predictive learning tasks.

## Installation

To set up the repository, install the required packages:

```bash
pip install -r requirements.txt
```

Prepare the data: Ensure your datasets are in the correct format and place them in the `data` directory.

## Training Instructions

### Train NuwaDynamics
To train the NuwaDynamics model, use the following command:

```bash
python main_copy.py \
    --mode nuwa \
    --ex_name nuwa_training \
    --batch_size 64 \
    --val_batch_size 64 \
    --epochs 10 \
    --lr 0.00001
```

### Train ConvHawkes
To train the ConvHawkes model, use the following command:

```bash
python /home/ansingh/Nuwa_Hawkes/main_copy.py \
    --mode hawkes \
    --batch_size 8 \
    --val_batch_size 8 \
    --nuwa_checkpoint "path to checkpoint" \
    --hawkes_epochs 10 \
    --hawkes_lr 0.001
```

**Notes**:
- By default, the NuwaDynamics checkpoint is located at:
```bash
/output/simvp_nighttime_mask/Debug/checkpoints/checkpoint.pth
```

## Directory Structure
- `data/`: Place your datasets here, ensuring they follow the required format.
- `output/`: Contains training logs, checkpoints, and results.

Feel free to modify the training parameters and paths as needed to fit your use case.


## Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.


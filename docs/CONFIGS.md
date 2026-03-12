# Configuration Files

This project uses a configuration-driven approach similar to the `mmengine` paradigm. All settings for dataset processing, model architecture, training loops, and evaluation metrics are completely defined in Python dictionaries and assembled into a single training or evaluation pipeline.

There are generally two types of config files in this repository:

1. **Data Creation & Visualization Configs**
2. **Training & Evaluation Configs**

## Data Creation & Visualization Configs

These configuration files are often minimal and used primarily to guide the dataset creation tools (e.g., `tools/create_data.py`) or purely for visualization tools that render the dataset and bounding boxes.

- **Purpose:** To define how to load, filter, transform, and package raw data.
- **Key Fields:** `data_root`, `data_prefix`, `train_pipeline`, `train_dataloader`.
- **Example:** `projects/rawpose/configs/dca1000evm_default_config.py`. Notice that while it has a `model` dict, it lacks training loops (`train_cfg`), validation dataloaders (`val_dataloader`), or metric definitions (`metric`).

## Training & Evaluation Configs

These are complete experiment configurations designed to be passed to `tools/train.py` or `tools/test.py`.

- **Purpose:** To define the entire end-to-end training and evaluation pipeline.
- **Key Fields Required:**
  - `model`: Defines the complete PyTorch architecture.
  - `train_dataloader`: Dictionary specifying the training dataset, batch size, and the `train_pipeline`.
  - `val_dataloader` & `test_dataloader`: Dictionaries specifying the validation/test datasets and `val_pipeline`/`test_pipeline`.
  - `optimizer_cfg` (or `optim_wrapper`): Defines the optimizer (e.g., AdamW, SGD) and learning rate.
  - `train_cfg`: Specifies the training loop (e.g., `dict(type='EpochBasedTrainLoop', max_epochs=50, val_interval=5)`).
  - `val_cfg` & `test_cfg`: Specifies the validation and test evaluation loops.
  - `metric`: Defines how the model's performance will be quantified (e.g., `SimpleGTPredAnalyzer` or `SoftDTW`).

- **Examples:**
  - `projects/mmmesh/configs/mmmesh-final_f32p64ag3df10r10-_b256_e100_gaming-S15OTBP9VLY05.py`: A very complete, complex training config.
  - `projects/rawpose/configs/rawpose_training_example.py`: A dedicated training config template for custom `rawpose` training.

## How to Create Your Own Training Config

If you want to train a model on your own dataset, the easiest path is to use the inheritance system:

1. Create a new file (e.g., `my_custom_training_config.py`).
2. Point its `_base_` to a smaller base configuration to inherit defaults, or just copy the entirety of `projects/rawpose/configs/rawpose_training_example.py` and modify it.
3. Update the `data_root` to point to your dataset location and modify the filenames (`train_info`, `val_info`, etc.).
4. Adjust the `batch_size`, `max_epochs` in `train_cfg`, and `optimizer_cfg` to match your VRAM limits and training schedule.

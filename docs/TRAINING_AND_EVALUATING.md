## Training and Evaluating Models

The training and evaluation pipelines follow a workflow similar to the mmengine paradigm. Both processes are configuration-driven and can be launched using the provided command-line tools.

### Training

To train a model, run:
```bash
python tools/train.py ${CONFIG_FILE}
```
where `${CONFIG_FILE}` specifies the full training configuration, including model architecture, dataset settings, and optimization parameters.

For a detailed explanation of how configuration files work, the difference between data creation configs and training configs, and the essential fields required for training configs, please see [CONFIGS.md](CONFIGS.md). 

For a complete working training configuration template that you can modify for your custom dataset, check: `projects/rawpose/configs/rawpose_training_example.py`.

### Evaluation

To evaluate a trained model, run:
```bash
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE}
```
where:
- `${CONFIG_FILE}` is the same configuration used during training
- `${CHECKPOINT_FILE}` points to the saved model weights to be evaluated

### Examples

Example configurations and usage patterns can be found under the `projects/` directory. These examples demonstrate how to define complete training and evaluation pipelines for different experimental setups.
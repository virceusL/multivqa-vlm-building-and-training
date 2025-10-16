# Multi-Image SFT Training and Testing

Build VLM based on SigLIP and Qwen2.5-0.5B, pre-train the fusion layer and do instruction-tuning with LoRA on Multi-VQA task.

## Repository structure

- [model.py](model.py) — model definitions and utilities.
- [train.py](train.py) — main training script.
- [sft_train_multi_images.py](sft_train_multi_images.py) — SFT training script for multi-image data.
- [test_multi_images.py](test_multi_images.py) — testing/evaluation script for multi-image inputs.
- [requirements.txt](requirements.txt) — Python dependencies.
- [pre_output/](pre_output/) — directory for intermediate/pretrained outputs.
- [sft_output/](sft_output/) — directory for SFT training outputs and checkpoints.

## Requirements

Install dependencies:

```
pip install -r requirements.txt
```

## Usage

- Train base model: ```python train.py```
- Run supervised fine-tuning on multi-image data: ```python sft_train_multi_images.py```
- Run tests: ```python test_multi_images.py```

Adjust dataset paths and hyperparameters inside the corresponding scripts as needed.


## Files of interest

- Model implementation: `model.py`
- Base training loop: `train.py`
- SFT on multi-image inputs: `sft_train_multi_images.py`
- Multi-VQA Test: `test_multi_images.py`
- Dependencies: `requirements.txt`





## Output
`pre_output/` and `sft_output/`

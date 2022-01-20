# Training
- Usage: `python train.py`

# Inference
- Usage examples:
    - `python inference.py --data_size 32 --batch_size 8 --partition test --random --checkpoint_path ~/PROJs/NeuralCompression_results/checkpoints/ --epoch 440 --save_path inference_results --half`

# Torch-script pretrained model
- Usage examples:
    - `python neuralcompress/utils/bcae_scriptor.py --checkpoint_path ~/PROJs/NeuralCompression_results/checkpoints/ --epoch 440 --save_path torchscript/`

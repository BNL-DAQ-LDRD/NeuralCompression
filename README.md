# Training
- Usage: `python train.py`
One can modified the parameters in `train.py`

When running the wavelet machine, please use
> `data_path = '/data/datasets/sphenix/highest_framedata_3d/outer'`


# Inference
- Usage examples:
    - `python inference.py --data_size 32 --batch_size 8 --partition test --random --checkpoint_path ./checkpoints/ --epoch 440 --save_path inference_results --half`

- parameters:
    - `--data_path`: The path to data.
    - `--device`:    Choose from {cuda,cpu}. The device to run the inference | default=cuda.
    - `--data_size`: Number of frames to load | default=1.
    - `--batch_size`: Batch size | default=1.
    - `--partition`: Choose from {train,valid,test} partition from which to load the data | default=test.
    - `--random`: Whether to get a random sample.
    - `--checkpoint_path`: The path to the pretrained checkpoints.
    - `--epoch`: The epoch of the pretrained checkpoints to load.
    - `--save_path`: The path to save output tensor.
    - `--half`: Whether to save the output with half precision.
    - `--prefix`: Output file prefix | default=output.


# Torch-script pretrained model for C++
- Usage examples:
    - `python neuralcompress/utils/bcae_scriptor.py --checkpoint_path ~/PROJs/NeuralCompression_results/checkpoints/ --epoch 440 --save_path torchscript/`
- parameters:
  - `--checkpoint_path`: The path to the checkpoints.
  - `--epoch`: The epoch of the pretrained checkpoints to load.
  - `--save_path`: The path to save the scripted encoder and decoder.
  - `--prefix`: Prefix to the filename of the scripted encoder and decoder | default=bcae.


# TO-DO:
- [ ] Add `kwargs` to bcae_trainer so it can accept parameters to the base class;
- [ ] Add the function to locate biggest epoch so that when `--epoch` is not given, the script can locate the most trained modeles.

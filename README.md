# Training
- Usage: `python train.py`
One can modified the parameters in `train.py`

When running the wavelet machine, please use
> `data_path = '/data/datasets/sphenix/highest_framedata_3d/outer'`


# Torch-script pretrained model for C++
- Usage examples:
    - `python neuralcompress/utils/bcae_scriptor.py --checkpoint_path ~/PROJs/NeuralCompression_results/checkpoints/ --epoch 440 --save_path torchscript/`
- parameters:
  - `--checkpoint_path`: The path to the checkpoints.
  - `--epoch`: The epoch of the pretrained checkpoints to load.
  - `--save_path`: The path to save the scripted encoder and decoder.
  - `--prefix`: Prefix to the filename of the scripted encoder and decoder | default=bcae.

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


# GPU benchmarking
## Usage example:
    - python benchmark/gpu_inference.py ~/PROJs/NeuralCompression_results/checkpoints/encoder_0440.pt --with_loader --data_root /data/datasets/sphenix/highest_framedata_3d/outer/ --num_workers 8 --half_precision
    - python benchmark/gpu_inference.py ~/PROJs/NeuralCompression_results/checkpoints/encoder_0440.pt --with_loader --data_root /data/datasets/sphenix/highest_framedata_3d/outer/ --num_workers 8 --half_precision --data_size 1024 --batch_size 8
    - python benchmark/gpu_inference.py ~/PROJs/NeuralCompression_results/checkpoints/encoder_0440.pt --num_workers 8 --half_precision --data_size 1024 --batch_size 8

## parameters:
- positional arguments:
  - `checkpoint`:          The path to the encoder pt file.

- optional arguments:
  - `--num_runs`:        Number of runs to calculate the run-time. (default=10)
  - `--benchmark`:       If used, set torch.backends.cudnn.benchmark to True.
  - `--with_loader`:     If used, use TPC dataloader. Use randomly generated data if otherwise
  - `--data_root`:       Path to data, when loading data with TPC dataloader (--with_loader). (default=None)
  - `--data_size`:       Number of frames to load. (default=1)
  - `--batch_size`:      Batch size. (default=1)
  - `--pin_memory`:      If used, the dataloader will copy Tensors into CUDA pinned memory before returning them.
  - `--num_workers`:     Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process. (default=0)
  - `--prefetch_factor`: Number of samples loaded in advance by each worker. (default=2)
  - `--result_fname`:    Result filename. (default=result.csv)
  - `--half_precision`:  If used, run inference with half precision (float16)

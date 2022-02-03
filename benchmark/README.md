# Inference

## Usage example:
    - python gpu_inference.py ~/PROJs/NeuralCompression_results/checkpoints/encoder_0440.pt --with_loader --data_root /data/datasets/sphenix/highest_framedata_3d/outer/ --num_workers 8 --half_precision
    - python gpu_inference.py ~/PROJs/NeuralCompression_results/checkpoints/encoder_0440.pt --with_loader --data_root /data/datasets/sphenix/highest_framedata_3d/outer/ --num_workers 8 --half_precision --data_size 1024 --batch_size 8
    - python gpu_inference.py ~/PROJs/NeuralCompression_results/checkpoints/encoder_0440.pt --num_workers 8 --half_precision --data_size 1024 --batch_size 8

## parameters:
- positional arguments:
  `checkpoint`:          The path to the encoder pt file.

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

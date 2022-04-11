## Test

- A sample time projection chamber (TPC) frame data are prepared in `./data` for testing. 
- First to uncompress sample data tar ball to `npy` files in `./data` with `tar xfvz sample_data.tar.gz`.
- A pretrained encoder and decoder `.pth` files are in `./checkpoints`.
- To run the pretrained models on sample data: `python test.py`.
- The output will be saved to the folder `./results`.

## Training

- Usage: `python train.py`
One can modified the parameters inside `train.py`.

## Torch-script pretrained model for C++

- Install `neuralcompressor`: `python setup.py develop --user`
- Usage examples:
    - `python neuralcompress/utils/bcae_scriptor.py --checkpoint_path checkpoints --epoch 2000 --save_path torchscript/`
- Parameters:
  - `--checkpoint_path`: The path to the checkpoints.
  - `--epoch`: The epoch of the pretrained checkpoints to load.
  - `--save_path`: The path to save the scripted encoder and decoder.
  - `--prefix`: Prefix to the filename of the scripted encoder and decoder | default=bcae.

## Inference

Produce compressed codes of each input TPC frame.

- Usage examples:
    - `python inference.py --data_size 8 --batch_size 4 --partition test --random --checkpoint_path ./checkpoints/ --epoch 2000 --save_path inference_results --half`

- Parameters:
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


## GPU benchmarking

- Usage example:
    - run on dummy data without data loader: `python benchmark/gpu_inference.py checkpoints/encoder_2000.pt`.
    - run on real data with data location and loader: `python benchmark/gpu_inference checkpoints/encoder_2000.pt --with_loader --data_root ./data`.
    - run with half-precision and adjust data loading workers with flags `--half_precision` and `--num_workers 8`.

- Parameters:
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

## Reference

```
@INPROCEEDINGS{huang2021bcae,
    author={Huang, Yi and Ren, Yihui and Yoo, Shinjae and Huang, Jin},
    booktitle={2021 20th IEEE International Conference on Machine Learning and Applications (ICMLA)},
    title={Efficient Data Compression for 3D Sparse TPC via Bicephalous Convolutional Autoencoder},
    year={2021},
    volume={},
    number={},
    pages={1094-1099},
    doi={10.1109/ICMLA52953.2021.00179}
}
```

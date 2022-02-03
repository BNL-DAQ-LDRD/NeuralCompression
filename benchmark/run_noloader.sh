#! bin/bash
for bs in 2 4 8 16 32 64 128
do
    cmd="python inference_time_noloader.py --data_size 1024 --batch_size ${bs} --benchmark --result_fname results/noloader_bm_${bs}.csv"
    echo $cmd
    $cmd
done

for bs in 2 4 8 16 32 64 128
do
    cmd="python inference_time_noloader.py --data_size 1024 --batch_size ${bs} --result_fname results/noloader_${bs}.csv"
    echo $cmd
    $cmd
done
